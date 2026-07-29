"""Give a rigged character human limb proportions, once, permanently.

Runs inside Blender:

    blender --background <in.blend> --python tools/reproportion_character.py -- \
            --out <out.blend> [--report]

Fixed-length retargeting can only put the character's knees and elbows where the
subject's are if the character is shaped like a person. The bundled rig is
stylised — its thigh is 1.04 against a 1.54 shin, where a human's are about
equal — so the knee lands ~12% of body height out no matter how good the solver
is.

This rebalances each limb: the thigh/shank and upper-arm/forearm split is moved
to human ratios while the TOTAL length of each limb is preserved, so the
character's height, reach and silhouette barely move. The change is baked into
the rest pose and the mesh, so it is a property of the character rather than
something applied per subject at runtime — nothing deforms while it animates.

Re-bake afterwards with tools/bake_character.py.
"""
import argparse
import sys

import bpy

# Drillis & Contini segment ratios. Thigh and shank are near enough equal; the
# upper arm is about 1.27x the forearm.
TARGET_RATIO = {("thigh.L", "shin.L"): 1.00, ("thigh.R", "shin.R"): 1.00,
                ("upper_arm.L", "forearm.L"): 1.27,
                ("upper_arm.R", "forearm.R"): 1.27}

# Hip joint height as a fraction of stature. Get this wrong and the figure
# reads as long-bodied and short-legged however good the limb ratios are.
HIP_HEIGHT = 0.53


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--report", action="store_true")
    p.add_argument("--balance-trunk", action="store_true",
                   help="also shorten the trunk so the hip joint sits at "
                        "HIP_HEIGHT of stature. A rig whose body above the hip "
                        "is too long reads as long-bodied and short-legged even "
                        "when its individual limb lengths are correct.")
    return p.parse_args(argv)


def find_rig():
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit("no armature in this .blend")
    arm = max(arms, key=lambda o: len(o.data.bones))
    meshes = [o for o in bpy.data.objects
              if o.type == "MESH" and o.parent is arm
              and any(m.type == "ARMATURE" for m in o.modifiers)
              and any(v.groups for v in o.data.vertices)]
    return arm, meshes


def bone_len(arm, name):
    b = arm.data.bones[name]
    return (b.tail_local - b.head_local).length


def plan_scales(arm):
    """Per-bone length multiplier that hits the human ratio without changing
    the limb's overall length."""
    out = {}
    for (upper, lower), ratio in TARGET_RATIO.items():
        if upper not in arm.data.bones or lower not in arm.data.bones:
            continue
        lu, ll = bone_len(arm, upper), bone_len(arm, lower)
        total = lu + ll
        want_u = total * ratio / (1.0 + ratio)
        want_l = total - want_u
        out[upper] = want_u / lu
        out[lower] = want_l / ll
    return out


def main():
    args = parse_args()
    arm, meshes = find_rig()
    if not meshes:
        raise SystemExit("no skinned mesh found")

    scales = plan_scales(arm)

    if args.balance_trunk:
        # World space throughout: mesh verts are object-local while bone heads
        # are armature-local, and mixing the two silently produces nonsense.
        meshes_z = [(m.matrix_world @ v.co).z
                    for m in meshes for v in m.data.vertices]
        floor, crown = min(meshes_z), max(meshes_z)
        hip = (arm.matrix_world @ arm.data.bones["thigh.L"].head_local).z
        h0 = hip - floor                      # hip height, unchanged by this
        above = crown - hip                   # what we scale
        f = HIP_HEIGHT
        # want above*t / (above*t + h0) == 1 - f
        t = h0 * (1 - f) / (above * f) if above > 1e-9 else 1.0
        t = float(min(max(t, 0.5), 2.0))
        print(f"hip at {100*h0/(crown-floor):.1f}% of stature (human {100*f:.0f}%): "
              f"scaling the trunk by {t:.3f}")
        for name in ("hips", "spine", "chest"):
            if name in arm.data.bones:
                scales[name] = scales.get(name, 1.0) * t

    print("rebalancing limbs (total length preserved):")
    for name, k in sorted(scales.items()):
        print(f"  {name:14s} {bone_len(arm, name):.3f} -> "
              f"{bone_len(arm, name) * k:.3f}   ({k:+.3f}x)")

    bpy.context.view_layer.objects.active = arm
    if arm.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    # A child must not inherit its parent's length change, or the shin would be
    # rescaled by the thigh on top of its own correction. Position still
    # follows, which is what moves the knee.
    for b in arm.data.bones:
        b.inherit_scale = "NONE"

    for pb in arm.pose.bones:
        k = scales.get(pb.name)
        if k:
            pb.scale = (1.0, k, 1.0)          # bones run along local +Y
    bpy.context.view_layer.update()

    # Bake the deformed shape into the mesh. Reading the evaluated mesh with
    # only the armature enabled keeps the vertex count (and therefore the skin
    # weights) intact, which applying a subsurf would not.
    dg = bpy.context.evaluated_depsgraph_get()
    for m in meshes:
        disabled = []
        for mod in m.modifiers:
            if mod.type != "ARMATURE" and mod.show_viewport:
                mod.show_viewport = False
                disabled.append(mod)
        dg = bpy.context.evaluated_depsgraph_get()
        coords = [v.co.copy() for v in m.evaluated_get(dg).data.vertices]
        if len(coords) != len(m.data.vertices):
            raise SystemExit(f"{m.name}: vertex count changed during evaluation")
        for i, v in enumerate(m.data.vertices):
            v.co = coords[i]
        for mod in disabled:
            mod.show_viewport = True

    # Make the new shape the rest pose.
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode="OBJECT")

    for b in arm.data.bones:                  # back to normal inheritance
        b.inherit_scale = "FULL"

    if args.report:
        for (upper, lower) in TARGET_RATIO:
            if upper in arm.data.bones:
                print(f"  {upper} : {lower} = "
                      f"{bone_len(arm, upper) / bone_len(arm, lower):.3f}")

    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
