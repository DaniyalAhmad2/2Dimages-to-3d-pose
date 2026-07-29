"""Build a rigged, human-proportioned character from the MakeHuman base mesh.

    blender --background --python tools/build_character_from_basemesh.py -- \
            --obj base.obj --out pose3d/assets/character.blend [--faces 3500]

The MakeHuman base mesh (CC0, released September 2020 by Data Collection AB) is
anatomically proportioned and carries "joint-*" helper cubes marking the centre
of every joint. We read the joint centres from those helpers, build a skeleton
with the bone names the app's role table expects, decimate the body for the live
viewport, and bind with automatic weights.

Proportions therefore come from real anthropometry rather than from an artist's
stylisation, which is what lets fixed-length retargeting put the character's
knees and elbows where the subject's actually are.
"""
import argparse
import sys

import bpy
from mathutils import Vector

# role -> (joint helper for the head, joint helper for the tail). Left side; the
# right side is mirrored automatically.
CHAIN = [
    ("hips",        "joint-pelvis",      "spine-lo"),
    ("spine",       "spine-lo",          "spine-mid"),
    ("chest",       "spine-mid",         "joint-neck"),
    ("neck",        "joint-neck",        "joint-head"),
    ("head",        "joint-head",        "head-top"),
]
SIDE_CHAIN = [
    ("clavicle.%s",  "joint-%s-clavicle", "joint-%s-shoulder"),
    ("upper_arm.%s", "joint-%s-shoulder", "joint-%s-elbow"),
    ("forearm.%s",   "joint-%s-elbow",    "joint-%s-hand"),
    ("hand.%s",      "joint-%s-hand",     "joint-%s-hand-2"),
    ("thigh.%s",     "joint-%s-upper-leg", "joint-%s-knee"),
    ("shin.%s",      "joint-%s-knee",     "joint-%s-ankle"),
    ("foot.%s",      "joint-%s-ankle",    "joint-%s-foot-1"),
]
PARENT = {
    "spine": "hips", "chest": "spine", "neck": "chest", "head": "neck",
    "clavicle.L": "chest", "clavicle.R": "chest",
    "upper_arm.L": "clavicle.L", "upper_arm.R": "clavicle.R",
    "forearm.L": "upper_arm.L", "forearm.R": "upper_arm.R",
    "hand.L": "forearm.L", "hand.R": "forearm.R",
    "thigh.L": "hips", "thigh.R": "hips",
    "shin.L": "thigh.L", "shin.R": "thigh.R",
    "foot.L": "shin.L", "foot.R": "shin.R",
}
# Chains whose child head sits exactly on the parent tail.
CONNECT = {"spine", "chest", "neck", "head", "forearm.L", "forearm.R",
           "hand.L", "hand.R", "shin.L", "shin.R", "foot.L", "foot.R"}

# A perfectly straight limb gives the IK no way to choose a bend direction, and
# a real knee/elbow is not straight anyway. Nudge them by this fraction of the
# limb's length: knees forward, elbows back.
BEND = 0.035


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--faces", type=int, default=3500)
    return p.parse_args(argv)


def main():
    args = parse_args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # one object per OBJ group, so the joint helper cubes stay identifiable
    bpy.ops.wm.obj_import(filepath=args.obj, use_split_groups=True)
    objs = {o.name: o for o in bpy.context.scene.objects if o.type == "MESH"}

    def centroid(name):
        o = objs.get(name)
        if o is None:
            return None
        vs = [o.matrix_world @ v.co for v in o.data.vertices]
        return sum(vs, Vector()) / len(vs)

    J = {n: centroid(n) for n in objs if n.startswith("joint-")}
    missing = [n for n in ("joint-pelvis", "joint-neck", "joint-head",
                           "joint-l-shoulder", "joint-l-elbow", "joint-l-hand",
                           "joint-l-upper-leg", "joint-l-knee", "joint-l-ankle")
               if n not in J]
    if missing:
        raise SystemExit(f"base mesh is missing joint helpers: {missing}")

    body = objs.get("body")
    if body is None:
        raise SystemExit("no 'body' group in the OBJ")

    # spine helpers are numbered but their order along the body varies; sort by
    # height so 'lo' and 'mid' mean what they say
    spine = sorted((n for n in J if n.startswith("joint-spine")),
                   key=lambda n: J[n].z)
    J["spine-lo"] = J[spine[0]]
    J["spine-mid"] = J[spine[len(spine) // 2]]
    # top of the skull, for the head bone's tail
    head_top = max((body.matrix_world @ v.co for v in body.data.vertices),
                   key=lambda p: p.z)
    J["head-top"] = Vector((J["joint-head"].x, J["joint-head"].y, head_top.z))

    # anatomical bend so the IK can pick a side
    for s in ("l", "r"):
        for mid, a, b, sign in ((f"joint-{s}-knee", f"joint-{s}-upper-leg",
                                 f"joint-{s}-ankle", -1.0),
                                (f"joint-{s}-elbow", f"joint-{s}-shoulder",
                                 f"joint-{s}-hand", +1.0)):
            if mid in J and a in J and b in J:
                span = (J[b] - J[a]).length
                J[mid] = J[mid] + Vector((0.0, sign * BEND * span, 0.0))

    # keep only the body; the helpers were only ever measurement scaffolding
    bpy.ops.object.select_all(action="DESELECT")
    for name, o in objs.items():
        if o is not body:
            o.select_set(True)
    bpy.ops.object.delete()

    # decimate for the live viewport: the app skins this in numpy every frame
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    tris = sum(len(p.vertices) - 2 for p in body.data.polygons)
    if tris > args.faces:
        d = body.modifiers.new("dec", "DECIMATE")
        d.ratio = args.faces / tris
        bpy.ops.object.modifier_apply(modifier=d.name)
    print(f"body: {tris} -> "
          f"{sum(len(p.vertices) - 2 for p in body.data.polygons)} triangles")

    # --- skeleton ---
    arm_data = bpy.data.armatures.new("metarig")
    arm = bpy.data.objects.new("metarig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")

    specs = list(CHAIN)
    for tmpl, h, t in SIDE_CHAIN:
        for s, S in (("l", "L"), ("r", "R")):
            specs.append((tmpl % S, h % s if "%s" in h else h,
                          t % s if "%s" in t else t))
    made = {}
    for name, hk, tk in specs:
        if hk not in J or tk not in J:
            print(f"  skipping {name}: no {hk if hk not in J else tk}")
            continue
        eb = arm_data.edit_bones.new(name)
        eb.head, eb.tail = J[hk], J[tk]
        if (eb.tail - eb.head).length < 1e-4:
            eb.tail = eb.head + Vector((0, 0, 0.01))
        made[name] = eb
    for name, eb in made.items():
        p = PARENT.get(name)
        if p in made:
            eb.parent = made[p]
            eb.use_connect = name in CONNECT and (eb.head - made[p].tail).length < 1e-5
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"skeleton: {len(made)} bones")

    # --- bind ---
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")

    unweighted = sum(1 for v in body.data.vertices if not v.groups)
    print(f"vertices with no weight: {unweighted}")

    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
