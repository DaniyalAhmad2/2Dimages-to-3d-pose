"""Bake a rigged .blend into the character.npz the app poses at runtime.

Runs inside Blender:

    blender --background <rig.blend> --python tools/bake_character.py -- \
            --out pose3d/assets/character.npz [--roles roles.json] \
            [--strip-helpers] [--report]

The app never opens the .blend for posing — it reads this npz — so the two must
be baked from the same file or the 3D view and the Blender export silently
disagree. `blend_sha256` in the npz is what lets that be detected.

Everything is stored in WORLD space with the armature transform applied, which
is what makes the runtime's rotation-only FK valid.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

# Reuse the app's role aliases so naming knowledge lives in exactly one place.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from pose3d.geometry.character import _ROLE_ALIASES
except Exception:                                    # running outside the repo
    _ROLE_ALIASES = {}

MAX_INFLUENCES = 4

# Segment length as a fraction of stature (Drillis & Contini), used by --report
# to say whether a candidate rig is shaped like a person.
_STATURE_FRACTION = {
    "thigh": 0.245, "shank": 0.246, "upper_arm": 0.186, "forearm": 0.146,
}


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--roles", help="JSON {role: bone name} overriding auto-detection")
    p.add_argument("--strip-helpers", action="store_true",
                   help="drop non-deform bones carrying no skin weight")
    p.add_argument("--report", action="store_true",
                   help="print the anthropometric check and fail if it does not pass")
    p.add_argument("--source", default="", help="provenance/licence note")
    return p.parse_args(argv)


def find_rig():
    """Same choice the exporter makes: the armature with the most bones."""
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit("no armature in this .blend")
    arm = max(arms, key=lambda o: len(o.data.bones))
    # Only SKINNED meshes belong in the npz. Props can be parented to the
    # armature object without being weighted to any bone (the bundled rig does
    # this with the eyes); baking those would collapse them onto the origin,
    # since linear blend skinning has no bone to place them with.
    meshes = []
    for o in bpy.data.objects:
        if o.type != "MESH" or o.parent is not arm or not o.data.vertices:
            continue
        skinned = (any(m.type == "ARMATURE" for m in o.modifiers)
                   and any(v.groups for v in o.data.vertices))
        if skinned:
            meshes.append(o)
        else:
            print(f"skipping '{o.name}' ({len(o.data.vertices)} verts): "
                  f"parented to the rig but not skinned to it")
    if not meshes:
        raise SystemExit(f"no skinned meshes found under armature '{arm.name}'")
    return arm, meshes


def apply_transforms(objs):
    """Bake object transforms in, so world space == armature space."""
    bpy.context.view_layer.objects.active = objs[0]     # mode_set polls on this
    if objs[0].mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in objs:
        if (o.matrix_world - Matrix.Identity(4)).median_scale > 1e-6:
            raise SystemExit(f"{o.name}: transform did not reduce to identity")


def strip_helper_bones(arm):
    """Remove non-deform bones with no skin weight (leg IK targets and the like).

    Left in, they keep their rest transform while the body moves and float away
    from it in the exported armature.
    """
    used = set()
    for o in bpy.data.objects:
        if o.type == "MESH" and o.parent is arm:
            for g in o.vertex_groups:
                used.add(g.name)
    doomed = [b.name for b in arm.data.bones
              if not b.use_deform and b.name not in used]
    if not doomed:
        return []
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for name in doomed:
        eb = arm.data.edit_bones.get(name)
        if eb is not None:
            arm.data.edit_bones.remove(eb)
    bpy.ops.object.mode_set(mode="OBJECT")
    return doomed


def bake_bones(arm):
    bones = list(arm.data.bones)
    names = [b.name for b in bones]
    index = {n: i for i, n in enumerate(names)}
    rest = np.array([[list(r) for r in b.matrix_local] for b in bones], dtype=np.float32)
    head = np.array([list(b.head_local) for b in bones], dtype=np.float32)
    tail = np.array([list(b.tail_local) for b in bones], dtype=np.float32)
    parent = np.array([index[b.parent.name] if b.parent else -1 for b in bones],
                      dtype=np.int32)
    # the +Y column of a bone's rest matrix must be its own direction, and the
    # matrix must carry no scale, or rotation-only posing is not valid
    for i, b in enumerate(bones):
        R = rest[i][:3, :3]
        if abs(np.linalg.det(R) - 1.0) > 1e-4:
            raise SystemExit(f"bone '{b.name}' rest matrix is not a pure rotation")
        d = tail[i] - head[i]
        n = np.linalg.norm(d)
        if n > 1e-9 and np.linalg.norm(R[:, 1] - d / n) > 1e-3:
            raise SystemExit(f"bone '{b.name}' +Y axis is not along the bone")
    return names, index, rest, head, tail, parent


def bake_mesh(meshes, index):
    """Rest-pose triangles and top-4 skin weights, concatenated across meshes."""
    verts, faces, w_idx, w_val = [], [], [], []
    offset = 0
    for o in meshes:
        me = o.data
        me.calc_loop_triangles()
        mw = o.matrix_world
        for v in me.vertices:
            verts.append(list(mw @ v.co))
            # vertex group -> bone; groups that aren't bones are dropped
            ws = []
            for g in v.groups:
                gname = o.vertex_groups[g.group].name
                b = index.get(gname)
                if b is not None and g.weight > 0.0:
                    ws.append((float(g.weight), b))
            ws.sort(reverse=True)
            ws = ws[:MAX_INFLUENCES]
            total = sum(w for w, _ in ws)
            idx = [b for _, b in ws] + [0] * (MAX_INFLUENCES - len(ws))
            val = ([w / total for w, _ in ws] if total > 1e-12
                   else [0.0] * len(ws)) + [0.0] * (MAX_INFLUENCES - len(ws))
            w_idx.append(idx); w_val.append(val)
        for t in me.loop_triangles:
            faces.append([t.vertices[0] + offset, t.vertices[1] + offset,
                          t.vertices[2] + offset])
        offset += len(me.vertices)
    return (np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32),
            np.array(w_idx, dtype=np.int32), np.array(w_val, dtype=np.float32))


def detect_roles(names, override):
    out = {}
    for role, aliases in _ROLE_ALIASES.items():
        if role in override and override[role] in names:
            out[role] = override[role]
            continue
        for cand in aliases:
            if cand in names:
                out[role] = cand
                break
    return out


def report(roles, index, head, tail):
    """Is this rig shaped like a person? Fixed-length retargeting can only put
    the knees and elbows where the subject's are if it is."""
    def seg(role):
        i = index[roles[role]]
        return float(np.linalg.norm(tail[i] - head[i]))

    need = ("thigh.L", "shin.L", "upper_arm.L", "forearm.L")
    if any(r not in roles for r in need):
        print("REPORT: cannot measure — missing roles", [r for r in need if r not in roles])
        return False
    thigh, shank = seg("thigh.L"), seg("shin.L")
    upper, fore = seg("upper_arm.L"), seg("forearm.L")
    stature = float(head[:, 2].max() - head[:, 2].min())

    checks = [
        ("thigh : shank", thigh / shank, 0.90, 1.15),
        ("upper_arm : forearm", upper / fore, 1.15, 1.40),
    ]
    print(f"\nanthropometry (bone-head stature {stature:.3f})")
    ok = True
    for label, val, lo, hi in checks:
        good = lo <= val <= hi
        ok &= good
        print(f"  {label:22s} {val:6.3f}   want {lo:.2f}-{hi:.2f}   {'ok' if good else 'FAIL'}")
    for label, length, key in (("thigh", thigh, "thigh"), ("shank", shank, "shank"),
                               ("upper_arm", upper, "upper_arm"), ("forearm", fore, "forearm")):
        frac = length / stature
        want = _STATURE_FRACTION[key]
        good = abs(frac - want) / want <= 0.12
        ok &= good
        print(f"  {label:22s} {frac:6.3f}   want {want:.3f} ±12%   {'ok' if good else 'FAIL'}")

    # a bent rest limb is what lets the IK pick a bend direction when the
    # captured mid-joint is too noisy to say
    for up, lo_, label in (("thigh.L", "shin.L", "knee"),
                           ("upper_arm.L", "forearm.L", "elbow")):
        a, b = index[roles[up]], index[roles[lo_]]
        v1 = tail[a] - head[a]; v2 = tail[b] - head[b]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        good = ang >= 5.0
        ok &= good
        print(f"  {label + ' rest bend':22s} {ang:6.1f}°  want >= 5°        {'ok' if good else 'FAIL'}")
    return ok


def main():
    args = parse_args()
    blend = Path(bpy.data.filepath)
    override = json.loads(Path(args.roles).read_text()) if args.roles else {}

    arm, meshes = find_rig()
    apply_transforms([arm] + meshes)
    if args.strip_helpers:
        dropped = strip_helper_bones(arm)
        if dropped:
            print(f"dropped {len(dropped)} helper bones: {dropped[:6]}")

    names, index, rest, head, tail, parent = bake_bones(arm)
    verts, faces, w_idx, w_val = bake_mesh(meshes, index)
    roles = detect_roles(names, override)

    missing = [r for r in ("hips", "upper_arm.L", "upper_arm.R", "thigh.L", "shin.L")
               if r not in roles]
    if missing:
        raise SystemExit(f"could not identify required bones {missing}; "
                         f"pass --roles. Bones present: {names[:12]}…")

    passed = report(roles, index, head, tail) if args.report else True

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, verts=verts, faces=faces, w_idx=w_idx, w_val=w_val,
        rest_mat=rest, head=head, tail=tail, parent=parent,
        bone_names=np.array(names), roles=json.dumps(roles),
        blend_sha256=hashlib.sha256(blend.read_bytes()).hexdigest() if blend.exists() else "",
        source=args.source or blend.name)
    print(f"\nwrote {out}: {len(verts)} verts, {len(faces)} faces, {len(names)} bones")
    print(f"roles: {len(roles)}/{len(_ROLE_ALIASES)} identified")
    if args.report and not passed:
        raise SystemExit("rig failed the anthropometric check")


if __name__ == "__main__":
    main()
