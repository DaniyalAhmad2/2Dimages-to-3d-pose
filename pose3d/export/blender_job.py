"""Runs INSIDE Blender (headless): build skeleton, animate, export.

Invoked as:
    blender --background --python blender_job.py -- --in poses.json --out DIR \
            --fps 30 [--no-video]

Input JSON schema (written by blender_export.py):
    {
      "fps": 30,
      "joint_names": ["HEAD", "NECK", ...],          # canonical order
      "bones": [[parent_idx, child_idx], ...],       # skeleton edges
      "mixamo_names": {"0": "mixamorig:Head", ...},  # joint_idx -> bone name
      "frames": [ [[x,y,z], ...], ... ]              # T x NUM_JOINTS x 3, null for missing
    }

Driving method: FK direct rotation. Each bone's rest axis (+Y) is rotated onto
the current child-direction; we keyframe rotation_quaternion. This is
deterministic and consumes joint world positions directly (no IK solver).

Blender 5.x specifics handled here:
- mp4: set image_settings.media_type='VIDEO' BEFORE file_format='FFMPEG'.
- BVH exporter has no frame_step arg in 5.x.
- FBX/BVH addons are enabled by default; we enable defensively anyway.
"""
import argparse
import json
import sys

import bpy  # noqa: available only inside Blender
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outdir", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--name", default="pose3d")
    p.add_argument("--character", default="")   # .blend already opened as base
    return p.parse_args(argv)


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.armatures, bpy.data.meshes, bpy.data.cameras,
                 bpy.data.lights):
        for block in list(coll):
            coll.remove(block)


def rest_positions(frames):
    """Pick the first frame with all joints present as the rest pose."""
    for fr in frames:
        if all(p is not None for p in fr):
            return [Vector(p) for p in fr]
    # fallback: fill missing with zeros from first frame
    fr = frames[0]
    return [Vector(p) if p is not None else Vector((0, 0, 0)) for p in fr]


def build_armature(rest, bones, joint_names):
    arm_data = bpy.data.armatures.new("PoseSkel")
    arm_obj = bpy.data.objects.new("PoseSkel", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_data.edit_bones
    bone_by_child = {}
    for (a, b) in bones:
        head = rest[a]
        tail = rest[b]
        if (tail - head).length < 1e-5:
            tail = head + Vector((0, 0, 0.05))
        eb = edit_bones.new(joint_names[b])
        eb.head = head
        eb.tail = tail
        eb.use_connect = False
        bone_by_child[b] = joint_names[b]
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj, bone_by_child


def animate(arm_obj, frames, bones, bone_by_child, rest):
    """Keyframe each bone's rotation from the per-frame child direction."""
    scene = bpy.context.scene
    bpy.ops.object.mode_set(mode="POSE")
    # rest direction per bone (in world/edit space)
    rest_dir = {b: (rest[b] - rest[a]).normalized() for (a, b) in bones}

    for fi, fr in enumerate(frames):
        frame_no = fi + 1
        for (a, b) in bones:
            name = bone_by_child[b]
            pb = arm_obj.pose.bones.get(name)
            if pb is None:
                continue
            pa = fr[a]
            pc = fr[b]
            if pa is None or pc is None:
                # keep previous keyframe (occluded); still key to hold value
                pb.keyframe_insert("rotation_quaternion", frame=frame_no)
                continue
            target = (Vector(pc) - Vector(pa))
            if target.length < 1e-6:
                pb.keyframe_insert("rotation_quaternion", frame=frame_no)
                continue
            target = target.normalized()
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = rest_dir[b].rotation_difference(target)
            pb.keyframe_insert("rotation_quaternion", frame=frame_no)
        # root translation from pelvis-like first bone's parent (use frame root)
    bpy.ops.object.mode_set(mode="OBJECT")
    scene.frame_start = 1
    scene.frame_end = len(frames)
    scene.render.fps = int(round(scene.render.fps if False else 0)) or scene.render.fps


def enable_addons():
    import addon_utils
    for a in ("io_scene_fbx", "io_anim_bvh"):
        try:
            addon_utils.enable(a, default_set=True)
        except Exception:
            pass


def export_bvh(arm_obj, path, scene):
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.export_anim.bvh(
        filepath=path, frame_start=scene.frame_start, frame_end=scene.frame_end,
        rotate_mode="NATIVE", root_transform_only=False, global_scale=1.0)


def export_fbx(arm_obj, path):
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"ARMATURE"},
        add_leaf_bones=False, bake_anim=True, bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True, bake_anim_step=1.0,
        primary_bone_axis="Y", secondary_bone_axis="X",
        apply_unit_scale=True, global_scale=1.0,
        axis_forward="-Z", axis_up="Y")


def _detect_up(frames, joint_names):
    """Return (axis, sign) so that sign*coord[axis] is 'up' (head above feet)."""
    idx = {n: i for i, n in enumerate(joint_names)}
    for fr in frames:
        head = fr[idx["HEAD"]] if "HEAD" in idx else None
        lower = None
        for names in (["LEFT_ANKLE", "RIGHT_ANKLE"], ["LEFT_KNEE", "RIGHT_KNEE"],
                      ["PELVIS"], ["LEFT_HIP", "RIGHT_HIP"]):
            pts = [fr[idx[n]] for n in names if n in idx and fr[idx[n]] is not None]
            if pts:
                lower = [sum(c) / len(pts) for c in zip(*pts)]
                break
        if head is not None and lower is not None:
            diff = [head[i] - lower[i] for i in range(3)]
            axis = max(range(3), key=lambda i: abs(diff[i]))
            return axis, (1.0 if diff[axis] >= 0 else -1.0)
    return 2, 1.0


def _remap(p, axis, sign):
    """Remap world coords so the up-axis becomes +Z (upright), as a proper
    ROTATION (det=+1) — flipping one horizontal axis when needed so the figure
    isn't left/right mirrored."""
    others = [i for i in range(3) if i != axis]
    perm_parity = -1.0 if axis == 1 else 1.0
    hx = sign * perm_parity
    return Vector((hx * p[others[0]], p[others[1]], sign * p[axis]))


def _bbox(rframes):
    import math
    lo = [math.inf] * 3; hi = [-math.inf] * 3
    for fr in rframes:
        for p in fr:
            if p is None:
                continue
            for i in range(3):
                lo[i] = min(lo[i], p[i]); hi[i] = max(hi[i], p[i])
    if lo[0] == math.inf:
        return Vector((-1, -1, -1)), Vector((1, 1, 1))
    return Vector(lo), Vector(hi)


def _materials():
    def _mat(name, rgba):
        m = bpy.data.materials.new(name)
        m.use_nodes = True                       # EEVEE renders the node base color
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = rgba
        m.diffuse_color = rgba
        return m
    return _mat("joint", (0.20, 0.65, 1.0, 1)), _mat("bone", (0.92, 0.92, 0.96, 1))


def add_mesh_figure(rframes, bones, jr, br, jmat, bmat, animate):
    """Build the EXACT figure shown in the app as mesh: a sphere per joint and a
    cylinder per bone, positioned from the 3D points. If animate and there are
    multiple frames, keyframe every frame; else place statically at rframes[0].
    Missing joints collapse to zero size. Returns the created objects."""
    n = len(rframes[0])
    joints, bone_objs = [], []
    for j in range(n):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=jr, segments=16, ring_count=10)
        o = bpy.context.active_object; o.name = f"Joint_{j}"
        o.data.materials.append(jmat); joints.append(o)
    for k in range(len(bones)):
        bpy.ops.mesh.primitive_cylinder_add(radius=br, depth=1.0, vertices=12)
        o = bpy.context.active_object; o.name = f"Bone_{k}"
        o.rotation_mode = "QUATERNION"; o.data.materials.append(bmat)
        bone_objs.append(o)

    zaxis = Vector((0, 0, 1))

    def place(fr):
        for j, o in enumerate(joints):
            p = fr[j]
            if p is None:
                o.scale = (0, 0, 0)
            else:
                o.location = Vector(p); o.scale = (1, 1, 1)
        for k, (a, b) in enumerate(bones):
            o = bone_objs[k]; pa, pb = fr[a], fr[b]
            d = None if (pa is None or pb is None) else (Vector(pb) - Vector(pa))
            if d is None or d.length < 1e-9:
                o.scale = (0, 0, 0)
            else:
                o.location = (Vector(pa) + Vector(pb)) / 2.0
                o.rotation_quaternion = zaxis.rotation_difference(d.normalized())
                o.scale = (1, 1, d.length)

    if animate and len(rframes) > 1:
        for fi, fr in enumerate(rframes):
            f = fi + 1; place(fr)
            for o in joints:
                o.keyframe_insert("location", frame=f)
                o.keyframe_insert("scale", frame=f)
            for o in bone_objs:
                o.keyframe_insert("location", frame=f)
                o.keyframe_insert("rotation_quaternion", frame=f)
                o.keyframe_insert("scale", frame=f)
    else:
        place(rframes[0])
    return joints + bone_objs


def export_fbx_mesh(objs, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"MESH"},
        bake_anim=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True,
        bake_anim_step=1.0, mesh_smooth_type="FACE",
        apply_unit_scale=True, global_scale=1.0,
        axis_forward="-Z", axis_up="Y")


def _add_camera_light(center, diag, scene):
    tgt = bpy.data.objects.new("Tgt", None)
    scene.collection.objects.link(tgt); tgt.location = center
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.clip_end = max(1000.0, diag * 20)
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = center + Vector((0, -diag * 1.9, diag * 0.12))
    con = cam.constraints.new("TRACK_TO")
    con.target = tgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
    scene.camera = cam
    for pos in ((diag, -diag, diag * 2), (-diag, -diag, diag), (0, diag, diag)):
        ld = bpy.data.lights.new("L", type="SUN"); ld.energy = 2.5
        lo = bpy.data.objects.new("L", ld)
        scene.collection.objects.link(lo); lo.location = center + Vector(pos)


def turntable_spin(objs, center, scene, seconds, fps):
    """Parent the figure to an empty and spin it 360° so the loop is seamless."""
    import math
    empty = bpy.data.objects.new("Spin", None)
    scene.collection.objects.link(empty); empty.location = center
    inv = Matrix.Translation(-center)
    for o in objs:
        o.parent = empty
        o.matrix_parent_inverse = inv
    total = max(2, int(round(seconds * fps)))
    # keyframe every frame (dense = constant-speed spin) so we don't depend on
    # the fcurve interpolation API (which changed in Blender 5.x Actions).
    for f in range(1, total + 1):
        empty.rotation_euler = (0, 0, 2 * math.pi * (f - 1) / total)  # f=total < 360°
        empty.keyframe_insert("rotation_euler", index=2, frame=f)
    scene.frame_start = 1; scene.frame_end = total


def render_mp4(path, scene, fps):
    scene.render.filepath = path
    r = scene.render.image_settings
    r.media_type = "VIDEO"          # 5.x: MUST come before file_format
    r.file_format = "FFMPEG"
    ff = scene.render.ffmpeg
    ff.format = "MPEG4"; ff.codec = "H264"
    ff.constant_rate_factor = "MEDIUM"; ff.audio_codec = "NONE"
    scene.render.fps = fps; scene.render.fps_base = 1.0
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"):
        if eng in engines:
            scene.render.engine = eng
            break
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    bpy.ops.render.render(animation=True)


def _delete(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        try:
            o.select_set(True)
        except Exception:
            pass
    bpy.ops.object.delete()


# --- character retargeting (drive a rigged .blend humanoid from the pose) ----

# our canonical joint -> rig deform bone that should point at it (Damped Track)
_RIG_TRACK = [
    ("spine", "NECK"), ("chest", "NECK"), ("neck", "HEAD"),
    ("upper_arm.L", "LEFT_ELBOW"), ("forearm.L", "LEFT_WRIST"),
    ("upper_arm.R", "RIGHT_ELBOW"), ("forearm.R", "RIGHT_WRIST"),
    ("thigh.L", "LEFT_KNEE"), ("shin.L", "LEFT_ANKLE"),
    ("thigh.R", "RIGHT_KNEE"), ("shin.R", "RIGHT_ANKLE"),
]


def _find_rig():
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        return None, []
    arm = max(arms, key=lambda o: len(o.data.bones))
    meshes = [o for o in bpy.data.objects
              if o.type == "MESH" and o.parent is arm]
    return arm, meshes


def _retarget_character(arm, rframes, joint_names, scene):
    import math
    idx = {n: i for i, n in enumerate(joint_names)}

    zs = [(arm.matrix_world @ pt).z for b in arm.data.bones
          for pt in (b.head_local, b.tail_local)]
    rig_h = (max(zs) - min(zs)) or 1.0
    allz = [p[2] for fr in rframes for p in fr if p is not None]
    our_h = (max(allz) - min(allz)) or 1.0
    scale = rig_h / our_h
    hips_world = arm.matrix_world @ arm.data.bones["hips"].head_local

    # rotate about Z so our left/right + facing match the rig's
    Rz = Matrix.Identity(3)
    Lh = arm.data.bones["upper_arm.L"].head_local
    Rh = arm.data.bones["upper_arm.R"].head_local
    rig_right = Vector((Rh.x - Lh.x, Rh.y - Lh.y, 0.0)).normalized()
    for fr in rframes:
        ls, rs = fr[idx["LEFT_SHOULDER"]], fr[idx["RIGHT_SHOULDER"]]
        if ls and rs:
            our_right = Vector((rs[0] - ls[0], rs[1] - ls[1], 0.0)).normalized()
            dt = (math.atan2(rig_right.y, rig_right.x)
                  - math.atan2(our_right.y, our_right.x))
            Rz = Matrix.Rotation(dt, 3, "Z")
            break

    empties = {}
    for n in joint_names:
        e = bpy.data.objects.new(f"T_{n}", None)
        scene.collection.objects.link(e); empties[n] = e

    def pelvis_of(fr):
        p = fr[idx["PELVIS"]]
        if p is not None:
            return Vector(p)
        hips = [fr[idx[n]] for n in ("LEFT_HIP", "RIGHT_HIP") if fr[idx[n]]]
        return Vector([sum(c) / len(hips) for c in zip(*hips)]) if hips else Vector()

    last = {}
    for fi, fr in enumerate(rframes):
        f = fi + 1
        pv = pelvis_of(fr)
        for n in joint_names:
            p = fr[idx[n]]
            if p is not None:
                w = hips_world + (Rz @ (Vector(p) - pv)) * scale
                last[n] = w
            else:
                w = last.get(n, hips_world)     # hold last known if occluded
            empties[n].location = w
            empties[n].keyframe_insert("location", frame=f)

    pb = arm.pose.bones
    if "hips" in pb:
        pb["hips"].constraints.new("COPY_LOCATION").target = empties["PELVIS"]
    for bone, tgt in _RIG_TRACK:
        if bone in pb and tgt in empties:
            # aim only (no stretch) so the exported character keeps its natural
            # proportions — the render shows the character, not the skeleton, so
            # limb-length mismatch isn't visible and distortion is avoided.
            c = pb[bone].constraints.new("DAMPED_TRACK")
            c.target = empties[tgt]; c.track_axis = "TRACK_Y"


def _bake_and_clean(arm, scene):
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.nla.bake(
        frame_start=scene.frame_start, frame_end=scene.frame_end,
        only_selected=False, visual_keying=True, clear_constraints=True,
        clear_parents=False, use_current_action=True, bake_types={"POSE"})
    bpy.ops.object.mode_set(mode="OBJECT")
    _delete([o for o in bpy.data.objects if o.name.startswith("T_")])


def _character_bounds(meshes):
    dg = bpy.context.evaluated_depsgraph_get()
    lo = Vector((1e18, 1e18, 1e18)); hi = -lo
    for m in meshes:
        me = m.evaluated_get(dg)
        for corner in me.bound_box:
            w = m.matrix_world @ Vector(corner)
            lo = Vector((min(lo.x, w.x), min(lo.y, w.y), min(lo.z, w.z)))
            hi = Vector((max(hi.x, w.x), max(hi.y, w.y), max(hi.z, w.z)))
    return (lo + hi) / 2.0, (hi - lo).length or 1.0


def _export_character_fbx(objs, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True,
        object_types={"ARMATURE", "MESH"}, add_leaf_bones=False,
        bake_anim=True, bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True, bake_anim_step=1.0,
        primary_bone_axis="Y", secondary_bone_axis="X",
        apply_unit_scale=True, global_scale=1.0, axis_forward="-Z", axis_up="Y")


def character_main(data, args, scene):
    frames = [[(None if p is None else tuple(p)) for p in fr]
              for fr in data["frames"]]
    joint_names = data["joint_names"]
    axis, sign = _detect_up(frames, joint_names)
    rframes = [[None if p is None else _remap(p, axis, sign) for p in fr]
               for fr in frames]

    arm, meshes = _find_rig()
    if arm is None or "hips" not in arm.pose.bones:
        return False           # not a usable rig -> caller falls back

    # drop the ground plane + any pre-existing camera from the template
    _delete([o for o in bpy.data.objects
             if (o.type == "MESH" and o.parent is None) or o.type == "CAMERA"])

    scene.render.fps = args.fps
    _retarget_character(arm, rframes, joint_names, scene)
    scene.frame_start = 1; scene.frame_end = len(rframes)
    _bake_and_clean(arm, scene)

    enable_addons()
    import os
    os.makedirs(args.outdir, exist_ok=True)
    export_bvh(arm, os.path.join(args.outdir, args.name + ".bvh"), scene)
    _export_character_fbx([arm] + meshes, os.path.join(args.outdir, args.name + ".fbx"))

    if not args.no_video:
        center, diag = _character_bounds(meshes)
        _add_camera_light(center, diag, scene)
        turntable_spin([arm], center, scene, seconds=6, fps=args.fps)
        render_mp4(os.path.join(args.outdir, args.name + ".mp4"), scene, args.fps)
    return True


def main():
    args = parse_args()
    with open(args.infile) as f:
        data = json.load(f)

    scene = bpy.context.scene
    # character mode: a rigged .blend was opened as the base file
    if args.character:
        try:
            if character_main(data, args, scene):
                print("POSE3D_EXPORT_OK")
                return
        except Exception as e:
            print(f"character retarget failed ({e}); falling back to skeleton")

    frames = [[(None if p is None else tuple(p)) for p in fr]
              for fr in data["frames"]]
    bones = [tuple(b) for b in data["bones"]]
    joint_names = data["joint_names"]
    display = int(data.get("display_frame", 0))

    clean_scene()
    scene = bpy.context.scene
    scene.render.fps = args.fps
    enable_addons()
    import os
    os.makedirs(args.outdir, exist_ok=True)

    # upright remap so exports/renders are the right way up at any unit scale
    axis, sign = _detect_up(frames, joint_names)
    rframes = [[None if p is None else _remap(p, axis, sign) for p in fr]
               for fr in frames]
    lo, hi = _bbox(rframes)
    center = (lo + hi) / 2.0
    diag = (hi - lo).length or 1.0
    jr, br = diag * 0.024, diag * 0.014
    jmat, bmat = _materials()

    # 1) BVH from the FK armature (skeletal mocap format needs a bone hierarchy)
    rest = rest_positions(frames)
    arm_obj, bbc = build_armature(rest, bones, joint_names)
    animate(arm_obj, frames, bones, bbc, rest)
    scene.frame_start = 1; scene.frame_end = len(frames)
    export_bvh(arm_obj, os.path.join(args.outdir, args.name + ".bvh"), scene)
    _delete([arm_obj])                     # keep it out of the FBX/render

    # 2) FBX: the EXACT mesh figure shown in the app (animated if multi-frame)
    figure = add_mesh_figure(rframes, bones, jr, br, jmat, bmat, animate=True)
    scene.frame_start = 1; scene.frame_end = len(rframes)
    export_fbx_mesh(figure, os.path.join(args.outdir, args.name + ".fbx"))
    _delete(figure)

    # 3) MP4: a spinning turntable loop of the displayed pose
    if not args.no_video:
        disp = rframes[display] if 0 <= display < len(rframes) else rframes[0]
        static = add_mesh_figure([disp], bones, jr, br, jmat, bmat, animate=False)
        _add_camera_light(center, diag, scene)
        turntable_spin(static, center, scene, seconds=6, fps=args.fps)
        render_mp4(os.path.join(args.outdir, args.name + ".mp4"), scene, args.fps)

    print("POSE3D_EXPORT_OK")


if __name__ == "__main__":
    main()
