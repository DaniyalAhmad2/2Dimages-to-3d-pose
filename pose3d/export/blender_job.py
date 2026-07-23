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
from mathutils import Vector


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outdir", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--name", default="pose3d")
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
    """Remap world coords so the up-axis becomes +Z (upright render)."""
    others = [i for i in range(3) if i != axis]
    return Vector((p[others[0]], p[others[1]], sign * p[axis]))


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


def build_render_scene(frames, bones, joint_names, scene):
    """Armatures don't render, so build visible mesh geometry (spheres at
    joints + cylinders for bones), keyframed per frame, with an auto-framed
    upright camera + light. Works at any unit scale (metres or centimetres)."""
    axis, sign = _detect_up(frames, joint_names)
    rframes = [[None if p is None else _remap(p, axis, sign) for p in fr]
               for fr in frames]
    lo, hi = _bbox(rframes)
    center = (lo + hi) / 2.0
    diag = (hi - lo).length or 1.0
    jr, br = diag * 0.022, diag * 0.013

    def _mat(name, rgba):
        m = bpy.data.materials.new(name)
        m.use_nodes = True                       # EEVEE renders node base color
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = rgba
        m.diffuse_color = rgba                    # viewport colour too
        return m

    jmat = _mat("joint", (0.20, 0.65, 1.0, 1))
    bmat = _mat("bone", (0.92, 0.92, 0.96, 1))

    n = len(joint_names)
    joints = []
    for j in range(n):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=jr, segments=12, ring_count=8)
        o = bpy.context.active_object; o.name = f"J{j}"
        o.data.materials.append(jmat); joints.append(o)
    bone_objs = []
    for k in range(len(bones)):
        bpy.ops.mesh.primitive_cylinder_add(radius=br, depth=1.0, vertices=10)
        o = bpy.context.active_object; o.name = f"B{k}"
        o.rotation_mode = "QUATERNION"
        o.data.materials.append(bmat); bone_objs.append(o)

    zaxis = Vector((0, 0, 1))
    for fi, fr in enumerate(rframes):
        f = fi + 1
        for j, o in enumerate(joints):
            p = fr[j]
            o.hide_render = p is None
            if p is not None:
                o.location = p
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("hide_render", frame=f)
        for k, (a, b) in enumerate(bones):
            o = bone_objs[k]
            pa, pb = fr[a], fr[b]
            hidden = pa is None or pb is None
            if not hidden:
                d = pb - pa; L = d.length
                hidden = L < 1e-9
                if not hidden:
                    o.location = (pa + pb) / 2.0
                    o.rotation_quaternion = zaxis.rotation_difference(d.normalized())
                    o.scale = (1.0, 1.0, L)
            o.hide_render = hidden
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("rotation_quaternion", frame=f)
            o.keyframe_insert("scale", frame=f)
            o.keyframe_insert("hide_render", frame=f)

    # camera framing the figure from the front, slightly above and to the side
    target = bpy.data.objects.new("Target", None)
    scene.collection.objects.link(target); target.location = center
    cam_data = bpy.data.cameras.new("Cam"); cam_data.clip_end = max(1000.0, diag * 20)
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = center + Vector((diag * 0.35, -diag * 1.7, diag * 0.15))
    con = cam.constraints.new("TRACK_TO")
    con.target = target; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
    scene.camera = cam

    light_data = bpy.data.lights.new("Sun", type="SUN"); light_data.energy = 3.0
    light = bpy.data.objects.new("Sun", light_data)
    scene.collection.objects.link(light)
    light.location = center + Vector((diag, -diag, diag * 2))


def render_mp4(path, scene, fps):
    scene.render.filepath = path
    r = scene.render.image_settings
    r.media_type = "VIDEO"          # 5.x: MUST come before file_format
    r.file_format = "FFMPEG"
    ff = scene.render.ffmpeg
    ff.format = "MPEG4"
    ff.codec = "H264"
    ff.constant_rate_factor = "MEDIUM"
    ff.audio_codec = "NONE"
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    # Blender 5.1.1 exposes EEVEE as 'BLENDER_EEVEE' (EEVEE-Next took the name
    # in 4.2+). Fall back defensively if the enum differs across builds.
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"):
        if eng in engines:
            scene.render.engine = eng
            break
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    bpy.ops.render.render(animation=True)


def main():
    args = parse_args()
    with open(args.infile) as f:
        data = json.load(f)

    frames = [[(None if p is None else tuple(p)) for p in fr]
              for fr in data["frames"]]
    bones = [tuple(b) for b in data["bones"]]
    joint_names = data["joint_names"]

    clean_scene()
    scene = bpy.context.scene
    scene.render.fps = args.fps

    rest = rest_positions(frames)
    arm_obj, bone_by_child = build_armature(rest, bones, joint_names)
    animate(arm_obj, frames, bones, bone_by_child, rest)
    scene.frame_start = 1
    scene.frame_end = len(frames)

    enable_addons()
    import os
    os.makedirs(args.outdir, exist_ok=True)
    export_bvh(arm_obj, os.path.join(args.outdir, args.name + ".bvh"), scene)
    export_fbx(arm_obj, os.path.join(args.outdir, args.name + ".fbx"))
    if not args.no_video:
        build_render_scene(frames, bones, joint_names, scene)
        render_mp4(os.path.join(args.outdir, args.name + ".mp4"), scene, args.fps)
    print("POSE3D_EXPORT_OK")


if __name__ == "__main__":
    main()
