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


def setup_camera_light(rest):
    center = sum(rest, Vector((0, 0, 0))) / len(rest)
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = center + Vector((0, -4.0, 0.3))
    cam.rotation_euler = (1.4, 0, 0)
    bpy.context.scene.camera = cam
    light_data = bpy.data.lights.new("Sun", type="SUN")
    light = bpy.data.objects.new("Sun", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = center + Vector((2, -2, 4))


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
        setup_camera_light(rest)
        render_mp4(os.path.join(args.outdir, args.name + ".mp4"), scene, args.fps)
    print("POSE3D_EXPORT_OK")


if __name__ == "__main__":
    main()
