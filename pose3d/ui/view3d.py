"""3D skeleton preview using pyqtgraph.opengl.

Renders a semi-transparent grey "shadow figure" (a ghost body built from
capsules along the bones + spheres at the joints) with the coloured skeleton
(joints + bones) overlaid on top — matching the mockup's grey mannequin.

The pose stands ON the grid (grid = ground): the world up-axis is detected from
the skeleton (head vs ankles), mapped to view +Z, and the figure placed by ONE
rule for the whole take — centred on the take's pelvis and seated on the lowest
sole the take reaches (see `ground_datum`, `Character.take_pelvis_ref`). Placing
per frame instead is what used to discard the subject's translation here while
the exported file discarded it by a different rule again; one take-wide rigid
placement is the same rule `pose_bone_matrices(keep_root_motion=True)` gives the
export, so the preview and the delivered file agree frame for frame.
Unit-agnostic (metres or centimetres).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector
from PySide6.QtCore import Signal

from pose3d.core.skeleton import BONES, NUM_JOINTS, Joint
from pose3d.geometry.character import PoseUnavailable
from pose3d.geometry.orient import detect_vertical, upright_matrix


def ground_datum(verts, joints, drop):
    """View-space z of the ground plane under a posed character.

    The SOLE beneath the lower ANKLE, not the lowest mesh vertex. Which vertex
    is lowest changes from frame to frame — a foot, a knee, a fingertip — so
    the old rule slid the ground plane about under the figure and it bobbed
    against the grid by up to 11 % of body height. The ankles are tracked
    joints, so this datum moves only when the subject does. `drop` is the rig's
    rest ankle-to-sole height in these same units (`Character.ground_drop`).
    Falls back to the old rule when neither ankle could be posed.

    The sole is deliberately NOT levelled onto the plane: the shin of this
    rigid-footed mannequin genuinely tilts 16-86 deg, and flattening the foot
    would replace a measurement with a convention.
    """
    if joints is not None:
        z = [joints[int(j)][2]
             for j in (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)]
        z = [q for q in z if np.isfinite(q)]
        if z:
            return float(min(z)) - float(drop)
    return float(verts[:, 2].min())


class View3D(gl.GLViewWidget):
    # Anything that stops the character being posed for a reason OTHER than
    # "this frame has no hips". A missing rig asset, a corrupt .npz, a bad
    # head convention: all of it used to land in a blanket `except Exception`
    # and leave an empty 3D card with no explanation anywhere in the app.
    characterError = Signal(str)

    GHOST_COLOR = (0.72, 0.72, 0.77, 0.85)     # low-poly character skin
    JOINT_COLOR = (0.30, 0.85, 1.0, 1.0)
    BONE_COLOR = (0.95, 0.95, 0.98, 1.0)
    CAPTURE_COLOR = (1.0, 0.72, 0.25, 0.85)    # the measured skeleton
    FILLED_COLOR = (1.0, 0.62, 0.10, 1.0)      # interpolated, not measured

    def __init__(self):
        super().__init__()
        self.setBackgroundColor((14, 16, 22))
        self.setCameraPosition(distance=4.0, elevation=14, azimuth=-70)
        self._grid = gl.GLGridItem()          # XY plane, z = 0 (the ground)
        self._grid.setSize(2, 2)
        self._grid.setSpacing(0.2, 0.2)
        self.addItem(self._grid)

        # --- smooth human body surface (metaball skin around the skeleton) ---
        self._body = gl.GLMeshItem(
            vertexes=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]),
            smooth=True, color=self.GHOST_COLOR, shader="shaded",
            glOptions="translucent", drawEdges=False)
        self._body.setVisible(False)
        self.addItem(self._body)

        # --- the CHARACTER's own skeleton (drawn on top of the mesh) ---
        # Read off the posed rig rather than from the triangulated points, so
        # the overlay always sits inside the body it belongs to.
        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), size=11.0, color=self.JOINT_COLOR, pxMode=True)
        self.addItem(self._scatter)
        self._lines = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), width=2.5, color=self.BONE_COLOR, mode="lines")
        self.addItem(self._lines)

        # --- the CAPTURED skeleton (what the cameras measured), off by default.
        # Kept available so the fit can be judged, in a dimmer colour so it
        # reads as reference rather than as the result.
        self._cap_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), size=8.0, color=self.CAPTURE_COLOR, pxMode=True)
        self._cap_scatter.setVisible(False)
        self.addItem(self._cap_scatter)
        self._cap_lines = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), width=1.5, color=self.CAPTURE_COLOR, mode="lines")
        self._cap_lines.setVisible(False)
        self.addItem(self._cap_lines)

        self._show_body = True
        self._show_capture = False
        self._character = None          # lazily-loaded skinned character
        self._take = None               # the whole take, raw world poses
        # (offset (3,), pelvis travel) in view space; (None, 0.0) once the take
        # has been measured and yielded no placement. None means "not measured
        # yet" — see `_take_placement`, which caches both answers.
        self._place = None
        self._framed = False
        self._vaxis = None
        self._vsign = 1.0
        self._R = None                  # world->view rotation (sequence de-tilt)
        self._char_error = ""           # last message sent to characterError
        self._char_error_source = ""    # which stage put it there

    # --- orientation / framing ---
    def _detect_vertical(self, pose3d, valid):
        return detect_vertical(pose3d, valid)

    def set_orientation(self, R):
        """Set an explicit world->view rotation (3x3), e.g. a whole-sequence
        de-tilt so the figure stands upright. None -> per-frame auto-detect."""
        self._R = None if R is None else np.asarray(R, float).reshape(3, 3)
        self._vaxis = None
        self._place = None              # view space moved: re-measure the take
        self._framed = False

    def fit_subject(self, poses):
        """Size the character to the subject once, from the whole take.

        One uniform scale, so the character changes size but never shape.
        `poses` are raw world poses; they get de-tilted here exactly as
        set_pose() does."""
        try:
            self._ensure_character()
            raw = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
            self._take = raw
            self._place = None
            poses = raw @ self._R.T if self._R is not None else raw
            scale = self._character.fit_to_subject(poses)
        except Exception as e:
            self._report(f"The character could not be prepared for this take "
                         f"({type(e).__name__}: {e}) — the 3D view is showing "
                         f"the captured skeleton only.", "fit")
            return
        if scale is None:
            # No bone in the take was long enough to size the rig against, so
            # `_frame_scale` falls back to a PER-FRAME height ratio — which
            # pulses the figure over a 37.9 % range on the client's take.
            # Silence here is what made that look like the reconstruction
            # breathing rather than the fit never having happened.
            self._report(
                "The character could not be sized to this subject (no bone "
                "was reconstructed well enough to fit against), so its size "
                "is re-guessed every frame and the figure will pulse.", "fit")
        else:
            self._report("")

    def _ensure_character(self):
        if self._character is None:
            from pose3d.geometry.character import Character
            self._character = Character()
        return self._character

    def _take_placement(self):
        """(offset (3,), pelvis travel) placing the WHOLE take, in view space.

        One rigid offset for the sequence, not one per frame:
          * horizontally, the take's pelvis (`take_pelvis_ref`) — the old rule
            re-centred on the mean of the valid joints every frame, which moves
            against the pelvis by up to 5.8 % of body height and quietly
            deleted the subject's travel;
          * vertically, Phase 2's ankle datum — the sole beneath the lower
            ankle, never the lowest mesh vertex — taken at its LOWEST over the
            take, so the grid is the ground the subject actually stood on and
            the figure rises off it when the subject did instead of being
            re-seated frame by frame (that re-seating was a 34 % of rig height
            swing the export had no counterpart for).

        VISIBLE CONSEQUENCE, on the record: seating on the LOWEST sole the take
        reaches means the figure touches the grid on exactly one frame and
        stands above it on the rest — by up to 34.34 % of rig height on the
        client take, which is the spread of the per-frame seat that used to be
        applied. That is the honest reading of the data (the subject really
        was higher on those frames) and it is what makes the preview and the
        export the same rigid map, but it is a change to what the preview
        looks like and worth saying out loud rather than discovering.

        (None, 0.0) when the take is not known yet — a single `set_pose` with
        no `fit_subject` still draws, on the old per-frame rule.

        A frame the character cannot be posed on (`PoseUnavailable`: no PELVIS
        and no hip) contributes no seat and is SKIPPED, exactly as the export
        skips it (`_character_document`'s `pose_bone_matrices` gives None for
        that frame). It must not cost the take its placement: letting one such
        frame abandon the take-wide rule sends the view back to per-frame
        centring — the F13 preview defect Phase 3 fixed — with the export still
        placing by the take, and no message anywhere.

        A genuine fault (a missing rig asset, a corrupt .npz) is a different
        thing and is reported through `characterError` like `_skin` does,
        rather than silently degrading.

        Costs one posing pass per frame of the take, ONCE, cached until the
        take or the orientation changes — the fallback is cached too, so a take
        that yields no placement is not re-posed on every frame change. The
        app's takes are photographed poses (26 on the client's), so that is
        milliseconds.
        """
        if self._place is not None:
            return self._place
        if self._take is None:
            return None, 0.0            # no take yet: nothing to cache
        try:
            from pose3d.geometry.character import take_pelvis_ref
            ch = self._ensure_character()
            up = self._to_view(self._take)
            ref = take_pelvis_ref(up)
            if ref is None:
                self._place = (None, 0.0)
                return self._place
            seats, pelvis = [], []
            for pose in up:
                valid = ~np.isnan(pose).any(1)
                if not valid.any():
                    continue
                vpose = np.where(valid[:, None], pose, np.nan)
                try:
                    verts, _faces, cj = ch.pose_and_joints(vpose, valid)
                except PoseUnavailable:
                    continue            # no root to stand on: no seat either
                if verts is None or not len(verts):
                    continue
                seats.append(ground_datum(verts, cj, ch.ground_drop(vpose, valid)))
                pelvis.append(take_pelvis_ref(pose[None]))
            if not seats:
                self._place = (None, 0.0)
                return self._place
            offset = np.array([ref[0], ref[1], float(min(seats))])
            pel = np.asarray([p for p in pelvis if p is not None], float)
            travel = float(np.linalg.norm(pel.max(0) - pel.min(0))) if len(pel) else 0.0
            self._place = (offset, travel)
        except Exception as e:
            self._report(f"The character could not be placed for this take "
                         f"({type(e).__name__}: {e}) — the 3D view is centring "
                         f"each frame on its own, so it no longer matches the "
                         f"export.", "place")
            self._place = (None, 0.0)
        return self._place

    def _to_view(self, pts):
        """World -> view rotation. Shape-agnostic: used for the canonical
        joints and for the face keypoints alike."""
        if self._R is not None:
            return pts @ self._R.T
        return pts @ upright_matrix(self._vaxis, self._vsign).T

    # --- public API ---
    def set_show_body(self, on: bool):
        self._show_body = on
        self._body.setVisible(on and self._body.opts.get("_active", False))

    def set_show_capture(self, on: bool):
        """Show the skeleton the cameras measured, alongside the character's."""
        self._show_capture = on
        self._cap_scatter.setVisible(on)
        self._cap_lines.setVisible(on)

    def reframe(self):
        self._framed = False
        self._vaxis = None
        self._place = None

    def set_projection(self, mode: str):
        """'Perspective' or 'Orthographic' (approximated via a narrow FOV)."""
        self.opts["fov"] = 1.0 if mode.lower().startswith("ortho") else 60.0
        self.update()

    def set_pose(self, pose3d: np.ndarray, head3d: np.ndarray | None = None,
                 filled: np.ndarray | None = None):
        """`head3d` is the optional (NUM_HEAD_KP, 3) face keypoints; with them
        the character's head is oriented rather than left riding the neck.

        `filled` is the optional (NUM_JOINTS,) flag array from
        `pipeline.fill_gaps`: those joints were interpolated across a
        one-frame dropout, and are drawn in a distinct amber so a fill is
        never read as a measurement."""
        pose3d = np.asarray(pose3d, float).reshape(NUM_JOINTS, 3)
        valid = ~np.isnan(pose3d).any(1)
        if not valid.any():
            self._clear()
            return

        if self._R is None and self._vaxis is None:
            self._vaxis, self._vsign = self._detect_vertical(pose3d, valid)

        v = self._to_view(pose3d)
        vv = v[valid]
        place, travel = self._take_placement()
        if place is not None:
            # ONE placement for the whole take, so what moves on screen is the
            # subject and nothing else — and it is the same rigid map the
            # export applies (see `_take_placement`).
            v = v - place
        else:
            # no take yet: centre horizontally and ground tentatively on the
            # lowest joint (the ankle, since feet aren't detected)
            cx, cy = vv[:, 0].mean(), vv[:, 1].mean()
            v[:, 0] -= cx; v[:, 1] -= cy; v[:, 2] -= vv[:, 2].min()

        # pose the character, then (per-frame path only) ground it on the sole
        # under its lower ankle so the feet rest ON the plane instead of the
        # ankle, which would pierce it.
        vpose = np.where(valid[:, None], v, np.nan)
        # Same world->view rotation as the pose; the grounding translation is
        # deliberately NOT applied, because only a direction basis is read off
        # these and directions are translation-invariant.
        vhead = self._to_view(head3d) if head3d is not None else None
        verts, faces, cj, drop = self._skin(vpose, vhead)
        if place is None and verts is not None and len(verts):
            dz = ground_datum(verts, cj, drop)
            verts = verts.copy(); verts[:, 2] -= dz
            v[:, 2] -= dz
            if cj is not None:
                cj = cj.copy(); cj[:, 2] -= dz          # keep the overlay with the body

        # The character is the subject of the view, so frame on it; fall back to
        # the captured points when it could not be posed.
        ref = verts if (verts is not None and len(verts)) else v[valid]
        height = float(ref[:, 2].max() - ref[:, 2].min()) or 1.0

        # primary overlay: the character's OWN joints, which by construction lie
        # inside the mesh. Falls back to the captured points if there's no rig.
        if cj is not None:
            self._draw_skeleton(self._scatter, self._lines, cj,
                                ~np.isnan(cj).any(1), self.JOINT_COLOR, filled)
        else:
            self._draw_skeleton(self._scatter, self._lines, v, valid,
                                self.JOINT_COLOR, filled)

        # Reference overlay: what the cameras actually MEASURED — so a joint
        # that was interpolated across a dropout is simply absent from it,
        # rather than sitting there in the same amber as everything else.
        measured = valid if filled is None else valid & ~np.asarray(filled, bool)
        self._draw_skeleton(self._cap_scatter, self._cap_lines, v, measured,
                            self.CAPTURE_COLOR)
        self._set_body(verts, faces)

        if not self._framed:
            # the figure PLUS how far it walks: framing on one frame's figure
            # is what let a subject with real translation leave the grid the
            # moment the view stopped re-centring on it every frame
            span = (float(np.linalg.norm(ref.max(0) - ref.min(0))) + travel) or 1.0
            self._grid.setSize(span * 1.6, span * 1.6)
            self._grid.setSpacing(span / 8.0, span / 8.0)
            self.setCameraPosition(pos=Vector(0, 0, height * 0.5),
                                   distance=span * 1.9, elevation=12, azimuth=-70)
            self._framed = True

    @classmethod
    def _draw_skeleton(cls, scatter, lines, pts, valid, base_color=None,
                       filled=None):
        idx = np.flatnonzero(valid)
        if idx.size and base_color is not None:
            colors = np.tile(np.asarray(base_color, float), (idx.size, 1))
            if filled is not None:
                colors[np.asarray(filled, bool)[idx]] = cls.FILLED_COLOR
            scatter.setData(pos=pts[idx], color=colors)
        else:
            scatter.setData(pos=pts[idx] if idx.size else np.zeros((1, 3)))
        seg = []
        for a, b in BONES:
            if valid[int(a)] and valid[int(b)]:
                seg.append(pts[int(a)]); seg.append(pts[int(b)])
        lines.setData(pos=np.array(seg) if seg else np.zeros((2, 3)))

    def _report(self, message: str, source: str = "") -> None:
        """Say a character problem out loud, once per distinct message.

        `source` says which stage put it there. It matters because the two
        stages report at different rates: `fit_subject` speaks once for the
        whole take ("the rig could not be sized, so the figure will pulse"),
        while `_skin` speaks per frame — and a per-frame stage must only
        withdraw its OWN message, or the next good frame would quietly delete
        a take-wide warning that is still true.
        """
        if message == self._char_error:
            return
        self._char_error = message
        self._char_error_source = source if message else ""
        self.characterError.emit(message)

    def _skin(self, vpose, vhead=None):
        """Pose the character -> (verts, faces, canonical joints, ground drop).

        The ground drop is computed HERE rather than at the call site: it is a
        `Character` call like the skinning itself, so a character failure must
        land in the same place instead of escaping `set_pose` into the Qt slot
        (where Qt swallows it and the view silently empties).

        Only `PoseUnavailable` — this frame has no usable pelvis — is normal
        and quiet. Everything else is a real fault and is reported, because
        the blanket catch that used to be here turned a missing character
        asset into an empty 3D card with no message anywhere in the app.
        """
        try:
            self._ensure_character()
            valid = ~np.isnan(vpose).any(1)
            verts, faces, cj = self._character.pose_and_joints(
                vpose, valid, vhead)
            drop = self._character.ground_drop(vpose, valid)
        except PoseUnavailable:
            return None, None, None, 0.0
        except Exception as e:
            self._report(f"The character could not be posed "
                         f"({type(e).__name__}: {e}) — the 3D view is showing "
                         f"the captured skeleton only.", "skin")
            return None, None, None, 0.0
        # The character IS posed, so a message saying it could not be is now
        # false. `_report` de-duplicates on the last message and only
        # `fit_subject` ever sent an empty one, so without this a single bad
        # frame — one LinAlgError on a degenerate pose — pinned "the 3D view
        # is showing the captured skeleton only" in the 3D card header for
        # every good frame after it. Only THIS stage's message is withdrawn:
        # `fit_subject`'s take-wide "the figure will pulse" stays true while
        # every frame skins perfectly, which is exactly the case it warns
        # about.
        if self._char_error_source == "skin":
            self._report("", "skin")
        return verts, faces, cj, drop

    def _set_body(self, verts, faces):
        active = verts is not None and len(verts) > 0
        self._body.opts["_active"] = active
        if active:
            self._body.setMeshData(vertexes=verts, faces=faces)
        self._body.setVisible(self._show_body and active)

    def _clear(self):
        for s, l in ((self._scatter, self._lines),
                     (self._cap_scatter, self._cap_lines)):
            s.setData(pos=np.zeros((1, 3)))
            l.setData(pos=np.zeros((2, 3)))
        self._body.opts["_active"] = False
        self._body.setVisible(False)
