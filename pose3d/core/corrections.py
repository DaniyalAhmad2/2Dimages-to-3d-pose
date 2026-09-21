"""Reversible 2D correction stack (undo/redo) + correction log.

Every manual edit is a discrete, reversible operation. Applying an edit records
the old value so it can be undone; edits are also appended to the SQLite log
(seeds the v2 learning loop). This module is Qt-free and unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pose3d.core.project import Correction, Frame
from pose3d.core.skeleton import face_kp_index, is_face_kp


@dataclass
class Edit:
    frame_id: str
    cam: str
    joint: int
    old_xy: tuple[float, float]
    new_xy: tuple[float, float]
    old_score: float
    old_corrected: bool


@dataclass
class CorrectionStack:
    """Undo/redo of joint edits against a set of frames."""
    frames_by_id: dict[str, Frame]
    _undo: list[Edit] = field(default_factory=list)
    _redo: list[Edit] = field(default_factory=list)
    log: list[Correction] = field(default_factory=list)

    def apply(self, frame_id: str, cam: str, joint: int,
              x: float, y: float, ts: str = "") -> Edit:
        """A face id (`skeleton.face_kp_id(k)`, at the fixed FACE_KP_BASE)
        addresses face keypoint k (nose/eyes/ears); anything below it is a
        body joint. The one convention shared with the camera views and the
        SQLite log, whose integer column simply extends."""
        f = self.frames_by_id[frame_id]
        if is_face_kp(joint):
            k = face_kp_index(joint)
            old = tuple(f.head2d[cam][k])
            edit = Edit(frame_id, cam, joint, (float(old[0]), float(old[1])),
                        (float(x), float(y)), float(f.head_scores[cam][k]),
                        bool(f.head_corrected[cam][k]))
            f.set_head_kp(cam, k, x, y, score=1.0, corrected=True)
        else:
            old = tuple(f.kp2d[cam][joint])
            edit = Edit(frame_id, cam, joint, (float(old[0]), float(old[1])),
                        (float(x), float(y)), float(f.scores[cam][joint]),
                        bool(f.corrected[cam][joint]))
            f.set_kp(cam, joint, x, y, score=1.0, corrected=True)
        self._undo.append(edit)
        self._redo.clear()
        self.log.append(Correction(frame_id, cam, joint,
                                    edit.old_xy, edit.new_xy, ts))
        return edit

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> Edit | None:
        if not self._undo:
            return None
        e = self._undo.pop()
        f = self.frames_by_id[e.frame_id]
        if is_face_kp(e.joint):
            k = face_kp_index(e.joint)
            f.head2d[e.cam][k] = e.old_xy
            f.head_scores[e.cam][k] = e.old_score
            f.head_corrected[e.cam][k] = e.old_corrected
        else:
            f.kp2d[e.cam][e.joint] = e.old_xy
            f.scores[e.cam][e.joint] = e.old_score
            f.corrected[e.cam][e.joint] = e.old_corrected
        self._redo.append(e)
        return e

    def redo(self) -> Edit | None:
        if not self._redo:
            return None
        e = self._redo.pop()
        f = self.frames_by_id[e.frame_id]
        if is_face_kp(e.joint):
            f.set_head_kp(e.cam, face_kp_index(e.joint),
                          e.new_xy[0], e.new_xy[1], score=1.0, corrected=True)
        else:
            f.set_kp(e.cam, e.joint, e.new_xy[0], e.new_xy[1],
                     score=1.0, corrected=True)
        self._undo.append(e)
        return e
