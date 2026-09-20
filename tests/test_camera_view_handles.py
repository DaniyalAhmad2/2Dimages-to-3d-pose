"""The dots the client has to grab, and the ones he has to place.

Two client complaints live here (2026-08-16, R11):

1. The joint handles were 6 units in IMAGE coordinates. His photographs are
   3072x4080 and the panel fits the whole frame into ~400 px, so the dot he
   had to hit drew at ~1.4 px across and the click had to land on the exact
   pixel of the joint — a miss pans the photo instead. The handles are now
   sized in SCREEN pixels and are the same size to grab at any zoom.

2. A joint the detector did not find was hidden, and an invisible
   QGraphicsItem is not hit-tested — so the frames the tool exists for
   (occlusion, extreme poses) were exactly the ones that could not be fixed
   by hand. An undetected joint is now drawn as a dashed placeholder that can
   be dragged onto the limb, which records the correction like any other drag.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt                       # noqa: E402
from PySide6.QtGui import QColor, QPixmap                            # noqa: E402

from pose3d.core.skeleton import (                                   # noqa: E402
    BONES, NUM_HEAD_KP, NUM_JOINTS)
from pose3d.ui.camera_view import (                                  # noqa: E402
    FACE_HANDLE_R, HANDLE_R, MIN_GRAB_PX, RAG_COLORS, CameraView)

#: The client's left camera: tests/fixtures/client_take .. left_intrinsics.json
CLIENT_IMAGE = (3072, 4080)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _xy():
    # spread down the diagonal of the photograph: at the fitted scale the
    # handles are ~20 screen px across, so joints 60 image px apart would
    # overlap and `itemAt` would answer the neighbour
    return np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 170 + 200


def _scores():
    return np.full(NUM_JOINTS, 0.9)


def _client_view(tmp_path, w=500, h=700):
    """A view of a client-sized photograph, fitted into a small panel."""
    pm = QPixmap(*CLIENT_IMAGE)
    pm.fill(QColor(30, 30, 30))
    path = tmp_path / "left.png"
    assert pm.save(str(path))
    v = CameraView("left")
    v.resize(w, h)
    v.set_image(str(path))
    return v


def _on_screen(view, item):
    """The item's bounding box in viewport pixels.

    `deviceTransform`, not `sceneTransform`: an ItemIgnoresTransformations
    item has no meaningful scene transform — that is the whole point of it.
    """
    return item.deviceTransform(view.viewportTransform()).mapRect(
        item.boundingRect())


# --- 1. the handle is a screen-pixel size, not an image-pixel one ----------

def test_a_body_handle_is_a_grabbable_size_on_the_clients_photograph(
        qapp, tmp_path):
    v = _client_view(tmp_path)
    v.set_pose(_xy(), _scores())
    assert v.transform().m11() < 0.25, \
        "this test is pointless unless the photo is fitted down hard"

    r = _on_screen(v, v._joints[0])
    assert r.width() >= MIN_GRAB_PX and r.height() >= MIN_GRAB_PX, \
        f"joint handle is {r.width():.1f}x{r.height():.1f} px on screen"


def test_the_handle_is_the_same_size_at_any_zoom(qapp, tmp_path):
    """Zoom used to be the only way to hit a joint, and every frame step threw
    it away (`set_image` re-fits). Size may not depend on the transform."""
    v = _client_view(tmp_path)
    v.set_pose(_xy(), _scores())
    fitted = _on_screen(v, v._joints[0])

    v.zoom(8.0)
    zoomed = _on_screen(v, v._joints[0])
    v.fit()
    refitted = _on_screen(v, v._joints[0])

    assert zoomed.width() == pytest.approx(fitted.width())
    assert refitted.width() == pytest.approx(fitted.width())


def test_the_face_points_are_smaller_than_the_body_joints(qapp, tmp_path):
    v = _client_view(tmp_path)
    head = np.tile(np.arange(NUM_HEAD_KP, dtype=float)[:, None], (1, 2)) * 20 + 900
    v.set_pose(_xy(), _scores(), head_xy=head, head_source="skull",
               head_mode="face")

    body = _on_screen(v, v._joints[0]).width()
    face = _on_screen(v, v._face[0]).width()
    assert face < body, "the face points must stay secondary to the skeleton"
    assert FACE_HANDLE_R < HANDLE_R
    assert face >= 2 * FACE_HANDLE_R, "…but still be grabbable"


def test_a_near_miss_grabs_the_joint_instead_of_panning(qapp, tmp_path):
    """The hit area is the handle: a press that is not on a JointItem pans the
    photo (CameraView.mousePressEvent), so 'nearly on it' must still grab."""
    v = _client_view(tmp_path)
    v.set_pose(_xy(), _scores())
    item = v._joints[5]
    centre = v.mapFromScene(item.pos())

    assert v.itemAt(centre) is item
    off = int(MIN_GRAB_PX / 2) - 2          # inside the ring, nowhere near the
    assert v.itemAt(centre + QPoint(off, 0)) is item      # exact pixel
    assert v.itemAt(centre + QPoint(0, -off)) is item


def test_a_real_drag_still_reports_image_coordinates(qapp, tmp_path):
    """The handle is sized in screen pixels; the POSITION it reports is still
    the joint's position in the photograph, which is what the model stores."""
    from PySide6.QtGui import QMouseEvent

    v = _client_view(tmp_path)
    v.show()
    v.set_pose(_xy(), _scores())
    seen = []
    v.jointDragged.connect(lambda cam, j, p: seen.append((cam, j, p)))

    item = v._joints[5]
    vp = v.viewport()
    start = v.mapFromScene(item.pos())
    end = start + QPoint(40, 25)

    def send(kind, pos, buttons):
        qapp.sendEvent(vp, QMouseEvent(
            kind, QPointF(pos), QPointF(vp.mapToGlobal(pos)),
            Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier))

    send(QMouseEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    send(QMouseEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton)
    send(QMouseEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton)

    want = v.mapToScene(end)
    assert len(seen) == 1, "the drag must commit exactly once, on mouse-up"
    cam, jid, pos = seen[0]
    assert (cam, jid) == ("left", 5)
    # within a pixel of the image point under the cursor: the item keeps the
    # sub-pixel offset the press had from its centre, as any drag does
    assert abs(pos.x() - want.x()) < 2 and abs(pos.y() - want.y()) < 2
    assert pos == item.pos()


def test_the_bones_stay_in_scene_space(qapp, tmp_path):
    """Only the handles are screen-sized. A bone is a limb in the photograph:
    it must follow the image when the view zooms."""
    v = _client_view(tmp_path)
    v.set_pose(_xy(), _scores())
    a, b = BONES[0]
    line = v._bones[0]
    assert not (line.flags()
                & line.GraphicsItemFlag.ItemIgnoresTransformations)
    assert line.line().x1() == pytest.approx(v._joints[int(a)].pos().x())
    assert line.line().y2() == pytest.approx(v._joints[int(b)].pos().y())


# --- 2. an undetected joint is placed, not hidden -------------------------

def test_an_undetected_joint_gets_a_placeholder_where_it_was_last_seen(
        qapp, tmp_path):
    v = _client_view(tmp_path)
    xy = _xy()
    v.set_pose(xy, _scores())                      # seen here...
    gone = 3
    missing = xy.copy(); missing[gone] = np.nan
    v.set_pose(missing, _scores())                 # ...and not here

    item = v._joints[gone]
    assert item.is_placeholder
    assert item.isVisible(), "an invisible item cannot be dragged"
    assert item.pos().x() == pytest.approx(xy[gone][0])
    assert item.pos().y() == pytest.approx(xy[gone][1])
    assert "not detected" in item.toolTip()
    assert "drag to place" in item.toolTip()


def test_a_placeholder_can_never_be_read_as_a_measurement(qapp, tmp_path):
    v = _client_view(tmp_path)
    missing = _xy(); missing[3] = np.nan
    v.set_pose(missing, _scores())
    item = v._joints[3]

    assert item.brush().style() == Qt.BrushStyle.NoBrush, "must be hollow"
    assert item.pen().style() == Qt.PenStyle.DashLine, "must be dashed"
    assert item.pen().color() == RAG_COLORS["missing"]
    # and it is still the full grab size: placing it is the whole point
    assert _on_screen(v, item).width() >= MIN_GRAB_PX


def test_a_joint_this_view_has_never_seen_starts_at_the_image_centre(
        qapp, tmp_path):
    v = _client_view(tmp_path)
    missing = _xy(); missing[0] = np.nan
    v.set_pose(missing, _scores())

    item = v._joints[0]
    assert item.is_placeholder
    assert item.pos().x() == pytest.approx(CLIENT_IMAGE[0] / 2)
    assert item.pos().y() == pytest.approx(CLIENT_IMAGE[1] / 2)


def test_no_bone_is_drawn_to_a_placeholder(qapp, tmp_path):
    """A bone to a joint nobody saw would draw a limb out of a guess."""
    v = _client_view(tmp_path)
    v.set_pose(_xy(), _scores())
    a, b = (int(x) for x in BONES[0])
    missing = _xy(); missing[a] = np.nan
    v.set_pose(missing, _scores())

    assert not v._bones[0].isVisible()
    others = [i for i, (p, q) in enumerate(BONES)
              if a not in (int(p), int(q)) and b not in (int(p), int(q))]
    assert v._bones[others[0]].isVisible(), "the rest of the skeleton stays"


def test_dragging_a_placeholder_reports_the_new_position_like_any_drag(
        qapp, tmp_path):
    """Placing it goes out through the SAME `released` -> jointDragged path,
    so the model records a correction and marks the joint corrected."""
    v = _client_view(tmp_path)
    missing = _xy(); missing[4] = np.nan
    v.set_pose(missing, _scores())
    seen = []
    v.jointDragged.connect(lambda cam, j, p: seen.append((cam, j, p)))

    v._joints[4].signals.released.emit(4, QPointF(1200.0, 2400.0))

    assert seen == [("left", 4, QPointF(1200.0, 2400.0))]


def test_a_placed_joint_renders_as_a_normal_corrected_handle(qapp, tmp_path):
    v = _client_view(tmp_path)
    missing = _xy(); missing[4] = np.nan
    v.set_pose(missing, _scores())
    assert v._joints[4].is_placeholder

    placed = _xy(); placed[4] = (1200.0, 2400.0)
    corrected = np.zeros(NUM_JOINTS, bool); corrected[4] = True
    v.set_pose(placed, _scores(), corrected=corrected)

    item = v._joints[4]
    assert not item.is_placeholder
    assert item.pen().style() == Qt.PenStyle.SolidLine
    assert item.brush().color() == RAG_COLORS["corrected"]
    assert "corrected by hand" in item.toolTip()


def test_the_joints_toggle_still_only_hides_and_unhides(qapp, tmp_path):
    """`set_show_joints` may not overrule what `set_pose` decided — including
    a placeholder, which is now something `set_pose` deliberately shows."""
    v = _client_view(tmp_path)
    missing = _xy(); missing[3] = np.nan
    v.set_pose(missing, _scores())

    v.set_show_joints(False)
    assert not any(it.isVisible() for it in v._joints)
    v.set_show_joints(True)
    assert v._joints[3].isVisible() and v._joints[3].is_placeholder
    assert all(it.isVisible() for it in v._joints)
