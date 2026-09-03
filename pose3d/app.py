"""Application entry point.

Usage:
    python -m pose3d.app [PROJECT_FOLDER]

Loads a project (or an empty session), builds the dashboard, runs the Qt event
loop. The Import wizard (top-bar button) creates a new project from uploaded
images + calibration and opens it in a fresh window.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# keep window references so they are not garbage-collected
_WINDOWS: list = []


def apply_dark_theme(app: QApplication) -> None:
    """Make the app dark everywhere, not just where the stylesheet reaches.

    dark.qss covers the widgets we style by name, but Qt draws everything else
    — splitter handles, tool tips, menus, combo popups, disabled text, focus
    rings — from the *system* palette and with the host's native style. On a
    Windows machine set to light mode that means light chrome punched through a
    dark UI: the splitter handles vanished against the panels, and the accuracy
    gauge was filled with the native window colour behind near-white text.

    Fusion is the one style that honours a supplied palette identically on
    every platform, so pin both rather than inheriting whatever the desktop is
    set to. This is a dark-themed application by design; it should not change
    appearance with the user's OS setting.
    """
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor("#0d0f15"))
    p.setColor(QPalette.ColorRole.WindowText, QColor("#e6e8ee"))
    p.setColor(QPalette.ColorRole.Base, QColor("#0f1219"))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor("#12151d"))
    p.setColor(QPalette.ColorRole.Text, QColor("#e6e8ee"))
    p.setColor(QPalette.ColorRole.Button, QColor("#161a24"))
    p.setColor(QPalette.ColorRole.ButtonText, QColor("#e6e8ee"))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor("#12151d"))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor("#e6e8ee"))
    p.setColor(QPalette.ColorRole.Highlight, QColor("#3d7bfd"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor("#8a91a3"))
    p.setColor(QPalette.ColorRole.Link, QColor("#6ea8ff"))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor("#5a6072"))
    app.setPalette(p)

    qss = Path(__file__).parent / "ui" / "dark.qss"
    if qss.exists():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))


# kept: older call sites (and tests) refer to this name
load_stylesheet = apply_dark_theme


def build_model(project_folder: str | None):
    from pose3d.core.io_project import load_project
    from pose3d.core.project import ProjectData
    from pose3d.ui.model import ProjectModel

    if project_folder and Path(project_folder).exists():
        project = load_project(project_folder)
        rig, reason = load_rig_with_reason(
            Path(project_folder) / "calibration")
        model = ProjectModel(project, rig, project_dir=project_folder)
        # the sidebar states it: a broken calibration must not look like no
        # calibration, and neither must look like a broken 3D view
        model.rig_error = reason
        # A take saved by an older pipeline is corrected here, once, before
        # anything is drawn — the window then shows what changed and offers
        # the stored pose back. Nothing is written to disk.
        model.upgrade_pipeline()
        return model
    project = ProjectData(name="No project loaded")
    return ProjectModel(project, None)


def _check_extrinsics(cam: str, d) -> tuple:
    """(R, t) for one camera, or raise ValueError naming what is wrong.

    THE implementation is `calib.rigio.check_extrinsics`, so the headless
    loaders every non-UI caller uses reject the same file for the same reason;
    this name is kept because the UI's own reason-reporting reads it.
    """
    from pose3d.calib.rigio import check_extrinsics
    return check_extrinsics(cam, d)


def load_rig_with_reason(calib_dir: Path):
    """(CalibratedRig | None, reason). The reason is "" when there is nothing
    to say — an uncalibrated project is a normal state, not a fault.

    Everything else used to be swallowed by one blanket `except Exception`
    that returned None, so a calibration folder with a typo in it was reported
    to the user in exactly the same way as no calibration folder at all: the
    3D view was empty and nothing anywhere said why.
    """
    import json

    from pose3d.calib.extrinsics import Extrinsics
    from pose3d.calib.intrinsics import Intrinsics
    from pose3d.pipeline import CalibratedRig
    try:
        il = Intrinsics.load(calib_dir / "left_intrinsics.json")
        ir = Intrinsics.load(calib_dir / "right_intrinsics.json")
        ext = json.loads(
            (calib_dir / "extrinsics.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, ""                  # genuinely uncalibrated: stay quiet
    except json.JSONDecodeError as e:
        return None, (f"The calibration in this project could not be read: "
                      f"one of its files is not valid JSON ({e}).")
    except (KeyError, ValueError, TypeError, OSError) as e:
        return None, (f"The calibration in this project could not be read "
                      f"({type(e).__name__}: {e}).")
    try:
        left = _check_extrinsics("left", ext["left"])
        right = _check_extrinsics("right", ext["right"])
    except KeyError as e:
        return None, (f"extrinsics.json is missing the {e} camera, so no 3D "
                      f"can be reconstructed.")
    except (TypeError, ValueError) as e:
        return None, f"This project's calibration is not usable — {e}."
    el = Extrinsics(R=left[0], t=left[1])
    er = Extrinsics(R=right[0], t=right[1])
    return CalibratedRig(il, ir, el, er), ""


def _load_rig(calib_dir: Path):
    """Load a CalibratedRig from a calibration folder if fully present."""
    return load_rig_with_reason(calib_dir)[0]


def open_project_window(project_folder: str | None):
    """Build a model + MainWindow for a project folder and show it."""
    from pose3d.ui.main_window import MainWindow
    model = build_model(project_folder)
    win = MainWindow(model, open_callback=open_project_window)
    _WINDOWS.append(win)
    # In the container the app IS the desktop, so fill the virtual screen rather
    # than floating a fixed-size window inside it. showMaximized() is no use
    # here: maximising is a window-manager job and the container runs none, so
    # set the geometry outright.
    if os.environ.get("POSE3D_MAXIMIZE", "0") == "1":
        from PySide6.QtWidgets import QApplication
        qapp = QApplication.instance()
        screen = qapp.primaryScreen() if qapp is not None else None
        if screen is not None:
            win.setGeometry(screen.availableGeometry())
    win.show()
    return win


def install_crash_handler() -> None:
    """Make an unhandled exception visible instead of silent.

    Started from a shortcut with no console, a crash otherwise just closes the
    window: the client sees the app "not open" and has nothing to send us. Log
    it and say where the log is.
    """
    import traceback

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text, file=sys.stderr, flush=True)
        try:
            from PySide6.QtWidgets import QApplication

            from pose3d.ui.guard import log_hint, report_error
            if QApplication.instance() is not None:
                # Through the sink, not QMessageBox: this is a modal like any
                # other, and the suite's autouse fixture replaces exactly one
                # function to keep every one of them off a CI machine.
                report_error(None, "Pose3D stopped",
                             f"{exc_type.__name__}: {exc}\n\n{log_hint()}")
        except Exception:
            pass                        # a dialog must never mask the crash

    sys.excepthook = hook


def install_qt_message_handler() -> None:
    """Send Qt's own diagnostics to the same place as everything else.

    Qt writes "Failed to create OpenGL context", "Could not load the Qt
    platform plugin 'windows'" and every other diagnosis of the failures this
    build is most likely to hit through its message handler, which defaults to
    a console — and a windowed exe has none, so the client's most useful
    error messages went nowhere at all.
    """
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    levels = {
        QtMsgType.QtDebugMsg: "debug",
        QtMsgType.QtInfoMsg: "info",
        QtMsgType.QtWarningMsg: "warning",
        QtMsgType.QtCriticalMsg: "critical",
        QtMsgType.QtFatalMsg: "fatal",
    }

    def handler(mode, context, message):
        print(f"[qt.{levels.get(mode, 'message')}] {message}",
              file=sys.stderr, flush=True)

    qInstallMessageHandler(handler)


def _pre_qt_checks() -> None:
    """Checks that must run before Qt exists, and may exit instead of starting.

    The bundle-integrity check has to run before Qt is asked to load a plugin
    that may be missing from the bundle — a missing `platforms\\qwindows.dll`
    aborts the QApplication constructor itself, so no Qt dialog could report
    it — and this is the point in `main()` where "before" is guaranteed: after
    the crash handler, so a failure inside it is still reported, and after
    `--selftest` has taken its own exit, so the self-test's report is never
    pre-empted by a dialog. Imported inside the function, like the rest of
    what `main()` reaches for, so importing this module stays cheap.
    """
    from pose3d import integrity
    integrity.run_startup_check()


def _configure_gl(argv) -> str:
    """Decide hardware or software OpenGL. MUST run before QApplication.

    Qt reads QT_OPENGL and AA_UseSoftwareOpenGL when the application object is
    constructed and ignores both afterwards, so this cannot be a reaction to a
    3D view that turned out black — it has to be a decision taken from three
    things known beforehand: the command line, the environment, and a marker
    file the "Restart with software 3D" button leaves next to the exe so the
    choice survives the restart that applies it.

    Honest limit, recorded in DECISIONS: this makes *Qt* run on software
    OpenGL. pyqtgraph draws through PyOpenGL, which loads the machine's own
    opengl32.dll and is not redirected by Qt's opengl32sw.dll, so the 3D card
    still depends on the driver being there.
    """
    from PySide6.QtCore import QCoreApplication, Qt

    from pose3d.runtime import SOFTWARE_GL_MARKER, app_dir

    if not ("--software-gl" in argv
            or os.environ.get("POSE3D_GL", "").lower() == "software"
            or (app_dir() / SOFTWARE_GL_MARKER).exists()):
        return "hardware"
    os.environ["QT_OPENGL"] = "software"
    QCoreApplication.setAttribute(
        Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)
    return "software"


def main():
    # A frozen windowed build has no console to inherit, so anything that
    # writes to stdout/stderr — rtmlib's download progress, a Qt warning, a
    # traceback — hits None and raises. Give them a file to land in first.
    from pose3d.runtime import ensure_std_streams
    ensure_std_streams()
    install_crash_handler()
    install_qt_message_handler()
    # Before the --selftest branch: the self-test builds its own QApplication,
    # and `Pose3D.exe --selftest --software-gl` is how a 3D fault on the
    # client's machine gets diagnosed, so it has to run under the same GL the
    # app would have used.
    _configure_gl(sys.argv[1:])

    if "--selftest" in sys.argv[1:]:
        from pose3d.selftest import main as selftest
        sys.exit(selftest([a for a in sys.argv[1:] if a != "--selftest"]))

    # Before _pre_qt_checks, like --selftest and for a sharper reason: a
    # bundle with files missing is exactly when the client is asked to run
    # this, and that check ends the process with a dialog.
    if "--diagnose" in sys.argv[1:]:
        from pose3d.diagnostics import main as diagnose
        sys.exit(diagnose())

    _pre_qt_checks()

    folder = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    # Share one GL context across windows so pyqtgraph's cached shader programs
    # stay valid when the Import flow opens a second window (otherwise
    # glUseProgram raises GLError 1281 on the new context).
    from PySide6.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    # Containers ship no desktop environment, so Qt has no icon theme to fall
    # back on and file-dialog toolbar buttons render blank. Name one explicitly
    # when the image provides it.
    from PySide6.QtGui import QIcon
    if not QIcon.themeName() and os.path.isdir("/usr/share/icons/Adwaita"):
        QIcon.setThemeName("Adwaita")
    apply_dark_theme(app)
    open_project_window(folder)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
