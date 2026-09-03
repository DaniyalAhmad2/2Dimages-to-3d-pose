@echo off
rem Pose3D diagnostics, for when the app itself cannot tell you anything.
rem
rem Double-click this file. It runs the self-test in a window that stays open
rem and then prints the log, so there is something to read and copy into an
rem email even when nothing else in the bundle will start.
rem
rem It exists as the fallback for Pose3D-diagnose.exe: that build prints the
rem same self-test to a console directly, but if it is missing or refuses to
rem run, Pose3D.exe is windowed and writes its output ONLY into
rem pose3d-log.txt, which is why running --selftest by hand shows nothing.
setlocal
cd /d "%~dp0"

echo Pose3D diagnostics
echo ==================
echo Folder: %CD%
echo.

if exist "Pose3D-diagnose.exe" (
    echo Running the self-test ^(Pose3D-diagnose.exe^)...
    echo.
    "Pose3D-diagnose.exe" --selftest --no-video
) else (
    if exist "Pose3D.exe" (
        echo Pose3D-diagnose.exe is not in this folder; running Pose3D.exe
        echo instead. It is a windowed build, so its output goes to the log
        echo printed below rather than here.
        echo.
        "Pose3D.exe" --selftest --no-video
    ) else (
        echo Neither Pose3D-diagnose.exe nor Pose3D.exe is in this folder.
        echo This copy of the bundle is incomplete: extract Pose3D-Windows.zip
        echo again, to a short path on a local disk such as C:\Pose3D.
    )
)

echo.
echo ---------------- pose3d-log.txt ----------------
if exist "pose3d-log.txt" (
    type "pose3d-log.txt"
) else (
    echo No log file has been written in this folder.
)
echo ------------------------------------------------
echo.
echo Copy everything above into your email to us.
pause
