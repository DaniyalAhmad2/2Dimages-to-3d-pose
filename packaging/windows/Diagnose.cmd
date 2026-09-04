@echo off
rem Pose3D diagnostics, for when the app itself cannot tell you anything.
rem
rem Double-click this file. It writes pose3d-diagnostics.txt next to the app,
rem prints the same report in a window that stays open, and then prints the
rem log, so there is something to read and to copy into an email even when
rem nothing else in the bundle will start.
rem
rem It runs Pose3D-diagnose.exe --diagnose. That is the console build, and
rem --diagnose (not --selftest) is what writes pose3d-diagnostics.txt: the
rem self-test is a verdict, this is the description we need to see. Pose3D.exe
rem is windowed and writes its output ONLY into pose3d-log.txt, which is why
rem running anything by hand with it appears to show nothing.
setlocal
cd /d "%~dp0"

echo Pose3D diagnostics
echo ==================
echo Folder: %CD%
echo.

if exist "Pose3D-diagnose.exe" (
    echo Running the diagnostics ^(Pose3D-diagnose.exe --diagnose^)...
    echo.
    "Pose3D-diagnose.exe" --diagnose
) else (
    if exist "Pose3D.exe" (
        echo Pose3D-diagnose.exe is not in this folder; running Pose3D.exe
        echo instead. It is a windowed build, so its output goes to the log
        echo printed below rather than here.
        echo.
        "Pose3D.exe" --diagnose
    ) else (
        echo Neither Pose3D-diagnose.exe nor Pose3D.exe is in this folder.
        echo This copy of the bundle is incomplete: extract Pose3D-Windows.zip
        echo again, to a short path on a local disk such as C:\Pose3D.
    )
)

echo.
echo ---------------- pose3d-log.txt ----------------
rem The log is written next to the exe when that folder is writable, and in
rem %LOCALAPPDATA%\Pose3D when it is not, which is every install under
rem Program Files. Look in both, or this prints "no log file" while the log
rem exists.
if exist "pose3d-log.txt" (
    type "pose3d-log.txt"
) else (
    if exist "%LOCALAPPDATA%\Pose3D\pose3d-log.txt" (
        echo ^(from %LOCALAPPDATA%\Pose3D^)
        type "%LOCALAPPDATA%\Pose3D\pose3d-log.txt"
    ) else (
        echo No log file has been written in this folder, and none in
        echo %LOCALAPPDATA%\Pose3D either.
    )
)
echo ------------------------------------------------
echo.
echo Send us pose3d-diagnostics.txt ^(written next to this file, or in
echo %LOCALAPPDATA%\Pose3D^) and copy everything above into the email.
pause
