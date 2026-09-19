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

rem ---- Checks that need no Python at all. They run even when python312.dll
rem ---- will not load, which is exactly the failure Pose3D-diagnose.exe below
rem ---- cannot report, because it is a Python program too and dies the same
rem ---- way. "Failed to load Python DLL ... python312.dll ... could not be
rem ---- found" means one of THESE files is missing or empty, not that the
rem ---- client's Windows is wrong.
echo Files Windows loads before any Python runs:
set "PRE_OK=1"
call :need "_internal\python312.dll"      "the Python interpreter"
call :need "_internal\vcruntime140.dll"   "the Visual C++ runtime"
call :need "_internal\vcruntime140_1.dll" "the Visual C++ runtime, part 2"
call :need "_internal\msvcp140.dll"       "the C++ standard library"
call :need "_internal\ucrtbase.dll"       "the Universal C Runtime"
call :need "_internal\api-ms-win-crt-runtime-l1-1-0.dll" "the C runtime forwarders"
if defined PROCESSOR_ARCHITEW6432 (set "ARCH=%PROCESSOR_ARCHITEW6432%") else (set "ARCH=%PROCESSOR_ARCHITECTURE%")
if /i "%ARCH%"=="AMD64" (
    echo   ok       64-bit Windows ^(%ARCH%^)
) else (
    echo   PROBLEM  this Windows is %ARCH%. Pose3D needs 64-bit ^(AMD64^) Windows 10 or 11.
    set "PRE_OK=0"
)
rem Mark of the Web: Windows tags every file extracted from an internet
rem download with the built-in extractor, and some security settings refuse
rem to load DLLs that carry the tag. Unblocking the .zip before extracting
rem clears it; this only detects it.
more <"Pose3D.exe:Zone.Identifier" >nul 2>nul && (
    echo   NOTE     Pose3D.exe is marked as downloaded from the internet.
    echo            If the app will not start: right-click the .zip, Properties,
    echo            tick Unblock, OK, then extract it again.
)
if "%PRE_OK%"=="1" (
    echo   OK       every file Windows loads first is present.
) else (
    echo.
    echo   A file above is MISSING or EMPTY. Usual causes: the extraction
    echo   stopped early, an anti-virus program removed it, or the folder is
    echo   under OneDrive. Extract Pose3D-Windows.zip again to C:\Pose3D, and
    echo   look in Windows Security ^> Protection history for anything from
    echo   _internal\ and restore it.
)
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
goto :eof

rem ---- :need <relative path> <what it is>: present and not empty, or say so.
:need
if not exist "%~1" (
    echo   MISSING  %~1  ^(%~2^)
    set "PRE_OK=0"
    goto :eof
)
for %%F in ("%~1") do set "SZ=%%~zF"
if "%SZ%"=="0" (
    echo   EMPTY    %~1  ^(%~2, 0 bytes^)
    set "PRE_OK=0"
) else (
    echo   ok       %~1
)
goto :eof
