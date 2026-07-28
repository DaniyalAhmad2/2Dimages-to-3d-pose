#!/usr/bin/env bash
# Bring up a virtual display + VNC + web gateway, then run the desktop app.
# The client opens http://localhost:8080 and sees the real Qt window.
set -euo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-99}"
export DISPLAY=":${DISPLAY_NUM}"
GEOMETRY="${POSE3D_GEOMETRY:-1600x1000x24}"
WEB_PORT="${POSE3D_PORT:-8080}"

cleanup() { pkill -P $$ >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "[pose3d] starting virtual display ${DISPLAY} (${GEOMETRY})"
Xvfb "${DISPLAY}" -screen 0 "${GEOMETRY}" -nolisten tcp &

for _ in $(seq 1 100); do
    xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1 && break
    sleep 0.1
done
if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    echo "[pose3d] ERROR: virtual display failed to start" >&2
    exit 1
fi

echo "[pose3d] starting VNC server"
x11vnc -display "${DISPLAY}" -forever -shared -nopw -quiet -noxdamage \
       -rfbport 5900 -bg >/dev/null

echo "[pose3d] serving UI on http://localhost:${WEB_PORT}"
websockify --web=/usr/share/novnc "${WEB_PORT}" localhost:5900 >/dev/null 2>&1 &

# Sanity-check the GL stack the 3D view depends on; warn rather than die so the
# user still gets a window and a readable message.
if ! python -c "
import OpenGL.GL  # noqa
" >/dev/null 2>&1; then
    echo '[pose3d] WARNING: PyOpenGL import failed; the 3D view may not render' >&2
fi

echo "[pose3d] launching application"
echo "[pose3d] ---------------------------------------------------------------"
echo "[pose3d]  Open  http://localhost:${WEB_PORT}  in your browser"
echo "[pose3d]  Your files are under /workspace (mapped to the folder you ran"
echo "[pose3d]  this from). Projects are saved to /workspace/pose3d_projects."
echo "[pose3d] ---------------------------------------------------------------"

exec python -m pose3d.app "$@"
