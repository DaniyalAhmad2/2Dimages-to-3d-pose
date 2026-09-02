#!/usr/bin/env bash
# Build, verify and publish the client image.
#
#   ./docker/build.sh            build + smoke test
#   ./docker/build.sh --push     build + smoke test + push to the registry
#
# Prerequisites for --push (one time):
#   echo <GITHUB_PAT> | docker login ghcr.io -u <github-username> --password-stdin
# The PAT needs the write:packages scope.
set -euo pipefail

IMAGE="${POSE3D_IMAGE:-ghcr.io/daniyalahmad2/pose3d:latest}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Blender and the pose weights are baked in; stage them if missing.
VENDOR="docker/vendor"
if ! ls "$VENDOR"/blender-*-linux-x64.tar.xz >/dev/null 2>&1; then
    echo "ERROR: no Blender tarball in $VENDOR/" >&2
    echo "Put blender-5.x.x-linux-x64.tar.xz there (export needs Blender 5.x)." >&2
    exit 1
fi
# Same staging step the Windows release runs, so both deliveries carry exactly
# the checkpoints rtmlib asks for rather than a hand-copied guess — both pose
# models (Halpe-26, which the app detects with, and the COCO-17 fallback) and
# the shared YOLOX detector. The image bakes the whole folder in, so nothing
# downloads at the client's run time; docker/smoketest.py runs the same
# pose3d.selftest check that fails on a missing one.
PY="${POSE3D_PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python3
if ! "$PY" tools/fetch_weights.py --out "$VENDOR/rtmlib-cache"; then
    echo "ERROR: could not stage the ONNX weights into $VENDOR/rtmlib-cache/" >&2
    exit 1
fi

echo "==> building $IMAGE"
docker build -f docker/Dockerfile -t "$IMAGE" -t pose3d:latest .

echo "==> smoke test"
docker run --rm --shm-size=1g --entrypoint bash "$IMAGE" -c '
Xvfb :99 -screen 0 1280x800x24 -nolisten tcp >/dev/null 2>&1 &
for _ in $(seq 1 100); do xdpyinfo -display :99 >/dev/null 2>&1 && break; sleep 0.1; done
export DISPLAY=:99
# --require-video: the image ships Mesa so headless Blender can render, so a
# missing preview video means the image is broken, not merely degraded. The
# Windows bundle deliberately does NOT require it — no software GL there.
python /app/docker/smoketest.py --require-video'

if [ "${1:-}" = "--push" ]; then
    echo "==> pushing $IMAGE"
    docker push "$IMAGE"
    echo "Done. The client runs: docker compose up"
else
    echo
    echo "Built and verified. To publish:  ./docker/build.sh --push"
fi
