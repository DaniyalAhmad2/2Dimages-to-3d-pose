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
if ! ls "$VENDOR"/rtmlib-cache/*.onnx >/dev/null 2>&1; then
    echo "ERROR: no ONNX weights in $VENDOR/rtmlib-cache/" >&2
    echo "Copy them from ~/.cache/rtmlib/hub/checkpoints/ (yolox_m + rtmpose-m body7)." >&2
    exit 1
fi

echo "==> building $IMAGE"
docker build -f docker/Dockerfile -t "$IMAGE" -t pose3d:latest .

echo "==> smoke test"
docker run --rm --shm-size=1g --entrypoint bash "$IMAGE" -c '
Xvfb :99 -screen 0 1280x800x24 -nolisten tcp >/dev/null 2>&1 &
for _ in $(seq 1 100); do xdpyinfo -display :99 >/dev/null 2>&1 && break; sleep 0.1; done
export DISPLAY=:99
python /app/docker/smoketest.py'

if [ "${1:-}" = "--push" ]; then
    echo "==> pushing $IMAGE"
    docker push "$IMAGE"
    echo "Done. The client runs: docker compose up"
else
    echo
    echo "Built and verified. To publish:  ./docker/build.sh --push"
fi
