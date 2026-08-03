"""In-container smoke test.

    docker run --rm --entrypoint python pose3d:latest /app/docker/smoketest.py

The checks themselves live in pose3d.selftest, shared with the Windows bundle
(`Pose3D.exe --selftest`), so a fix to one delivery cannot leave the other
untested. This file only exists so docker/build.sh has a stable path to run.
"""
import sys

from pose3d.selftest import main

raise SystemExit(main(sys.argv[1:]))
