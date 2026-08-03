# Pose3D — self-contained desktop app served to the browser.
#
# The app is a Qt/OpenGL desktop program, so the image runs a virtual display
# (Xvfb) with software OpenGL (Mesa llvmpipe) and publishes the window over
# noVNC. That is what makes it work identically on Windows, macOS and Linux.
#
# Blender 5.x and the RTMPose weights are baked in, so the client needs no
# network and no separate installs.
# AWS's mirror of the Docker Official Image. Same content as
# docker.io/library/python, but reachable from networks that cannot resolve
# registry-1.docker.io. Override with:
#   docker build --build-arg PYTHON_IMAGE=python:3.12-slim-bookworm ...
ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- system runtime -------------------------------------------------------
#  * xvfb/x11vnc/novnc/websockify : virtual display + browser delivery
#  * mesa (libgl1-mesa-dri)       : software OpenGL for the 3D view
#  * libxcb-*/libxkbcommon-x11    : Qt's xcb platform plugin
#  * libx*/libsm/libice           : also required by headless Blender
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11vnc novnc websockify x11-utils procps \
        libgl1 libglx-mesa0 libgl1-mesa-dri libegl1 libglu1-mesa \
        libxkbcommon0 libxkbcommon-x11-0 libdbus-1-3 \
        libxcb1 libx11-xcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 \
        libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
        libxcb-cursor0 libxcb-glx0 libxcb-util1 \
        libx11-6 libxext6 libxrender1 libxi6 libxfixes3 libxxf86vm1 \
        libsm6 libice6 libfontconfig1 libfreetype6 \
        fonts-dejavu-core xz-utils ca-certificates \
        adwaita-icon-theme \
    && rm -rf /var/lib/apt/lists/*

# --- Blender (export engine) ---------------------------------------------
# Must be 5.x: the video writer sets image_settings.media_type and the BVH
# exporter signature changed, neither of which exist in 4.x.
COPY docker/vendor/blender-*-linux-x64.tar.xz /tmp/blender.tar.xz
RUN mkdir -p /opt \
    && tar -xJf /tmp/blender.tar.xz -C /opt \
    && mv /opt/blender-*-linux-x64 /opt/blender \
    && rm -f /tmp/blender.tar.xz \
    && rm -rf /opt/blender/*/datafiles/locale \
    && /opt/blender/blender --version

# --- Python dependencies --------------------------------------------------
# Installed from the committed lockfile, hashes and all, so this image and the
# Windows bundle contain the same versions the test suite ran against. This
# used to be a hand-written list here, which silently drifted a whole major
# OpenCV version ahead of the one anything was tested on.
#
# The constraints themselves live in pyproject.toml, including why opencv must
# be the *contrib* build (cv2.aruco, for calibration) and *headless* (a second
# Qt collides with PySide6), and why rtmlib is stripped of its dependencies.
#
# --no-deps is required, not an optimisation: the exported file is already the
# lock's complete pinned closure, and without it pip re-reads rtmlib's own
# metadata and reinstates the plain opencv build, shadowing contrib and taking
# cv2.aruco — calibration — with it. uv's dependency override governs uv's
# resolver only; it does not follow the package into pip.
COPY pyproject.toml uv.lock /tmp/dep/
RUN pip install --no-cache-dir uv \
    && uv export --frozen --no-dev --no-emit-project \
         --directory /tmp/dep -o /tmp/dep/requirements.txt \
    && pip install --no-cache-dir --no-deps -r /tmp/dep/requirements.txt \
    && pip uninstall -y uv \
    && rm -rf /tmp/dep

# --- pose detection weights ----------------------------------------------
# Baked in so first use is instant and works offline. rtmlib looks under
# $XDG_CACHE_HOME/rtmlib, which we point at a read-only system location so it
# is unaffected by HOME (which we hand to the mounted workspace).
COPY docker/vendor/rtmlib-cache/ /opt/cache/rtmlib/hub/checkpoints/
ENV XDG_CACHE_HOME=/opt/cache

# --- application ----------------------------------------------------------
WORKDIR /app
COPY pose3d/ /app/pose3d/
COPY docker/smoketest.py /app/docker/smoketest.py
COPY docker/entrypoint.sh /usr/local/bin/pose3d-entrypoint
COPY docker/index.html /usr/share/novnc/index.html
RUN chmod +x /usr/local/bin/pose3d-entrypoint

# Software GL for the 3D view; the container has no GPU.
# PYTHONPATH is needed because the working directory is the client's
# /workspace, so the app must stay importable from an absolute location.
ENV LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    QT_X11_NO_MITSHM=1 \
    QT_QPA_PLATFORM=xcb \
    POSE3D_BLENDER=/opt/blender/blender \
    POSE3D_CONTAINER=1 \
    POSE3D_MAXIMIZE=1 \
    HOME=/workspace \
    PYTHONPATH=/app

# The client's files live here (bind-mounted); projects default to
# $HOME/pose3d_projects, i.e. /workspace/pose3d_projects.
RUN mkdir -p /workspace
VOLUME ["/workspace"]
WORKDIR /workspace

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/pose3d-entrypoint"]
