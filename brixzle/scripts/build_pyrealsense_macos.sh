#!/usr/bin/env bash
# Build pyrealsense2 from Intel's source for this venv (macOS). The community `pyrealsense2-macosx`
# wheel (2.56.5) segfaults on macOS 26 when it opens the D455; librealsense 2.58.4 (as in Homebrew)
# works under sudo. The module goes to ~/.local/share/pyrealsense2-<ver> and a .pth file points the
# venv at it, so `uv sync` leaves it alone.
#
#   bash scripts/build_pyrealsense_macos.sh            # ~10 min, no sudo
#   sudo .venv/bin/testflight capture --out runs/cap1  # the camera itself needs root on macOS
set -euo pipefail
VER=${VER:-2.58.4}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY="$HERE/.venv/bin/python"
WORK=${WORK:-$(mktemp -d)}
DEST="$HOME/.local/share/pyrealsense2-$VER"

git clone --depth 1 --branch "v$VER" https://github.com/realsenseai/librealsense.git "$WORK/src"
uvx --from cmake cmake -S "$WORK/src" -B "$WORK/build" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE="$PY" -DPython_EXECUTABLE="$PY" \
  -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_WITH_OPENMP=OFF -DCHECK_FOR_UPDATES=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_OSX_ARCHITECTURES=arm64
uvx --from cmake cmake --build "$WORK/build" --target pyrealsense2 -j "$(sysctl -n hw.ncpu)"

rm -rf "$DEST" && mkdir -p "$DEST/pyrealsense2"
find "$WORK/build" \( -name 'pyrealsense2*.so' -o -name 'librealsense2*.dylib' \) -exec cp -a {} "$DEST/pyrealsense2/" \;
cp "$WORK/src/wrappers/python/pyrealsense2/__init__.py" "$DEST/pyrealsense2/" 2>/dev/null || \
  printf 'from .pyrealsense2 import *\n' > "$DEST/pyrealsense2/__init__.py"
for f in "$DEST"/pyrealsense2/*.so "$DEST"/pyrealsense2/*.dylib; do   # load from its own folder, not the build
  [ -L "$f" ] && continue
  install_name_tool -delete_rpath "$WORK/build/Release" "$f" 2>/dev/null || true
  install_name_tool -add_rpath @loader_path "$f" 2>/dev/null || true
  codesign -f -s - "$f"
done
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$DEST" > "$SITE/pyrealsense2_source_build.pth"
"$PY" -c 'import pyrealsense2 as rs; print("pyrealsense2 from", rs.__file__)'
