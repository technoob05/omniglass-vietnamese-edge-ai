#!/usr/bin/env bash
set -euo pipefail

# Build the MiniCPM-o Omni runtime against Ubuntu 22.04 / glibc 2.35 so it
# runs on the QCS_KALAMAP Ubuntu 22.04 image.  Run this on the H100 build host.
OCI_ROOT="${OCI_ROOT:-/network-volume/icse27/edge-ai/openglass-native/snapdragon-toolchain-oci}"
SOURCE="${SOURCE:-/network-volume/icse27/edge-ai/openglass-native/llama.cpp-omni}"
SYSROOT_DEBS="${SYSROOT_DEBS:-/network-volume/icse27/edge-ai/openglass-native/jammy-arm64-sysroot-debs}"
OUT_ROOT="${OUT_ROOT:-/network-volume/icse27/edge-ai/openglass-native/minicpmo-qcs8550-glibc235}"
REVISION="09f5c3f1b484759f17b06fc63574f749c89c8761"
ROOTFS="$(mktemp -d /tmp/minicpmo-qcs8550-glibc235.XXXXXX)"

cleanup() { rm -rf --one-file-system "$ROOTFS"; }
trap cleanup EXIT

[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$REVISION" ]]
mkdir -p "$ROOTFS/workspace" "$ROOTFS/opt/jammy-arm64-sysroot" "$OUT_ROOT"

python3 - "$OCI_ROOT" "$ROOTFS" <<'PY'
import json, pathlib, subprocess, sys
oci, root = map(pathlib.Path, sys.argv[1:])
manifest = json.loads((oci / "manifest.json").read_text())
for layer in manifest["layers"]:
    digest = layer["digest"].split(":", 1)[1]
    subprocess.check_call(["tar", "--no-same-owner", "-xzf", str(oci / digest), "-C", str(root)])
PY

git -C "$SOURCE" archive "$REVISION" | tar -x -C "$ROOTFS/workspace"
cp "$SOURCE/docs/backend/snapdragon/CMakeUserPresets.json" "$ROOTFS/workspace/CMakeUserPresets.json"
# The upstream Omni desktop entrypoints hard-code Token2Wav to gpu:0.  On the
# QCS Linux HTP backend that selects HTP0 even when -ngl 0 is requested, while
# Token2Wav's CPU thread controls require a CPU backend.  Keep the main model
# and native protocol unchanged; only make the auxiliary TTS chain CPU-safe.
python3 - "$ROOTFS/workspace" <<'PY'
from pathlib import Path
root = Path(__import__('sys').argv[1])
replacements = {
    'tools/omni/omni-cli.cpp': [
        ('omni_init(&params, media_type, use_tts, tts_bin_dir, -1, "gpu:0")',
         'omni_init(&params, media_type, use_tts, tts_bin_dir, 0, "cpu")'),
    ],
    'tools/omni/test/single_test_omni.cpp': [
        ('omni_init(&params, /*media_type=*/2, use_tts, tts_bin_dir, -1, "gpu:0")',
         'omni_init(&params, /*media_type=*/2, use_tts, tts_bin_dir, 0, "cpu")'),
    ],
    'tools/server/ws_handler.cpp': [
        ('/*tts_gpu_layers*/99,\n                                     /*token2wav_device*/"gpu:0", duplex_mode,',
         '/*tts_gpu_layers*/p.n_gpu_layers,\n                                     /*token2wav_device*/"cpu", duplex_mode,'),
    ],
}
for name, edits in replacements.items():
    path = root / name
    text = path.read_text(encoding='utf-8')
    for old, new in edits:
        if old not in text:
            raise SystemExit(f'missing patch anchor: {name}')
        text = text.replace(old, new, 1)
    path.write_text(text, encoding='utf-8')
PY
for deb in "$SYSROOT_DEBS"/*.deb; do dpkg-deb -x "$deb" "$ROOTFS/opt/jammy-arm64-sysroot"; done
for pair in 'null c 1 3' 'zero c 1 5' 'random c 1 8' 'urandom c 1 9'; do
  set -- $pair
  [[ -e "$ROOTFS/dev/$1" ]] || mknod -m 666 "$ROOTFS/dev/$1" "$2" "$3" "$4"
done

chroot "$ROOTFS" /usr/bin/env -i \
  HOME=/root USER=root PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH=/opt/hexagon/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools/lib \
  HEXAGON_SDK_ROOT=/opt/hexagon/6.4.0.2 \
  HEXAGON_TOOLS_ROOT=/opt/hexagon/6.4.0.2/tools/HEXAGON_Tools/19.0.04 \
  DEFAULT_DSP_ARCH=v73 OPENCL_SDK_ROOT=/usr/aarch64-linux-gnu \
  /bin/bash -lc '
    set -euo pipefail
    cd /workspace
    /usr/bin/clang --target=aarch64-linux-gnu --sysroot=/opt/jammy-arm64-sysroot \
      --gcc-toolchain=/opt/jammy-arm64-sysroot/usr -x c -dM -E -include features.h /dev/null \
      | grep -Eq "^#define __GLIBC_MINOR__ 35$"
    cmake --preset arm64-linux-snapdragon-release -B build-snapdragon \
      -DCMAKE_SYSROOT=/opt/jammy-arm64-sysroot \
      -DCMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN=/opt/jammy-arm64-sysroot/usr \
      -DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/opt/jammy-arm64-sysroot/usr
    cmake --build build-snapdragon --target llama-omni-cli llama-omni-server llama-omni-single-test-omni htp-v73 -j 12
    mkdir -p pkg-omni/bin pkg-omni/lib
    cp -a build-snapdragon/bin/llama-omni-cli build-snapdragon/bin/llama-omni-server build-snapdragon/bin/llama-omni-single-test-omni pkg-omni/bin/
    find build-snapdragon/bin -maxdepth 1 \( -type f -o -type l \) -name "*.so*" -exec cp -a -t pkg-omni/lib {} +
    find build-snapdragon/ggml/src/ggml-hexagon -maxdepth 1 -type f -name "libggml-htp-v*.so" -exec cp -a -t pkg-omni/lib {} +
    test -f pkg-omni/lib/libggml-htp-v73.so
  '

rm -rf "$OUT_ROOT/pkg-omni"
cp -a "$ROOTFS/workspace/pkg-omni" "$OUT_ROOT/pkg-omni"
printf 'runtime_revision=%s\ntarget_glibc=2.35\ntarget_dsp=v73\nqcs_t2w_device=cpu\nqcs_tts_gpu_layers=0\n' "$REVISION" > "$OUT_ROOT/pkg-omni/BUILD_EVIDENCE.txt"
find "$OUT_ROOT/pkg-omni" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT_ROOT/pkg-omni/SHA256SUMS"
tar -C "$OUT_ROOT/pkg-omni" -czf "$OUT_ROOT/pkg-omni-glibc235.tar.gz" .
sha256sum "$OUT_ROOT/pkg-omni-glibc235.tar.gz" > "$OUT_ROOT/pkg-omni-glibc235.tar.gz.sha256"
echo "Built: $OUT_ROOT/pkg-omni-glibc235.tar.gz"
