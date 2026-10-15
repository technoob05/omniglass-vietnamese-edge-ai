#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to an isolated Linux build directory}"
REVISION="09f5c3f1b484759f17b06fc63574f749c89c8761"
BRANCH="master"
REPOSITORY="https://github.com/tc-mb/llama.cpp-omni.git"
TOOLCHAIN_IMAGE="${SNAPDRAGON_TOOLCHAIN_IMAGE:-ghcr.io/snapdragon-toolchain/arm64-linux:v0.1}"
BUILD_OMNI="${BUILD_OMNI:-1}"
BUILD_LLAMA_CLI="${BUILD_LLAMA_CLI:-0}"
BUILD_RESIDENT_SERVER="${BUILD_RESIDENT_SERVER:-1}"
SOURCE="${WORK_ROOT}/llama.cpp-omni-${REVISION:0:12}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESIDENT_SERVER_FRAGMENT="${SCRIPT_DIR}/../native-overrides/qcs8550/llama_server_target.cmake"

for tool in git docker; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "Missing required tool: ${tool}" >&2; exit 2; }
done
[[ "$(uname -s)" == "Linux" ]] || { echo "Run this cross-build on a Linux x86 host." >&2; exit 2; }
mkdir -p "${WORK_ROOT}"

if [[ ! -d "${SOURCE}/.git" ]]; then
  git clone --filter=blob:none --branch "${BRANCH}" --single-branch "${REPOSITORY}" "${SOURCE}"
fi
git -C "${SOURCE}" fetch --depth=1 origin "${REVISION}"
git -C "${SOURCE}" checkout --detach FETCH_HEAD
[[ "$(git -C "${SOURCE}" rev-parse HEAD)" == "${REVISION}" ]] || {
  echo "Runtime revision mismatch." >&2
  exit 2
}

# The pinned Linux preset enables CPU + Hexagon and explicitly disables OpenCL.
grep -q '"GGML_HEXAGON".*"ON"' "${SOURCE}/docs/backend/snapdragon/CMakeUserPresets.json"
grep -q '"GGML_OPENCL".*"OFF"' "${SOURCE}/docs/backend/snapdragon/CMakeUserPresets.json"

if [[ "${BUILD_RESIDENT_SERVER}" == "1" ]]; then
  [[ -f "${RESIDENT_SERVER_FRAGMENT}" ]] || { echo "Missing resident server CMake fragment." >&2; exit 2; }
  cat "${RESIDENT_SERVER_FRAGMENT}" >> "${SOURCE}/tools/server/CMakeLists.txt"
fi

docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --volume "${SOURCE}:/workspace" \
  --env BUILD_OMNI="${BUILD_OMNI}" \
  --env BUILD_LLAMA_CLI="${BUILD_LLAMA_CLI}" \
  --env BUILD_RESIDENT_SERVER="${BUILD_RESIDENT_SERVER}" \
  "${TOOLCHAIN_IMAGE}" \
  bash -lc '
    set -euo pipefail
    cd /workspace
    cp docs/backend/snapdragon/CMakeUserPresets.json .
    cmake --preset arm64-linux-snapdragon-release -B build-snapdragon \
      -DLLAMA_BUILD_SERVER=ON
    cmake --build build-snapdragon \
      --target llama-bench llama-mtmd-cli \
      -j "$(nproc)"
    cmake --build build-snapdragon --target htp-v73 -j "$(nproc)"
    # The pinned fork llama-cli includes server-common.h without exporting
    # the mtmd include directory. It is optional here: llama-bench supports the
    # same --list-devices probe needed by the board harness.
    if [[ "${BUILD_LLAMA_CLI}" == "1" ]]; then
      cmake --build build-snapdragon --target llama-cli -j "$(nproc)"
    fi
    if [[ "${BUILD_RESIDENT_SERVER}" == "1" ]]; then
      cmake --build build-snapdragon --target llama-server -j "$(nproc)"
    fi
    if [[ "${BUILD_OMNI}" == "1" ]]; then
      cmake --build build-snapdragon \
        --target llama-omni-cli llama-omni-server llama-omni-single-test-omni \
        -j "$(nproc)"
    fi
    # cmake --install does not build missing installable tools. This repository has
    # many unrelated install targets, so stage the audited runtime closure from the
    # explicitly built targets instead of forcing a slow full-tree build.
    mkdir -p pkg-snapdragon/bin pkg-snapdragon/lib
    cp -a build-snapdragon/bin/llama-bench \
          build-snapdragon/bin/llama-mtmd-cli \
          pkg-snapdragon/bin/
    if [[ "${BUILD_LLAMA_CLI}" == "1" ]]; then
      cp -a build-snapdragon/bin/llama-cli pkg-snapdragon/bin/
    fi
    if [[ "${BUILD_RESIDENT_SERVER}" == "1" ]]; then
      cp -a build-snapdragon/bin/llama-server pkg-snapdragon/bin/
    fi
    if [[ "${BUILD_OMNI}" == "1" ]]; then
      cp -a build-snapdragon/bin/llama-omni-cli \
            build-snapdragon/bin/llama-omni-server \
            build-snapdragon/bin/llama-omni-single-test-omni \
            pkg-snapdragon/bin/
    fi
    find build-snapdragon/bin -maxdepth 1 \( -type f -o -type l \) -name "*.so*" \
      -exec cp -a -t pkg-snapdragon/lib {} +
    find build-snapdragon/ggml/src/ggml-hexagon -maxdepth 1 -type f -name "libggml-htp-v*.so" \
      -exec cp -a -t pkg-snapdragon/lib {} +
    test -f pkg-snapdragon/lib/libggml-htp-v73.so

    readelf_bin="$(command -v llvm-readelf || command -v aarch64-linux-gnu-readelf || command -v readelf)"
    : > pkg-snapdragon/RUNTIME_DEPENDENCIES.txt
    for binary in pkg-snapdragon/bin/*; do
      echo "### ${binary}" >> pkg-snapdragon/RUNTIME_DEPENDENCIES.txt
      "${readelf_bin}" -d "${binary}" | grep -E "NEEDED|RPATH|RUNPATH" \
        >> pkg-snapdragon/RUNTIME_DEPENDENCIES.txt
    done
    find pkg-snapdragon/bin pkg-snapdragon/lib -maxdepth 1 \( -type f -o -type l \) \
      -printf "%p\n" | sort > pkg-snapdragon/RUNTIME_FILES.txt
  '

{
  echo "runtime_repository=${REPOSITORY}"
  echo "runtime_revision=${REVISION}"
  echo "toolchain_image=${TOOLCHAIN_IMAGE}"
  docker image inspect "${TOOLCHAIN_IMAGE}" --format 'toolchain_image_id={{.Id}} toolchain_repo_digests={{join .RepoDigests ","}}' 2>/dev/null || true
  echo "linux_preset_backends=CPU,Hexagon"
  echo "linux_preset_opencl=OFF"
  echo "full_omni_targets_built=${BUILD_OMNI}"
  echo "llama_cli_built=${BUILD_LLAMA_CLI}"
  echo "resident_multimodal_server_built=${BUILD_RESIDENT_SERVER}"
} > "${SOURCE}/pkg-snapdragon/BUILD_EVIDENCE.txt"

echo "QCS8550 ARM64 package: ${SOURCE}/pkg-snapdragon"
echo "Cross-build success is not QCS8550 execution proof; HTP operator placement and performance remain unverified."
