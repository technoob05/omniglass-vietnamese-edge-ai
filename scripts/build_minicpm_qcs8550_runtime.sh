#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to an isolated Linux build directory}"
REVISION="09f5c3f1b484759f17b06fc63574f749c89c8761"
BRANCH="master"
REPOSITORY="https://github.com/tc-mb/llama.cpp-omni.git"
TOOLCHAIN_IMAGE="${SNAPDRAGON_TOOLCHAIN_IMAGE:-ghcr.io/snapdragon-toolchain/arm64-linux:v0.1}"
BUILD_OMNI="${BUILD_OMNI:-1}"
SOURCE="${WORK_ROOT}/llama.cpp-omni-${REVISION:0:12}"

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

docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --volume "${SOURCE}:/workspace" \
  --env BUILD_OMNI="${BUILD_OMNI}" \
  "${TOOLCHAIN_IMAGE}" \
  bash -lc '
    set -euo pipefail
    cd /workspace
    cp docs/backend/snapdragon/CMakeUserPresets.json .
    cmake --preset arm64-linux-snapdragon-release -B build-snapdragon
    cmake --build build-snapdragon \
      --target llama-cli llama-bench llama-mtmd-cli \
      -j "$(nproc)"
    if [[ "${BUILD_OMNI}" == "1" ]]; then
      cmake --build build-snapdragon \
        --target llama-omni-cli llama-omni-server llama-omni-single-test-omni \
        -j "$(nproc)"
    fi
    cmake --install build-snapdragon --prefix pkg-snapdragon
  '

{
  echo "runtime_repository=${REPOSITORY}"
  echo "runtime_revision=${REVISION}"
  echo "toolchain_image=${TOOLCHAIN_IMAGE}"
  docker image inspect "${TOOLCHAIN_IMAGE}" --format 'toolchain_image_id={{.Id}} toolchain_repo_digests={{join .RepoDigests ","}}' 2>/dev/null || true
  echo "linux_preset_backends=CPU,Hexagon"
  echo "linux_preset_opencl=OFF"
  echo "full_omni_targets_built=${BUILD_OMNI}"
} > "${SOURCE}/pkg-snapdragon/BUILD_EVIDENCE.txt"

echo "QCS8550 ARM64 package: ${SOURCE}/pkg-snapdragon"
echo "Cross-build success is not QCS8550 execution proof; HTP operator placement and performance remain unverified."
