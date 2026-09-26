#!/usr/bin/env bash
# Start an SGLang ROCm container on Strix Halo (gfx1151) with the GPU exposed
# and this cookbook (plus your models dir) mounted at the same path.
#
#   bash run_docker.sh
#   MODELS_DIR=/data/models CONTAINER_NAME=sglang bash run_docker.sh
#
# IMAGE must be a gfx1151 rocm/sgl-dev build dated 20260913 or later; older
# images lack the dependencies for sglang's diffusion runtime.
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"
container_name="${CONTAINER_NAME:-sglang_strix_halo}"
image="${IMAGE:-rocm/sgl-dev:v0.5.20-rocm724-gfx1151-20260923}"
models_dir="${MODELS_DIR:-$repo_dir/models}"
mkdir -p "$models_dir"

# --group-add resolves group *names* against the container image's own
# /etc/group, not the host's. The image ships a "video" group but has no
# "render" entry, so pass render's host GID numerically instead.
render_gid="$(getent group render | cut -d: -f3)"

docker run -it --name "$container_name" \
  --ipc=host --shm-size 16G \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add "$render_gid" \
  -v "$repo_dir":"$repo_dir" \
  -v "$models_dir":"$models_dir" \
  -e MODELS_DIR="$models_dir" \
  -w "$repo_dir" \
  "$image" \
  bash
