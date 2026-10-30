# syntax=docker/dockerfile:1
# ZenTrainer in a container, for a cloud GPU (RunPod, Vast.ai, ...) or any Docker host with an NVIDIA
# GPU: the managed ComfyUI at the comfy.lock pin, the built web app, and a password in front of it.
# Build from the repository root:
#
#   docker build -f docker/Dockerfile -t zentrainer .
#
# How to run it and what it keeps where: docker/README.md.

# a torch build from comfy.lock. cu130 needs an NVIDIA driver for CUDA 13.0 or newer, and covers
# current GPUs including Blackwell (RTX 5090, B200).
ARG TORCH=cu130

# ---- the web app, built from this checkout ------------------------------------------------------
FROM node:24-slim AS web
RUN npm install --global pnpm@11.1.1
WORKDIR /src
# the app imports engine/catalog/families.json from outside web/, and reads its version from the engine
COPY engine/catalog engine/catalog
COPY engine/zentrainer/__init__.py engine/zentrainer/__init__.py
COPY web web
RUN cd web && pnpm install --frozen-lockfile && pnpm build

FROM caddy:2.10.2 AS caddy
FROM ghcr.io/astral-sh/uv:0.11.28 AS uv

# ---- the image ----------------------------------------------------------------------------------
FROM ubuntu:24.04
ARG TORCH
ENV DEBIAN_FRONTEND=noninteractive
# gcc + libc6-dev: Triton compiles its CUDA helper the first time torch runs one of its kernels
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates gcc git libc6-dev openssh-server procps tini \
 && rm -rf /var/lib/apt/lists/* \
 && userdel --remove ubuntu \
 && useradd --create-home --uid 1000 --shell /bin/bash zen \
 && mkdir -p /opt/zentrainer /opt/uv-python \
 && chown zen:zen /opt/zentrainer /opt/uv-python
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=caddy /usr/bin/caddy /usr/local/bin/caddy

# Python goes where the unprivileged user can read it, and uv leaves no download cache in the image
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_NO_CACHE=1 UV_LINK_MODE=copy ZT_TORCH=${TORCH}

# Everything under /opt/zentrainer belongs to zen, who runs ComfyUI. The heavy layer (ComfyUI and
# torch) depends only on comfy.lock and zt, so an engine or web change rebuilds in seconds.
USER zen
WORKDIR /opt/zentrainer
COPY --chown=zen:zen comfy.lock ./
COPY --chown=zen:zen cli cli
RUN uv sync --frozen --no-dev --project cli --python "$(sed -n 's/.*"python": *"\([^"]*\)".*/\1/p' comfy.lock)" \
 && cli/.venv/bin/zt setup --torch "$TORCH" \
 && cli/.venv/bin/zt extras optimizers
COPY --chown=zen:zen . .
COPY --chown=zen:zen --from=web /src/web/dist web/dist

USER root
COPY docker/Caddyfile /etc/zentrainer/Caddyfile
COPY docker/entrypoint.sh /usr/local/bin/zentrainer-entrypoint
RUN printf '#!/bin/sh\nexec /opt/zentrainer/cli/.venv/bin/zt "$@"\n' > /usr/local/bin/zt \
 && chmod 755 /usr/local/bin/zt /usr/local/bin/zentrainer-entrypoint

# 8188: the app, behind the password. 22: SSH, key only, when PUBLIC_KEY is set.
EXPOSE 8188 22
# -s: some hosts (RunPod) run their own init as PID 1, so tini registers to reap zombies anyway
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "/usr/local/bin/zentrainer-entrypoint"]
