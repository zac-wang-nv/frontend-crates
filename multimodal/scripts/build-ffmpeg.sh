#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the FFmpeg that the `video-decode` feature links: the same LGPL-only
# libraries Dynamo builds in-tree (ai-dynamo/dynamo
# container/templates/wheel_builder.Dockerfile, versions from
# container/context.yaml), installed under PREFIX with pkg-config files.
#
#   multimodal/scripts/build-ffmpeg.sh /opt/ffmpeg-dynamo
#   export PKG_CONFIG_PATH=/opt/ffmpeg-dynamo/lib/pkgconfig
#   export LD_LIBRARY_PATH=/opt/ffmpeg-dynamo/lib
#
# Needs a C toolchain, nasm, git, curl and pkg-config. The crate uses only
# libavformat (demuxing) and what it links; decoding stays on NVDEC / libvpx.
set -euo pipefail

PREFIX=${1:?usage: build-ffmpeg.sh PREFIX}
FFMPEG_VERSION=${FFMPEG_VERSION:-9.0.1}
LIBVPX_REF=${LIBVPX_REF:-v1.14.1}
JOBS=${JOBS:-$(nproc)}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)

cd "$WORK"
git clone --depth 1 --branch "$LIBVPX_REF" https://github.com/webmproject/libvpx.git
(
    cd libvpx
    ./configure --prefix="$PREFIX" --enable-shared --disable-static --enable-pic \
        --disable-examples --disable-unit-tests --disable-tools --disable-docs
    make -j"$JOBS"
    make install
)

curl -fsSL --retry 5 -o "ffmpeg-$FFMPEG_VERSION.tar.xz" \
    "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz"
tar xf "ffmpeg-$FFMPEG_VERSION.tar.xz"
cd "ffmpeg-$FFMPEG_VERSION"
# Dynamo's configure line, with PREFIX and an rpath for the local libvpx.
PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" ./configure \
    --prefix="$PREFIX" \
    --extra-ldflags="-Wl,-rpath,$PREFIX/lib" \
    --build-suffix=_dynamo \
    --disable-gpl \
    --disable-nonfree \
    --disable-doc \
    --disable-static \
    --disable-x86asm \
    --disable-network \
    --disable-bsfs \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb \
    --disable-devices \
    --disable-libdrm \
    --enable-shared \
    --enable-libvpx \
    --disable-encoders \
    --enable-encoder=libvpx_vp9 \
    --disable-decoders \
    --enable-decoder=vp8,vp9,rawvideo \
    --disable-muxers \
    --enable-muxer=mov,mp4,matroska,webm \
    --disable-demuxers \
    --enable-demuxer=mov,matroska,rawvideo \
    --disable-parsers \
    --enable-parser=vp8,vp9 \
    --disable-protocols \
    --enable-protocol=file,pipe,fd
make -j"$JOBS"
make install
# --build-suffix renames the pkg-config files; canonical names keep
# ffmpeg-sys-next's probe working (each still links the suffixed library).
for pc in "$PREFIX"/lib/pkgconfig/*_dynamo.pc; do
    ln -sf "$(basename "$pc")" "${pc%_dynamo.pc}.pc"
done
# The licence the libraries report: LGPL, never GPL or nonfree.
LD_LIBRARY_PATH="$PREFIX/lib" "$PREFIX/bin/ffmpeg" -hide_banner -version | grep -q -- --disable-gpl
echo "FFmpeg $FFMPEG_VERSION installed in $PREFIX"
