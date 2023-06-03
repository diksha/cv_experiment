#!/bin/bash

set -euo pipefail

SRCDIR="$1"

cd /tmp
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make install

cd "$SRCDIR"
./configure \
	--prefix=/usr \
	--extra-cflags=-I/usr/local/cuda/include \
	--extra-ldflags=-L/usr/local/cuda/lib64 \
	--disable-static \
	--enable-shared \
	--enable-gpl \
	--enable-gnutls \
	--enable-libx264 \
	--enable-libx265 \
	--enable-cuda-nvcc \
	--enable-libnpp \
	--enable-nonfree

make -j"$(nproc)"
make install

cd /
rm -rf "$SRCDIR"
rm -rf /tmp/nv-codec-headers
