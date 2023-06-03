#!/bin/bash

set -euo pipefail

WORKDIR=$(realpath "$1")

cd "$WORKDIR"
./configure \
	--prefix=/usr \
	--disable-static \
	--enable-shared \
	--enable-gpl \
	--enable-gnutls \
	--enable-libx264 \
	--enable-libx265 \
	--enable-libmfx \
	--enable-opencl \
	--enable-vaapi \
	--enable-xlib

make -j"$(nproc)"
make install

cd /
rm -rf "$WORKDIR"
