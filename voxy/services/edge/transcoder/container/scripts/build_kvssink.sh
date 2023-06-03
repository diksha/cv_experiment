#!/bin/bash

set -euo pipefail

SRCDIR="$1"
BUILDDIR="$SRCDIR"/build

mkdir -p "$BUILDDIR"
cd "$BUILDDIR"

cmake "$SRCDIR" \
	-DBUILD_GSTREAMER_PLUGIN=ON \
	-DBUILD_DEPENDENCIES=OFF \
	-DBUILD_TEST=FALSE \
	-DCODE_COVERAGE=OFF
make
install libKinesisVideoProducer.so /usr/lib
install libgstkvssink.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0
install dependency/libkvscproducer/kvscproducer-src/libcproducer.so /usr/lib
cd /
rm -rf "$SRCDIR"
