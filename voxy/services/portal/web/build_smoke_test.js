/*
 * Copyright 2020-2021 Voxel Labs, Inc.
 * All rights reserved.
 *
 * This document may not be reproduced, republished, distributed, transmitted,
 * displayed, broadcast or otherwise exploited in any manner without the express
 * prior written permission of Voxel Labs, Inc. The receipt or possession of this
 * document does not convey any rights to reproduce, disclose, or distribute its
 * contents, or to manufacture, use, or sell anything that it may describe, in
 * whole or in part.
 */
const assert = require("assert");
const fs = require("fs");
const { runfiles } = require("@bazel/runfiles");

// Make sure there's a file like build/static/js/main.12345678.chunk.js
const jsDir = runfiles.resolvePackageRelative("build/static/js");
assert.ok(fs.readdirSync(jsDir).some((f) => /main\.[0-9a-f]{8}\.chunk\.js/.test(f)));
