<!--
Copyright 2020-2021 Voxel Labs, Inc.
All rights reserved.

This document may not be reproduced, republished, distributed, transmitted,
displayed, broadcast or otherwise exploited in any manner without the express
prior written permission of Voxel Labs, Inc. The receipt or possession of this
document does not convey any rights to reproduce, disclose, or distribute its
contents, or to manufacture, use, or sell anything that it may describe, in
whole or in part.
-->

# Tools

- This directory serves as any tools we want to bring in ecosystem such as gcloud, gsutil etc. By
  setting those under bazel it makes it easy to maintain the versions etc.
- To add a tool bring it in the WORKSPACE and then create a genrule building, outputting the third
  party tool in bazel-out and create a same named file in the tools directory and update the shell
  script accordingly to call that.
