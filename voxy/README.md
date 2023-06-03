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

# Voxel

### Links

- [Slack](https://voxel-ai.slack.com)
- [Website](https://www.voxelai.com)

## Development Instructions

Please see the [wiki page](https://github.com/voxel-ai/voxel/wiki/Development-Workflow).

## Running Spark Pipelines

Some pipelines use the [Spark](https://spark.apache.org/) framework in order to
parallelize data processing. Spark is implemented in Java, and the Python pipelines
interact with the Java worker processes through the
[PySpark API](https://spark.apache.org/docs/latest/api/python/index.html).

In order to run such pipelines locally, you will need to have the following installed:

- Java - instructions for installing OpenJDK [here](https://openjdk.org/install/).
- Spark - instructions for installing [here](https://spark.apache.org/downloads.html).
