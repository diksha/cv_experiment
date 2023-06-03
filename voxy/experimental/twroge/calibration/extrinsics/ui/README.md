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
# Usage

Just run
```
./bazel run //experimental/twroge/calibration/extrinsics/ui:main
```

Dump the image in `./static/` then navigate to `localhost:8080/image/<image_name>/<deep_calib_focal_length>`. 
Make sure to have port forwarding setup (more information provided below).

For example, try: 
```
localhost:8080/image/warehouse.jpg/299
```

This will load warehouse.jpg with a focal length of 299 into the UI.
*Note:* the focal length should be given in pixel coordinates as it is given for deepcalib. The calibration
unit is in pixels and is scaled by 299 

You can also pass an image and focal length in using the commandline:
```
usage: ./bazel run //experimental/twroge/calibration/extrinsics/ui:main -- [-h] [--focal_length FOCAL_LENGTH] [--image_path IMAGE_PATH]

Generate calibration from an image and focal length

optional arguments:
  -h, --help            show this help message and exit
  --focal_length FOCAL_LENGTH
                        the focal length in image coordinates based on 299x299
                        sized image
  --image_path IMAGE_PATH
                        the image path
```

If you pass in commandline arguments, then you can just navigate to `localhost:8080`.


## Port Forwarding

For more information about port forwarding please see:

https://github.com/voxel-ai/voxel/wiki/Engineering-Productivity#starting-an-instance

The command to port forward is to pass the option `-L 8080:localhost:8080`. If you are using ssh, then you
can port forward using the following command:
```bash
ssh -i <key> <username>@<ip address> -L 8080:localhost:8080
```

For google cloud client, you can use this:
```
gcloud compute ssh <name>--workstation -- -L 8080:localhost:8080
```
