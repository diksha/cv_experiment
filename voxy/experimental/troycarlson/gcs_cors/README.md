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
To enable CORS for XHR requests to non-media assets in a GCS bucket:

* https://cloud.google.com/storage/docs/configuring-cors
* https://cloud.google.com/storage/docs/cross-origin

```
gsutil cors set experimental/troycarlson/gcs_cors/config.json gs://voxel-portal
```
