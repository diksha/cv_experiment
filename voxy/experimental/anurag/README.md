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
- To run jobs using Kube, copy the example_kube.yaml file. Modify the image, job name (ensure unique) and command/args.
- Run it using `kubectl apply -f example_kube.yaml`
- You will need to authorize kube first, for that run `gcloud container clusters get-credentials east1 --region us-east1-d`. This runs on cloud shell or GCP machine, may run on your local machine.
- Right now the cluster has 4 nodes each with 1 GPU, so max 4 trainings can run and it will scale up and down automatically.
- We can create another cluster with 4 more GPU nodes, such as west1, west2, because we have only 4 T4 GPUs available in each zone.
