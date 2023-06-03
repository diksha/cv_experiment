# Sematic

## Directory Overview

This directory contains the main [Sematic](https://sematic.dev) pipelines.
The `shared` subdirectory contains utilities and common Sematic function/type
definitions. The other subdirectories contain categories of pipelines, for
example "perception" for perception pipelines.

## Sematic Setup

You will need to set up your local Sematic configuration to execute pipelines
for local development or for launching Sematic jobs to the cloud. To set up:

1. `./tools/sematic settings set SEMATIC_API_ADDRESS http://sematic-internal.voxelplatform.com`
2. `./tools/sematic settings set AWS_S3_BUCKET 203670452561-sematic-ci-cd`
3. Visit [Voxel's Sematic deployment](https://sematic.voxelplatform.com/) and
   login with your Google account
4. In the lower left of the UI, locate your avatar and mouse over to get your API key
5. `./tools/sematic settings set SEMATIC_API_KEY <your API key>`
