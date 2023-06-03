# Voxel Tools

This directory holds tools that are specific to Voxel's environment typically with some extra setup

## Edge ssh/scp

The `edge-ssh` and `edge-scp` tools here provide scp/ssh which will accept an edge uuid as a hostname. An example of using these would be:

Run uptime on our dev edge:

`tools/voxel/edge-ssh d3ecbf7a-afb5-44b0-8464-c2951282d595 uptime`

Copy a file to our dev edge:

`tools/voxel/edge-scp testfile.txt d3ecbf7a-afb5-44b0-8464-c2951282d595:/tmp/testfile.txt`
