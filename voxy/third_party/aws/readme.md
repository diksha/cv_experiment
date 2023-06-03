# AWS Default Config

This directory holds a default aws config that can be used by engineers at voxel to build and ship code. Copy this file to your aws configs:

```shell
cp third_party/aws/config ~/.aws/config
```

Some internal tools will use this config instead of your local config, so it is important to have at a minimum the default profile from this config installed.
