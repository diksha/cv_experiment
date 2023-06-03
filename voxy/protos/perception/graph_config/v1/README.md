# Polygon Proto Schema

## Adding new fields to yaml / Updating the schema

If you need to add a new field to a `cha.yaml` file, you'll also need to update the proto schema. To update the Polygon proto schema, edit the `.proto` files in this directory. Which file(s) need editing depend on what fields you add to the yaml. Then, run `./tools/protogen` from the root of the Voxel repo. Commit the resulting changes to the `.pb.go` files along with the rest of the PR containing the change to the schema.

You can run `./bazel test //tests/configs/cameras:test_yaml_config` and `./bazel test //protos/perception/graph_config/v1:graphconfigpb_go_compiled_sources_test` as checks to see if things are correct.

[Here's an example PR](https://github.com/voxel-ai/voxel/pull/4156) of updating yaml+proto
Note that the names of the fields you add in `.proto` matter - in this PR, `bump_cap` was added to the proto file. `bumpcap` would not work since the field must exactly match what appears in the yaml file.

## Current limitations

Currently, the test that compares the proto schema against all `cha.yaml` files will not detect extraneous fields in the proto schema. However, the test will detect fields that are in the `cha.yaml` files that are not in the proto schema.

## Further reading

For more info on Protobuf in this repo, see [this README](https://github.com/voxel-ai/voxel/blob/main/protos/readme.md).
