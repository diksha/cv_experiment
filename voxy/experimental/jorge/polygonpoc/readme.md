# Polygon API POC

This folder holds a proof-of-concept version of the polygon api. The service definition is in the `polygonpoc.proto` file. If any updates are made to this file, the `./tools/proto_gen_go` script won't pick them up because it does not run updates for packages in the `experimental` directory to avoid experimental changes breaking the production build. To re-run the Go codegen, you'll need to run the following command:

`bazel run //experimental/jorge/polygonpoc:polygonpoc_go_compiled_sources.update`

This will re-run the code generation for the protobuf messages and grpc services in this project.
