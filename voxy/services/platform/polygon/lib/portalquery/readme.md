# portalquery

This package provides pre-defined queries against the Voxel portal db. To update this set of queries, do not edit the go files. Instead edit `queries.sql` to add your query, and then run `go generate .` in this folder (or `go generate ./experimental/jorge/sqlc-service-skeleton/...` from the repo root)

Be aware that changing already-existing queries may change their signatures in the generated code, which may break existing code.
