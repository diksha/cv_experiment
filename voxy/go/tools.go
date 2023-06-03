//go:build tools
// +build tools

package govoxel

import (
	_ "github.com/fullstorydev/grpcurl/cmd/grpcurl"
	_ "github.com/kyleconroy/sqlc/cmd/sqlc"
	_ "github.com/maxbrunsfeld/counterfeiter/v6"
	_ "golang.org/x/vuln/cmd/govulncheck"
	_ "google.golang.org/grpc/cmd/protoc-gen-go-grpc"
)
