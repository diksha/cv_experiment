package govoxel

// this fixes a bug with xerrors as a dependency
// for some reason gazelle breaks if we don't
// force an xerrors import, and putting it here causes
//
// go mod tidy
// ./bazel run //:gazelle-update-repos
//
// to both leave the import in place
// trunk-ignore(golangci-lint/revive)
import _ "golang.org/x/xerrors"
