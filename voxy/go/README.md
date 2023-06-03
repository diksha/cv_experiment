# Go Development Workflow

This readme covers various topics surrounding how to set up and work in Go in this repository.

## Initial Setup

It is recommended to install a system go as there are a number of tools that will work more effectively with a system go installation. The main Go VSCode extension though does use the SDK built by bazel, and you'll want to make sure that SDK has been downloaded/installed for go development and the go package driver has been built. Bazel should do this automatically but the process will be smoother if you run the following command:

`./bazel build @io_bazel_rules_go//go/tools/gopackagesdriver`

### Install Go

**Ubuntu**
The simplest way is to install via snap with `sudo snap install go --classic`
There are alternate installation instructions [here](https://go.dev/doc/install)

**Other Linux**
It is recommended to follow the steps [here](https://go.dev/doc/install)

**Mac**
It is recommended to just follow the steps [here](https://go.dev/doc/install)

### Install VSCode Plugin

VSCode should prompt you when opening a go file to install the Go Extension, but if you have not received this prompt or want to install it directly you can find it in the extensions manager as `Go`. Once the plugin is installed, it will prompt you about installing various tools (`gofmt`, `goimports`, `gocode`) which you should let it install. For now these will be installed to the system go, but in the future we will migrate to bazel installed versions of all of these tools and will provide vscode configurations to make using these simple.

The most important tool to ensure is installed is `gopls`, if you find that go editor features are not working correctly it is likely that the language server is not working correctly. To debug this you can try checking for errors by running this at the project root:

`echo {} | ./tools/gopackagesdriver.sh file=cmd/go/examples/hello/hello.go`

If you see a bunch of json, things are working correctly. If you get bazel errors, then go isn't able to actually pull language information out and you'll need to fix the bazel errors for the editor to work correctly.

## Writing new files/modules

New modules/files should mostly be in an appropriately named package under `go/voxel` although packages can exist other places, such as the experimental folder. It is not recommended to mix language code so outside of experiments it is strongly recommended to keep all go code in `go/voxel`

Once your new code/module has been written, make sure to run gazelle with `./build run //:gazelle` to generate/update Bazel build files and then check them in.

## Adding new Go modules to go.mod

Adding dependencies is slightly more complex than with the standard go tooling. The following steps should get you there:

1. Use the standard go tool to `go get` the package you want, like `go get github.com/twitchtv/twirp` if you were trying to add the twirp package.
2. Run `./bazel run //:gazelle-update-repos` to update the `go_deps.bzl` file with the new changes in `go.mod`
3. Add an import statement to your Go source file with this new import like `import "github.com/twitchtv/twirp"`
4. Run `./bazel run //:gazelle` to generate the correct dependencies list in the `BUILD.bazel` file for your Go source
5. Run `./bazel build` with your target to cause bazel to download and set up the new dependency. The build may fail but the dependency download steps will get the packages set up so they work with code completion.

From here you should be able to use the new dependency freely. To add this dependency to other packages, you'll need to perform steps 3-5 for each package that would like to import the new module.

## Writing and running tests

Tests in go are written in the same directory as the package they are testing. See `go/voxel/examples/hello`to see an example test.

## Quirks

If you have any trouble getting things to work correctly, consider the following notes:

- When adding new source files, the go editor features won't work until they are added to a build file with `./build run //:gazelle`
- When adding new imports, the go editor features won't work until those imports are added to `go_deps.bzl` with `./build run //:gazelle-update-repos`
- Most issues with editor features not working like autoformatting, autocompletion, autoimports will fail because the package server isn't starting. Try `echo {} | ./tools/gopackagesdriver.sh file=cmd/go/examples/hello/hello.go` to debug.
- There is currently a bug that causes VSCode to fail to load updates to bazel build files, tracked in [bazelbuild/rules.go#3014](https://github.com/bazelbuild/rules_go/issues/3014)
- The editor window seems to get out of sync semi-regularly, the easy way to resolve this is to just run the reload window command from the command palatte (cmd-shift-p or ctrl-shift-p)
