# Why is this vendored and not installed via pypi

For some reason, when this package is installed via pypi and bazel as an
entry in `third_party/pypi/requirements.in` the package does not install correctly. The module
is for some reason empty even though a statement like `import CloseableQueue` does not error out.
If anyone can figure out how to fix this, please do and change this back to a proper requirement.
