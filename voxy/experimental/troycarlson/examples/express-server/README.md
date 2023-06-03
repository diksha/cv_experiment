# Example Express (NodeJS) Server

## Setup

Install npm dependencies into the source tree so the editor and other tools
can find type definitions.

See the [rules_js docs](https://docs.aspect.build/rules/aspect_rules_js/docs/faq#making-the-editor-happy) for more info.

## Development

To run the Express server:

```shell
./bazel run //experimental/troycarlson/examples/express-server:index
```

Which listens on port 8000 and can be viewed in a browser at `http://localhost:8000`.
