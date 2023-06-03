#!/bin/bash

# Temporary until skaffold integration after initial server setup
docker run -p 4000:4000 --network host --mount "type=bind,source=./services/router,target=/dist/schema" --rm ghcr.io/apollographql/router:v0.10.0 -c schema/router.yaml -s schema/supergraph.graphql
