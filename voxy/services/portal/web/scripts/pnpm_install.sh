#!/bin/bash

set -eou pipefail

# Installs portal frontend packages per services/portal/web/package.json

tools/pnpm --dir services/portal/web install "$@"
