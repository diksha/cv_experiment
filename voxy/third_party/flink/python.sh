#!/bin/bash

set -euo pipefail

APP_NAME=$(basename "$APP")

exec "$RUNFILES_DIR"/"$APP_NAME".venv/bin/python "$@"
