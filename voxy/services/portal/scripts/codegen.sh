#!/bin/bash

set -euo pipefail

# ============================================================================
# Generate Django migrations
# ============================================================================
./core/portal/scripts/makemigrations.sh

# ============================================================================
# Generate GraphQL schema + client types
# ============================================================================
./services/portal/scripts/graphql_sync.sh

# ============================================================================
# Fun success message with ASCII bunny
# ============================================================================
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)
WHITE=$(tput setaf 7)
RESET=$(tput sgr0)

cat <<EOF
${YELLOW}  ____________
${YELLOW} |            |    ${WHITE}• ${GREEN}Codegen finished successfully!
${YELLOW} |   NOTICE   |    ${WHITE}• If you have a backend devserver running you must restart
${YELLOW} |____________|    ${WHITE}  it to apply new migrations.
${WHITE}(\__/) ${YELLOW}||          ${WHITE}• If you have a frontend devserver running you may have to 
${WHITE}(${BLUE}•${WHITE}ㅅ${BLUE}•${WHITE}) ${YELLOW}||            ${WHITE}restart it to pick up newly generated files.
${WHITE}/    づ
${GREEN}YYYYYYYYYYYYYYY${RESET}
EOF
