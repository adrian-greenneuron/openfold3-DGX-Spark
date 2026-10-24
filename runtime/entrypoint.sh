#!/usr/bin/env bash
set -euo pipefail
setup_openfold --non-interactive
(cd /root/.openfold3 && sha256sum --check /deployment/checkpoint.sha256)
exec run_openfold "$@"
