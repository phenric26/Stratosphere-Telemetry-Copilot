#!/usr/bin/env bash
set -e

# Exporta bibliotecas essenciais para sistemas Linux/Nix
if [ -d "/nix/store/7a2cjhcgzyym5zaz6m4fksw47qvm150w-gcc-16.2.0-lib/lib" ]; then
    export LD_LIBRARY_PATH=/nix/store/7a2cjhcgzyym5zaz6m4fksw47qvm150w-gcc-16.2.0-lib/lib:${LD_LIBRARY_PATH}
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/streamlit" ]; then
    echo "Iniciando Stratosphere Telemetry Copilot no ambiente virtual..."
    .venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
else
    echo "Iniciando com streamlit global..."
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
fi
