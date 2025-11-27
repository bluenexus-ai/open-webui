#!/bin/bash
# Load BlueNexus environment variables from parent .env if it exists
if [ -f "../.env" ]; then
    export ENABLE_BLUENEXUS=$(grep "^ENABLE_BLUENEXUS=" ../.env | head -1 | cut -d'=' -f2)
    export ENABLE_BLUENEXUS_SYNC=$(grep "^ENABLE_BLUENEXUS_SYNC=" ../.env | cut -d'=' -f2)
fi

export CORS_ALLOW_ORIGIN="http://localhost:5173;http://localhost:8080;http://localhost:9090"
PORT="${PORT:-9090}"
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
