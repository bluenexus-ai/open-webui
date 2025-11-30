export CORS_ALLOW_ORIGIN="http://localhost:5173;http://localhost:8080;http://localhost:9090"
PORT="${PORT:-9090}"
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
