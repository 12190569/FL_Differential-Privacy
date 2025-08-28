#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8084}"
ROUNDS="${ROUNDS:-15}"
NUM_CLIENTS="${NUM_CLIENTS:-4}"
PY="${PY:-python}"
LOGDIR="${LOGDIR:-.}"

mkdir -p "$LOGDIR"
export PYTHONUNBUFFERED=1
export GRPC_VERBOSITY=ERROR

echo "Using: PY=$PY | PORT=$PORT | ROUNDS=$ROUNDS | NUM_CLIENTS=$NUM_CLIENTS"
echo "Logs em: $LOGDIR"

# ---- Server ----
$PY -u server.py --port "$PORT" --rounds "$ROUNDS" --logdir "$LOGDIR" \
  2>&1 | tee "$LOGDIR/server.log" &

SERVER_PID=$!
sleep 3

# ---- Clients ----
PIDS=()
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  echo "Iniciando cliente $i ..."
  $PY -u client.py --server "127.0.0.1:${PORT}" --cid "$i" --num_clients "$NUM_CLIENTS" \
    2>&1 | tee "$LOGDIR/client_${i}.log" &
  PIDS+=($!)
  sleep 1
done

# ---- Espera servidor terminar e encerra clientes ----
wait $SERVER_PID || true
for pid in "${PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    wait "$pid" || true
  fi
done

