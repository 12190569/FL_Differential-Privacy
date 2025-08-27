#!/bin/bash

echo "=== EXPERIMENTO AGENTIC DP ==="

# Configurações
PORT=8084  # 🆕 Porta diferente
TIMEOUT=600

# Limpar ambiente
echo "🧹 Limpando ambiente..."
pkill -f "python server.py"
pkill -f "python client.py"
lsof -ti:${PORT} | xargs kill -9 2>/dev/null
rm -f metrics_agentic.json server.log client_*.log
sleep 2

# Verificar MNIST
echo "🔍 Verificando MNIST..."
if [ ! -d "data/MNIST/raw" ] || [ -z "$(ls -A data/MNIST/raw 2>/dev/null)" ]; then
    echo "❌ MNIST não encontrado. Baixando..."
    python download_mnist.py
fi

# Iniciar servidor
echo "🚀 Iniciando servidor Agentic DP na porta ${PORT}..."
python server.py > server.log 2>&1 &
SERVER_PID=$!
echo "Servidor PID: $SERVER_PID"

# Aguardar servidor
echo "⏳ Aguardando servidor..."
for i in {1..30}; do
    if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null; then
        echo "✅ Servidor pronto na porta ${PORT}"
        break
    fi
    sleep 1
done

sleep 3

# Iniciar clientes
echo "👥 Iniciando 4 clientes Agentic DP..."
for i in {0..3}; do
    echo "   Cliente $i (alocação personalizada de ε)..."
    python client.py $i > client_$i.log 2>&1 &
    sleep 1
done

echo "=========================================="
echo "📊 Agentic DP em andamento..."
echo "🔍 Monitorar: tail -f server.log"
echo "🌐 Porta: ${PORT}"
echo "⏰ Timeout: ${TIMEOUT}s"
echo "=========================================="

# Wait for completion
wait $SERVER_PID 2>/dev/null

echo "✅ Experimento Agentic DP concluído!"
echo "📈 Métricas: metrics_agentic.json"

# Resultados
if [ -f "metrics_agentic.json" ]; then
    python3 -c "
import json
with open('metrics_agentic.json') as f:
    data = json.load(f)
print('📊 RESULTADOS AGENTIC DP:')
print(f'🎯 Acurácia final: {data.get(\"final_accuracy\", 0):.3f}')
print(f'🔒 Total ε usado: {data.get(\"total_epsilon_used\", 0):.1f}')
print(f'⚖️  Fairness médio: {data.get(\"avg_fairness\", 0):.3f}')
print(f'📈 Eficiência: {data.get(\"final_accuracy\", 0)/data.get(\"total_epsilon_used\", 1):.3f}% por ε')
"
fi
