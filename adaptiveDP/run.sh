#!/bin/bash

echo "=== EXPERIMENTO ADAPTIVE DP ==="

# Configurações
PORT=8081  # 🆕 Porta diferente do static_dp
TIMEOUT=600

# Limpar ambiente
echo "🧹 Limpando ambiente..."
pkill -f "python server.py"
pkill -f "python client.py"
lsof -ti:${PORT} | xargs kill -9 2>/dev/null
rm -f metrics_adaptive.json server.log client_*.log
sleep 2

# Verificar MNIST
echo "🔍 Verificando MNIST..."
if [ ! -d "data/MNIST/raw" ] || [ -z "$(ls -A data/MNIST/raw 2>/dev/null)" ]; then
    echo "❌ MNIST não encontrado. Baixando..."
    python download_mnist.py
    if [ $? -ne 0 ]; then
        echo "❌ Falha no download do MNIST"
        exit 1
    fi
fi

# Iniciar servidor
echo "🚀 Iniciando servidor Adaptive DP na porta ${PORT}..."
python server.py > server.log 2>&1 &
SERVER_PID=$!
echo "Servidor PID: $SERVER_PID"

# Aguardar servidor
echo "⏳ Aguardando servidor (30 segundos)..."
for i in {1..30}; do
    if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null; then
        echo "✅ Servidor pronto na porta ${PORT}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Servidor não iniciou após 30 segundos"
        tail -10 server.log
        exit 1
    fi
    sleep 1
done

sleep 3

# Iniciar clientes
echo "👥 Iniciando 4 clientes Adaptive DP..."
CLIENT_PIDS=()
for i in {0..3}; do
    echo "   Cliente $i..."
    python client.py $i > client_$i.log 2>&1 &
    CLIENT_PIDS[$i]=$!
    sleep 1
done

echo "=========================================="
echo "📊 Adaptive DP em andamento..."
echo "🔍 Monitorar: tail -f server.log"
echo "🌐 Porta: ${PORT}"
echo "⏰ Timeout: ${TIMEOUT}s"
echo "=========================================="

# Função para verificar conclusão
check_completion() {
    # Servidor terminou
    if ! ps -p $SERVER_PID >/dev/null 2>&1; then
        return 0
    fi
    
    # Verificar mensagem de conclusão no log
    if grep -q "SUMMARY\|finalizado" server.log 2>/dev/null; then
        return 0
    fi
    
    return 1
}

# Esperar conclusão com timeout
EXPERIMENT_DONE=false
for i in $(seq 1 $TIMEOUT); do
    if check_completion; then
        EXPERIMENT_DONE=true
        echo "✅ Experimento concluído!"
        break
    fi
    sleep 1
done

if [ "$EXPERIMENT_DONE" = false ]; then
    echo "⏰ Timeout após $TIMEOUT segundos"
fi

# Limpeza
echo "🧹 Finalizando processos..."
pkill -f "python client.py" 2>/dev/null
pkill -f "python server.py" 2>/dev/null
sleep 2

# Resultados
echo ""
echo "=== RESULTADOS ADAPTIVE DP ==="
if [ -f "metrics_adaptive.json" ]; then
    python3 -c "
import json
try:
    with open('metrics_adaptive.json') as f:
        data = json.load(f)
    print('📈 Rodadas concluídas:', len(data.get('metrics_history', {}).get('round', [])))
    print('⏱️  Tempo total: {:.1f}s'.format(data.get('total_time_seconds', 0)))
    print('🎯 Acurácia final: {:.3f}'.format(data.get('final_accuracy', 0)))
    print('🔒 Total ε usado: {:.1f}'.format(data.get('total_epsilon_used', 0)))
    if 'epsilon_schedule' in data:
        print('📊 Schedule ε:', [round(e, 1) for e in data['epsilon_schedule']])
except Exception as e:
    print('❌ Erro ao ler métricas:', e)
"
else
    echo "❌ metrics_adaptive.json não encontrado"
fi

echo "📋 Logs disponíveis: server.log, client_*.log"
