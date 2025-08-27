#!/bin/bash

echo "=== EXPERIMENTO STATIC DP - VERSÃO CORRIGIDA ==="

# Verificar se MNIST está disponível
echo "🔍 Verificando dataset MNIST..."
if [ ! -d "./data/MNIST/raw" ] || [ -z "$(ls -A ./data/MNIST/raw 2>/dev/null)" ]; then
    echo "❌ MNIST não encontrado. Fazendo download..."
    python download_mnist.py
    if [ $? -ne 0 ]; then
        echo "❌ Falha no download do MNIST. Usando modo dummy..."
        export USE_DUMMY_DATA=1
    fi
fi

# Limpar ambiente
echo "🧹 Limpando ambiente..."
pkill -f "python server.py"
pkill -f "python client.py"
lsof -ti:8080 | xargs kill -9 2>/dev/null
rm -f metrics.json server.log client_*.log
sleep 3

# Iniciar servidor
echo "🚀 Iniciando servidor..."
python server.py > server.log 2>&1 &
SERVER_PID=$!
echo "Servidor PID: $SERVER_PID"

# Aguardar servidor com timeout maior
echo "⏳ Aguardando servidor (30 segundos)..."
for i in {1..30}; do
    if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null; then
        echo "✅ Servidor pronto na porta 8080"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Servidor não iniciou"
        tail -10 server.log
        exit 1
    fi
    sleep 1
done

sleep 5  # Espera adicional

# Iniciar clientes sequencialmente com verificações
echo "👥 Iniciando 4 clientes..."
for i in {0..3}; do
    echo "   Iniciando Cliente $i..."
    python client.py $i > client_$i.log 2>&1 &
    CLIENT_PID=$!
    
    # Verificar se cliente iniciou corretamente
    sleep 3
    if ! ps -p $CLIENT_PID >/dev/null; then
        echo "❌ Cliente $i falhou ao iniciar"
        tail -5 client_$i.log
    else
        echo "✅ Cliente $i rodando (PID: $CLIENT_PID)"
    fi
done

echo "=========================================="
echo "📊 Experimento em andamento..."
echo "⏰ Timeout: 600 segundos (10 minutos)"
echo "=========================================="

# Monitorar com timeout maior
TIMEOUT=600
EXPERIMENT_DONE=false

for i in $(seq 1 $TIMEOUT); do
    # Verificar se servidor ainda está rodando
    if ! ps -p $SERVER_PID >/dev/null 2>&1; then
        EXPERIMENT_DONE=true
        echo "✅ Servidor finalizado - Experimento concluído"
        break
    fi
    
    # Verificar por mensagem de conclusão
    if grep -q "SUMMARY" server.log 2>/dev/null; then
        EXPERIMENT_DONE=true
        echo "✅ Experimento concluído normalmente"
        break
    fi
    
    # Status a cada 60 segundos
    if (( i % 60 == 0 )); then
        echo "⏰ Executando... $i/$TIMEOUT segundos"
        echo "=== CLIENTES ATIVOS ==="
        ps aux | grep "python client.py" | grep -v grep | wc -l
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
echo "=== RESULTADOS ==="
if [ -f "metrics.json" ]; then
    python3 -c "
import json
try:
    with open('metrics.json') as f:
        data = json.load(f)
    print('📈 Rodadas concluídas:', len(data.get('metrics_history', {}).get('round', [])))
    print('⏱️  Tempo total: {:.1f}s'.format(data.get('total_time_seconds', 0)))
    if data.get('metrics_history', {}).get('accuracy'):
        acc = data['metrics_history']['accuracy'][-1]
        print('🎯 Acurácia final: {:.3f}'.format(acc))
    print('🔒 Total ε usado:', sum([x.get('total', 0) for x in data.get('metrics_history', {}).get('privacy_consumption', [])]))
except Exception as e:
    print('❌ Erro ao ler métricas:', e)
"
else
    echo "❌ metrics.json não encontrado"
fi

echo "📋 Logs disponíveis: server.log, client_*.log"
