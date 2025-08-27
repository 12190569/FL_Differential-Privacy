import flwr as fl
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import time
from datetime import datetime

class OptimizedStaticDPServer(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,          # TODOS os clientes participam
            fraction_evaluate=1.0,     # TODOS avaliam
            min_fit_clients=4,         # Esperar por 4 clientes
            min_evaluate_clients=4,    # Esperar por 4 clientes para avaliação
            min_available_clients=4,   # Mínimo de 4 clientes disponíveis
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.eval_config,
        )
        
        # Estruturas para coleta de métricas
        self.metrics_history = {
            'round': [], 
            'loss': [], 
            'accuracy': [], 
            'privacy_consumption': [], 
            'fairness': [], 
            'efficiency': []
        }
        self.client_metrics = {}
        self.start_time = time.time()
        print("🚀 Servidor Static DP SUPER Otimizado inicializado")
        print("🎯 Novos Parâmetros: ε=8.0, clip_norm=2.0, LR=0.001, epochs=1")

    def fit_config(self, server_round: int):
        """Configurações de treinamento OTIMIZADAS"""
        return {
            "epsilon": 8.0,           # ✅ AUMENTADO: 4x menos ruído
            "delta": 1e-5,
            "clip_norm": 2.0,         # ✅ AUMENTADO: 4x menos clipping
            "round": server_round,
            "learning_rate": 0.001,   # ✅ REDUZIDO: 10x mais suave
            "epochs": 1,              # ✅ REDUZIDO: 1 época por rodada
        }

    def eval_config(self, server_round: int):
        """Configurações de avaliação"""
        return {"round": server_round}

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"🔧 RODADA {server_round} - Configurando treinamento")
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Configurar parâmetros de privacidade para cada cliente
        for i, (client, fit_ins) in enumerate(client_instructions):
            config = fit_ins.config
            config["epsilon"] = 8.0           # Menos ruído
            config["delta"] = 1e-5
            config["clip_norm"] = 2.0         # Menos clipping
            config["round"] = server_round
            config["learning_rate"] = 0.001   # LR menor
            config["epochs"] = 1              # 1 época
            
        print(f"   ✅ {len(client_instructions)} clientes configurados com ε=8.0")
        return client_instructions

    def aggregate_fit(self, server_round, results, failures):
        print(f"🔧 RODADA {server_round} - Agregando {len(results)} resultados")
        
        if not results:
            print("❌ Nenhum resultado recebido!")
            return None
            
        round_metrics = {
            'loss': [],
            'epsilon_used': [],
            'client_types': [],
            'accuracies': []
        }
        
        for client, fit_res in results:
            metrics = fit_res.metrics
            loss = metrics.get("loss", 0)
            epsilon_used = metrics.get("epsilon_used", 0)
            
            round_metrics['loss'].append(loss)
            round_metrics['epsilon_used'].append(epsilon_used)
            round_metrics['client_types'].append(metrics.get("client_type", "unknown"))
            
            print(f"   📊 Cliente: ε={epsilon_used:.1f}, Loss={loss:.4f}")
        
        # Calcular métricas da rodada
        self._calculate_round_metrics(server_round, round_metrics)
        
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        print(f"🔧 RODADA {server_round} - Avaliando {len(results)} clientes")
        
        if results:
            accuracies = []
            for client, evaluate_res in results:
                accuracy = evaluate_res.metrics.get("accuracy", 0)
                accuracies.append(accuracy)
            
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                self.metrics_history['accuracy'].append(avg_accuracy)
                
                print(f"📈 R{server_round} - Acurácia média: {avg_accuracy:.3f}")
                
                # Verificar se atingiu 90%
                if avg_accuracy >= 0.9:
                    efficiency_data = {
                        'target_round': server_round,
                        'accuracy': avg_accuracy,
                        'time_elapsed': time.time() - self.start_time
                    }
                    if 'efficiency' not in self.metrics_history:
                        self.metrics_history['efficiency'] = []
                    self.metrics_history['efficiency'].append(efficiency_data)
                    print(f"🎯 META ATINGIDA! 90% na rodada {server_round}")
        
        return super().aggregate_evaluate(server_round, results, failures)

    def _calculate_round_metrics(self, server_round: int, round_metrics: Dict):
        """Calcula todas as métricas para a rodada"""
        if not round_metrics['loss']:
            print(f"❌ Rodada {server_round} - Sem métricas para calcular")
            return
        
        # 1. Model Utility (Loss)
        avg_loss = np.mean(round_metrics['loss'])
        self.metrics_history['round'].append(server_round)
        self.metrics_history['loss'].append(avg_loss)
        
        # 2. Privacy Consumption
        total_epsilon = sum(round_metrics['epsilon_used'])
        epsilon_stats = {
            'total': total_epsilon,
            'mean': np.mean(round_metrics['epsilon_used']),
            'std': np.std(round_metrics['epsilon_used']),
            'max': max(round_metrics['epsilon_used']),
            'min': min(round_metrics['epsilon_used'])
        }
        self.metrics_history['privacy_consumption'].append(epsilon_stats)
        
        # 3. Fairness
        if len(round_metrics['epsilon_used']) > 1:
            fairness = 1 - (np.std(round_metrics['epsilon_used']) / np.mean(round_metrics['epsilon_used']))
            self.metrics_history['fairness'].append(fairness)
        else:
            self.metrics_history['fairness'].append(1.0)
        
        print(f"📊 R{server_round} - Loss: {avg_loss:.4f}, ε Total: {total_epsilon:.1f}")

    def save_metrics(self, filename: str = "metrics.json"):
        """Salva todas as métricas em JSON"""
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': time.time() - self.start_time,
            'total_rounds': len(self.metrics_history['round']),
            'final_accuracy': self.metrics_history['accuracy'][-1] if self.metrics_history['accuracy'] else 0,
            'total_epsilon_used': sum([x.get('total', 0) for x in self.metrics_history['privacy_consumption']]),
            'final_loss': self.metrics_history['loss'][-1] if self.metrics_history['loss'] else 0,
            'metrics_history': self.metrics_history,
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"💾 Métricas salvas em {filename}")

def main():
    strategy = OptimizedStaticDPServer()
    
    try:
        print("⏳ Iniciando servidor Flower...")
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=15),  # 15 rodadas
            strategy=strategy,
        )
    except Exception as e:
        print(f"❌ Erro no servidor: {e}")
    finally:
        strategy.save_metrics()
        print("✅ Servidor finalizado")

if __name__ == "__main__":
    main()
