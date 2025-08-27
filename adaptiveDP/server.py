import flwr as fl
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import time
import math
from datetime import datetime

class AdaptiveDPServer(fl.server.strategy.FedAvg):
    def __init__(self, total_rounds=15):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=4,
            min_evaluate_clients=4,
            min_available_clients=4,
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.eval_config,
        )
        
        self.total_rounds = total_rounds
        self.metrics_history = {
            'round': [], 'loss': [], 'accuracy': [], 
            'privacy_consumption': [], 'fairness': [], 'efficiency': [],
            'epsilon_schedule': []
        }
        self.start_time = time.time()
        print("🚀 Servidor Adaptive DP inicializado")
        print("🎯 Schedule: ε adaptativo baseado no progresso")

    def adaptive_epsilon_schedule(self, server_round, current_accuracy):
        """Schedule adaptativo de ε: mais ruído no início, menos no final"""
        progress = server_round / self.total_rounds
        
        # Estratégia: Decaimento de Cosseno
        base_epsilon = 10.0
        epsilon = base_epsilon * (0.3 + 0.7 * (1 + math.cos(math.pi * progress)) / 2)
        
        # Ajustes baseados na acurácia
        if current_accuracy < 0.3:
            epsilon = max(epsilon, 6.0)
        elif current_accuracy > 0.7:
            epsilon = min(epsilon, 12.0)
            
        # Limites de segurança
        epsilon = max(min(epsilon, 15.0), 3.0)
        
        print(f"   📊 Schedule Adaptive: R{server_round}, Acc={current_accuracy:.3f} → ε={epsilon:.1f}")
        return epsilon

    def fit_config(self, server_round: int):
        current_accuracy = self.metrics_history['accuracy'][-1] if self.metrics_history['accuracy'] else 0.1
        epsilon = self.adaptive_epsilon_schedule(server_round, current_accuracy)
        
        return {
            "epsilon": epsilon,
            "delta": 1e-5,
            "clip_norm": 2.0,
            "round": server_round,
            "learning_rate": 0.001,
            "epochs": 1,
            "schedule_type": "adaptive"
        }

    def eval_config(self, server_round: int):
        return {"round": server_round}

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"🔧 RODADA {server_round} - Configurando Adaptive DP")
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        current_accuracy = self.metrics_history['accuracy'][-1] if self.metrics_history['accuracy'] else 0.1
        epsilon = self.adaptive_epsilon_schedule(server_round, current_accuracy)
        
        for i, (client, fit_ins) in enumerate(client_instructions):
            fit_ins.config["epsilon"] = epsilon
            fit_ins.config["delta"] = 1e-5
            fit_ins.config["clip_norm"] = 2.0
            fit_ins.config["round"] = server_round
            fit_ins.config["learning_rate"] = 0.001
            fit_ins.config["epochs"] = 1
            
        self.metrics_history['epsilon_schedule'].append(epsilon)
        print(f"   ✅ {len(client_instructions)} clientes com ε={epsilon:.1f}")
        return client_instructions

    def aggregate_fit(self, server_round, results, failures):
        print(f"🔧 RODADA {server_round} - Agregando {len(results)} resultados")
        
        if not results:
            return None
            
        round_metrics = {'loss': [], 'epsilon_used': []}
        
        for client, fit_res in results:
            metrics = fit_res.metrics
            round_metrics['loss'].append(metrics.get("loss", 0))
            round_metrics['epsilon_used'].append(metrics.get("epsilon_used", 0))
            
        self._calculate_round_metrics(server_round, round_metrics)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        if results:
            accuracies = [evaluate_res.metrics.get("accuracy", 0) for _, evaluate_res in results]
            avg_accuracy = sum(accuracies) / len(accuracies)
            self.metrics_history['accuracy'].append(avg_accuracy)
            
            print(f"📈 R{server_round} - Acurácia: {avg_accuracy:.3f}")
        
        return super().aggregate_evaluate(server_round, results, failures)

    def _calculate_round_metrics(self, server_round: int, round_metrics: Dict):
        if round_metrics['loss']:
            avg_loss = np.mean(round_metrics['loss'])
            total_epsilon = sum(round_metrics['epsilon_used'])
            
            self.metrics_history['round'].append(server_round)
            self.metrics_history['loss'].append(avg_loss)
            self.metrics_history['privacy_consumption'].append({
                'total': total_epsilon,
                'mean': np.mean(round_metrics['epsilon_used']),
            })
            
            print(f"📊 R{server_round} - Loss: {avg_loss:.4f}, ε: {total_epsilon:.1f}")

    def save_metrics(self, filename: str = "metrics_adaptive.json"):
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': time.time() - self.start_time,
            'total_rounds': len(self.metrics_history['round']),
            'final_accuracy': self.metrics_history['accuracy'][-1] if self.metrics_history['accuracy'] else 0,
            'total_epsilon_used': sum([x.get('total', 0) for x in self.metrics_history['privacy_consumption']]),
            'epsilon_schedule': self.metrics_history['epsilon_schedule'],
            'metrics_history': self.metrics_history,
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"💾 Métricas adaptativas salvas em {filename}")

def main():
    strategy = AdaptiveDPServer(total_rounds=15)
    
    try:
        print("🌐 Iniciando servidor na porta 8081...")
        fl.server.start_server(
            server_address="0.0.0.0:8081",  # ✅ PORTA CORRIGIDA
            config=fl.server.ServerConfig(num_rounds=15),
            strategy=strategy,
        )
    except Exception as e:
        print(f"❌ Erro no servidor: {e}")
    finally:
        strategy.save_metrics()
        print("✅ Servidor finalizado")

if __name__ == "__main__":
    main()
