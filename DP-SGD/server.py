import flwr as fl
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import time
from datetime import datetime
import re

class DPSGDServer(fl.server.strategy.FedAvg):
    def __init__(self, total_rounds=15, target_epsilon=10.0, target_delta=1e-5):
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
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.metrics_history = {
            'round': [], 'loss': [], 'accuracy': [], 
            'privacy_consumption': [], 'fairness': [], 'efficiency': [],
            'client_epsilons': []
        }
        self.start_time = time.time()
        self.client_counter = 0
        print("🚀 Servidor DP-SGD (Standard) inicializado")
        print("🎯 Estratégia: DP-SGD com alocação uniforme de ε")

    def calculate_uniform_epsilon(self, server_round):
        """Calcula ε uniforme para todos os clientes usando composição DP"""
        # ε total distribuído igualmente entre rodadas
        epsilon_per_round = self.target_epsilon / self.total_rounds
        return epsilon_per_round

    def calculate_noise_multiplier(self, epsilon):
        """Calcula multiplicador de ruído baseado no ε desejado"""
        # Aproximação simplificada para o experimento
        if epsilon > 8.0:
            return 0.3
        elif epsilon > 5.0:
            return 0.5
        elif epsilon > 3.0:
            return 0.8
        else:
            return 1.2

    def fit_config(self, server_round: int):
        """Configurações DP-SGD para todas as chamadas fit"""
        uniform_epsilon = self.calculate_uniform_epsilon(server_round)
        noise_multiplier = self.calculate_noise_multiplier(uniform_epsilon)
        
        return {
            "round": server_round,
            "learning_rate": 0.001,
            "epochs": 1,
            "epsilon": uniform_epsilon,
            "delta": self.target_delta,
            "clip_norm": 1.5,
            "noise_multiplier": noise_multiplier,
            "dp_mechanism": "dp-sgd"
        }

    def eval_config(self, server_round: int):
        return {"round": server_round}

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"\n🔧 RODADA {server_round} - DP-SGD Uniforme")
        uniform_epsilon = self.calculate_uniform_epsilon(server_round)
        noise_multiplier = self.calculate_noise_multiplier(uniform_epsilon)
        
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        client_epsilons = {}
        
        for client, fit_ins in client_instructions:
            fit_ins.config["epsilon"] = uniform_epsilon
            fit_ins.config["noise_multiplier"] = noise_multiplier
            fit_ins.config["dp_mechanism"] = "dp-sgd"
            
            client_id = self._get_client_id(client)
            fit_ins.config["client_id"] = client_id
            
            client_epsilons[client_id] = uniform_epsilon
            
            print(f"   👤 Cliente {client_id}: ε={uniform_epsilon:.2f}, Noise={noise_multiplier:.2f} - DP-SGD Uniforme")
        
        self.metrics_history['client_epsilons'].append(client_epsilons)
        return client_instructions

    def _get_client_id(self, client):
        """Extrai ID do cliente corretamente"""
        try:
            # Método 1: Tentar extrair do CID
            if hasattr(client, 'cid'):
                cid = client.cid
                numbers = re.findall(r'\d+', str(cid))
                if numbers:
                    return int(numbers[0]) % 4
            
            # Método 2: Extrair da string de representação
            client_str = str(client)
            numbers = re.findall(r'\d+', client_str)
            if numbers:
                return int(numbers[-1]) % 4
                
        except Exception as e:
            print(f"⚠️  Erro ao extrair ID: {e}")
        
        # Fallback: distribuição round-robin
        client_id = self.client_counter % 4
        self.client_counter += 1
        return client_id

    def aggregate_fit(self, server_round, results, failures):
        print(f"🔧 RODADA {server_round} - Agregando {len(results)} resultados DP-SGD")
        
        if not results:
            return None
            
        round_metrics = {'loss': [], 'epsilon_used': [], 'client_ids': [], 'noise_multipliers': []}
        
        for client, fit_res in results:
            metrics = fit_res.metrics
            client_id = metrics.get("client_id", -1)
            epsilon_used = metrics.get("epsilon_used", 0)
            loss_val = metrics.get("loss", 0)
            noise_used = metrics.get("noise_multiplier", 0)
            
            round_metrics['loss'].append(loss_val)
            round_metrics['epsilon_used'].append(epsilon_used)
            round_metrics['client_ids'].append(client_id)
            round_metrics['noise_multipliers'].append(noise_used)
            
            print(f"   📊 Cliente {client_id}: ε={epsilon_used:.2f}, Noise={noise_used:.2f}, Loss={loss_val:.4f}")
        
        self._calculate_round_metrics(server_round, round_metrics)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        if results:
            accuracies = []
            for _, evaluate_res in results:
                accuracy = evaluate_res.metrics.get("accuracy", 0)
                accuracies.append(accuracy)
            
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            self.metrics_history['accuracy'].append(avg_accuracy)
            
            print(f"📈 R{server_round} - Acurácia média: {avg_accuracy:.3f}")
        
        return super().aggregate_evaluate(server_round, results, failures)

    def _calculate_round_metrics(self, server_round: int, round_metrics: Dict):
        if round_metrics['loss']:
            avg_loss = np.mean(round_metrics['loss'])
            total_epsilon = sum(round_metrics['epsilon_used'])
            avg_noise = np.mean(round_metrics['noise_multipliers'])
            
            self.metrics_history['round'].append(server_round)
            self.metrics_history['loss'].append(avg_loss)
            self.metrics_history['privacy_consumption'].append({
                'total': total_epsilon,
                'mean': np.mean(round_metrics['epsilon_used']),
                'std': np.std(round_metrics['epsilon_used']),
                'min': min(round_metrics['epsilon_used']),
                'max': max(round_metrics['epsilon_used']),
                'avg_noise': avg_noise
            })
            
            # Calcular fairness (dispersão do ε)
            epsilon_used = round_metrics['epsilon_used']
            if len(epsilon_used) > 1 and np.mean(epsilon_used) > 0:
                epsilon_std = np.std(epsilon_used)
                epsilon_mean = np.mean(epsilon_used)
                fairness = 1 - (epsilon_std / epsilon_mean)
            else:
                fairness = 1.0
                
            self.metrics_history['fairness'].append(fairness)
            
            # Calcular eficiência (accuracy por epsilon)
            if total_epsilon > 0 and self.metrics_history['accuracy']:
                current_accuracy = self.metrics_history['accuracy'][-1] if self.metrics_history['accuracy'] else 0
                efficiency = current_accuracy / total_epsilon
                self.metrics_history['efficiency'].append(efficiency)
            
            print(f"📊 R{server_round} - Loss: {avg_loss:.4f}, ε Total: {total_epsilon:.2f}")
            print(f"   🔊 Noise médio: {avg_noise:.2f}, ⚖️ Fairness: {fairness:.3f}")

    def save_metrics(self, filename: str = "metrics_dp_sgd.json"):
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': time.time() - self.start_time,
            'total_rounds': len(self.metrics_history['round']),
            'final_accuracy': self.metrics_history['accuracy'][-1] if self.metrics_history['accuracy'] else 0,
            'total_epsilon_used': sum([x.get('total', 0) for x in self.metrics_history['privacy_consumption']]),
            'avg_fairness': np.mean(self.metrics_history['fairness']) if self.metrics_history['fairness'] else 0,
            'avg_efficiency': np.mean(self.metrics_history['efficiency']) if self.metrics_history['efficiency'] else 0,
            'client_epsilons': self.metrics_history['client_epsilons'],
            'metrics_history': self.metrics_history,
            'strategy': 'dp-sgd',
            'target_epsilon': self.target_epsilon,
            'target_delta': self.target_delta
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"💾 Métricas DP-SGD salvas em {filename}")

def main():
    strategy = DPSGDServer(total_rounds=15, target_epsilon=10.0, target_delta=1e-5)
    
    try:
        print("🌐 Iniciando servidor DP-SGD na porta 8083...")
        fl.server.start_server(
            server_address="0.0.0.0:8083",
            config=fl.server.ServerConfig(num_rounds=15),
            strategy=strategy,
        )
    except Exception as e:
        print(f"❌ Erro no servidor: {e}")
    finally:
        strategy.save_metrics()
        print("✅ Servidor DP-SGD finalizado")

if __name__ == "__main__":
    main()
