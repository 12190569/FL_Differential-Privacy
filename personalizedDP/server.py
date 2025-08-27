import flwr as fl
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import time
from datetime import datetime
import re

class PersonalizedDPServer(fl.server.strategy.FedAvg):
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
            'client_epsilons': []  # 🆕 ε por cliente por rodada
        }
        self.start_time = time.time()
        print("🚀 Servidor Personalized DP inicializado")
        print("🎯 Alocação: ε personalizado por cliente")

    def personalize_epsilon(self, client_id, server_round):
        """Aloca ε personalizado - CORREÇÃO PARA DIFERENCIAÇÃO"""
        
        # 🎯 FORÇAR DIFERENCIAÇÃO para teste
        if client_id == 0:
            base_epsilon = 7.0  # Dataset grande
            reason = "Dataset grande (15k amostras)"
        elif client_id == 1:
            base_epsilon = 6.0  # Dataset médio
            reason = "Dataset médio"
        elif client_id == 2:
            base_epsilon = 5.0  # Dataset pequeno
            reason = "Dataset pequeno"
        else:
            base_epsilon = 4.0  # Dados sensíveis
            reason = "Dados sensíveis - máxima proteção"
        
        # Aumentar epsilon gradualmente
        epsilon = base_epsilon + (server_round * 0.2)
        
        return max(min(epsilon, 10.0), 3.0), reason

    def fit_config(self, server_round: int):
        """Configurações base para todas as chamadas fit"""
        return {
            "round": server_round,
            "learning_rate": 0.001,
            "epochs": 1,
            "delta": 1e-5,
            "clip_norm": 2.0
        }

    def eval_config(self, server_round: int):
        return {"round": server_round}

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"\n🔧 RODADA {server_round} - Personalizando ε por cliente")
        
        # DEBUG: Mostrar informações dos clientes
        print("   🔍 Debug - Informações dos clientes:")
        all_clients = list(client_manager.all().values())
        for i, client in enumerate(all_clients):
            client_id_debug = self._get_client_id(client)
            client_str = str(client)
            print(f"      Cliente {i}: ID={client_id_debug}, repr={client_str[:50]}...")
    
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        client_epsilons = {}
        
        for client, fit_ins in client_instructions:
            client_id = self._get_client_id(client)  # Extrair ID do cliente
            epsilon, reason = self.personalize_epsilon(client_id, server_round)
            
            fit_ins.config["epsilon"] = epsilon
            fit_ins.config["personalization_reason"] = reason
            fit_ins.config["client_id"] = client_id
            
            client_epsilons[client_id] = epsilon
            print(f"   👤 Cliente {client_id}: ε={epsilon:.1f} - {reason}")
        
        self.metrics_history['client_epsilons'].append(client_epsilons)
        return client_instructions

    def _get_client_id(self, client):
        """Extrai ID do cliente corretamente - CORREÇÃO DEFINITIVA"""
        try:
            # Método 1: Tentar extrair do CID (approach mais confiável no Flower)
            if hasattr(client, 'cid'):
                cid = client.cid
                # Extrair número do CID (ex: "client_0" -> 0)
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
        
        # Fallback: usar mapping baseado no hash (mais robusto)
        client_hash = hash(str(client))
        return abs(client_hash) % 4

    def aggregate_fit(self, server_round, results, failures):
        print(f"🔧 RODADA {server_round} - Agregando {len(results)} resultados")
        
        if not results:
            return None
            
        round_metrics = {'loss': [], 'epsilon_used': [], 'client_ids': []}
        
        for client, fit_res in results:
            metrics = fit_res.metrics
            client_id = metrics.get("client_id", -1)
            epsilon_used = metrics.get("epsilon_used", 0)
            loss_val = metrics.get("loss", 0)
            
            round_metrics['loss'].append(loss_val)
            round_metrics['epsilon_used'].append(epsilon_used)
            round_metrics['client_ids'].append(client_id)
            
            print(f"   📊 Cliente {client_id}: "
                  f"ε={epsilon_used:.1f}, "
                  f"Loss={loss_val:.4f}")
        
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
            
            self.metrics_history['round'].append(server_round)
            self.metrics_history['loss'].append(avg_loss)
            self.metrics_history['privacy_consumption'].append({
                'total': total_epsilon,
                'mean': np.mean(round_metrics['epsilon_used']),
                'std': np.std(round_metrics['epsilon_used']),
                'min': min(round_metrics['epsilon_used']),
                'max': max(round_metrics['epsilon_used'])
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
            
            print(f"📊 R{server_round} - Loss médio: {avg_loss:.4f}, ε total: {total_epsilon:.1f}")
            print(f"   ⚖️  Fairness: {fairness:.3f}")
            print(f"   📶 ε por cliente: {[f'{e:.1f}' for e in round_metrics['epsilon_used']]}")

    def save_metrics(self, filename: str = "metrics_personalized.json"):
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': time.time() - self.start_time,
            'total_rounds': len(self.metrics_history['round']),
            'final_accuracy': self.metrics_history['accuracy'][-1] if self.metrics_history['accuracy'] else 0,
            'total_epsilon_used': sum([x.get('total', 0) for x in self.metrics_history['privacy_consumption']]),
            'avg_fairness': np.mean(self.metrics_history['fairness']) if self.metrics_history['fairness'] else 0,
            'client_epsilons': self.metrics_history['client_epsilons'],
            'metrics_history': self.metrics_history,
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"💾 Métricas personalizadas salvas em {filename}")

def main():
    strategy = PersonalizedDPServer(total_rounds=15)
    
    try:
        print("🌐 Iniciando servidor Personalized DP na porta 8082...")
        fl.server.start_server(
            server_address="0.0.0.0:8082",
            config=fl.server.ServerConfig(num_rounds=15),
            strategy=strategy,
        )
    except Exception as e:
        print(f"❌ Erro no servidor: {e}")
    finally:
        strategy.save_metrics()
        print("✅ Servidor Personalized DP finalizado")

if __name__ == "__main__":
    main()
