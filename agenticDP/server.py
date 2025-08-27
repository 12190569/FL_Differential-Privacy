import flwr as fl
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import time
from datetime import datetime
import re
import random

class AgenticDPServer(fl.server.strategy.FedAvg):
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
        self.client_budgets = {}
        self.client_performance = {}
        self.gradient_importance_history = {}
        self.metrics_history = {
            'round': [], 'loss': [], 'accuracy': [], 
            'privacy_consumption': [], 'fairness': [], 'efficiency': [],
            'client_epsilons': [], 'budget_utilization': [], 'gradient_importance': []
        }
        self.start_time = time.time()
        self.client_counter = 0
        print("🚀 Servidor Agentic DP inicializado")
        print("🎯 Estratégia: Alocação inteligente e descentralizada de ε")

    def initialize_client_budget(self, client_id):
        """Inicializa orçamento para novo cliente"""
        if client_id not in self.client_budgets:
            if client_id == 0:
                self.client_budgets[client_id] = 15.0  # Dataset grande
            elif client_id == 1:
                self.client_budgets[client_id] = 12.0  # Dataset médio
            elif client_id == 2:
                self.client_budgets[client_id] = 10.0  # Dataset pequeno
            else:
                self.client_budgets[client_id] = 18.0  # Dados sensíveis + orçamento extra

    def agentic_epsilon_allocation(self, client_id, server_round, last_loss=None):
        """Alocação inteligente de ε pelo agente local"""
        self.initialize_client_budget(client_id)
        remaining_budget = self.client_budgets[client_id]
        
        progress = server_round / self.total_rounds
        rounds_remaining = self.total_rounds - server_round
        
        # Base allocation
        base_epsilon = remaining_budget / max(rounds_remaining, 1)
        
        # Adjust based on performance history
        performance_factor = 1.0
        if client_id in self.client_performance and len(self.client_performance[client_id]) > 0:
            avg_loss = np.mean(self.client_performance[client_id][-3:])  # Last 3 rounds
            if avg_loss < 1.0:  # Good performance
                performance_factor = 0.8  # Use less epsilon
            else:  # Poor performance
                performance_factor = 1.2  # Use more epsilon
        
        # Adjust based on round progress
        urgency_factor = 1.0 + (progress * 0.3)
        
        # Adjust based on remaining budget
        budget_factor = 1.0
        if remaining_budget < 3.0:
            budget_factor = 0.7  # Conservative when budget is low
        elif remaining_budget > 10.0:
            budget_factor = 1.2  # More aggressive when budget is high
        
        epsilon = base_epsilon * performance_factor * urgency_factor * budget_factor
        
        # Apply constraints
        epsilon = max(min(epsilon, remaining_budget * 0.7), 0.8)
        
        # Determine reason
        reasons = []
        if performance_factor < 1.0:
            reasons.append("boa performance")
        elif performance_factor > 1.0:
            reasons.append("performance ruim")
        if budget_factor < 1.0:
            reasons.append("orçamento baixo")
        elif budget_factor > 1.0:
            reasons.append("orçamento alto")
        if urgency_factor > 1.1:
            reasons.append("urgência alta")
        
        reason = "Alocação " + (" + ".join(reasons) if reasons else "equilibrada")
        
        return epsilon, reason, remaining_budget

    def fit_config(self, server_round: int):
        return {
            "round": server_round,
            "learning_rate": 0.001,
            "epochs": 1,
            "delta": 1e-5,
            "clip_norm": 1.5,
            "dp_mechanism": "agentic-dp"
        }

    def eval_config(self, server_round: int):
        return {"round": server_round}

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"\n🔧 RODADA {server_round} - Agentic DP Allocation")
        
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        client_epsilons = {}
        budget_utilization = {}
        gradient_importance = {}
        
        for client, fit_ins in client_instructions:
            client_id = self._get_client_id(client)
            
            # Get last loss for performance-based allocation
            last_loss = None
            if client_id in self.client_performance and self.client_performance[client_id]:
                last_loss = self.client_performance[client_id][-1]
            
            epsilon, reason, remaining_budget = self.agentic_epsilon_allocation(
                client_id, server_round, last_loss
            )
            
            # Simulate gradient importance (would come from client in real implementation)
            gradient_importance_val = random.uniform(0.4, 0.9)
            
            fit_ins.config["epsilon"] = epsilon
            fit_ins.config["allocation_reason"] = reason
            fit_ins.config["client_id"] = client_id
            fit_ins.config["gradient_importance"] = gradient_importance_val
            
            client_epsilons[client_id] = epsilon
            budget_utilization[client_id] = {
                'remaining_budget': remaining_budget,
                'used_epsilon': epsilon,
                'allocation_reason': reason
            }
            gradient_importance[client_id] = gradient_importance_val
            
            print(f"   👤 Cliente {client_id}: ε={epsilon:.2f} - {reason}")
            print(f"      📊 Orçamento: {remaining_budget:.2f}, Importância: {gradient_importance_val:.2f}")
        
        self.metrics_history['client_epsilons'].append(client_epsilons)
        self.metrics_history['budget_utilization'].append(budget_utilization)
        self.metrics_history['gradient_importance'].append(gradient_importance)
        return client_instructions

    def _get_client_id(self, client):
        try:
            if hasattr(client, 'cid'):
                cid = client.cid
                numbers = re.findall(r'\d+', str(cid))
                if numbers:
                    return int(numbers[0]) % 4
        except:
            pass
        
        client_hash = hash(str(client))
        return abs(client_hash) % 4

    def aggregate_fit(self, server_round, results, failures):
        print(f"🔧 RODADA {server_round} - Agregando {len(results)} resultados Agentic DP")
        
        if not results:
            return None
            
        round_metrics = {'loss': [], 'epsilon_used': [], 'client_ids': [], 'gradient_importance': []}
        
        for client, fit_res in results:
            metrics = fit_res.metrics
            client_id = metrics.get("client_id", -1)
            epsilon_used = metrics.get("epsilon_used", 0)
            loss_val = metrics.get("loss", 0)
            grad_importance = metrics.get("gradient_importance", 0.5)
            
            round_metrics['loss'].append(loss_val)
            round_metrics['epsilon_used'].append(epsilon_used)
            round_metrics['client_ids'].append(client_id)
            round_metrics['gradient_importance'].append(grad_importance)
            
            # Update client performance
            if client_id not in self.client_performance:
                self.client_performance[client_id] = []
            self.client_performance[client_id].append(loss_val)
            
            # Update budget
            if client_id in self.client_budgets:
                self.client_budgets[client_id] -= epsilon_used
            
            print(f"   📊 Cliente {client_id}: ε={epsilon_used:.2f}, Loss={loss_val:.4f}, Importância={grad_importance:.2f}")
        
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
            avg_importance = np.mean(round_metrics['gradient_importance'])
            
            self.metrics_history['round'].append(server_round)
            self.metrics_history['loss'].append(avg_loss)
            self.metrics_history['privacy_consumption'].append({
                'total': total_epsilon,
                'mean': np.mean(round_metrics['epsilon_used']),
                'std': np.std(round_metrics['epsilon_used']),
                'min': min(round_metrics['epsilon_used']),
                'max': max(round_metrics['epsilon_used']),
                'avg_importance': avg_importance
            })
            
            # Calculate fairness
            epsilon_used = round_metrics['epsilon_used']
            if len(epsilon_used) > 1 and np.mean(epsilon_used) > 0:
                epsilon_std = np.std(epsilon_used)
                epsilon_mean = np.mean(epsilon_used)
                fairness = 1 - (epsilon_std / epsilon_mean)
            else:
                fairness = 1.0
                
            self.metrics_history['fairness'].append(fairness)
            
            # Calculate efficiency
            if total_epsilon > 0 and self.metrics_history['accuracy']:
                current_accuracy = self.metrics_history['accuracy'][-1] if self.metrics_history['accuracy'] else 0
                efficiency = current_accuracy / total_epsilon
                self.metrics_history['efficiency'].append(efficiency)
            
            print(f"📊 R{server_round} - Loss: {avg_loss:.4f}, ε Total: {total_epsilon:.2f}")
            print(f"   📶 Importância média: {avg_importance:.2f}, ⚖️ Fairness: {fairness:.3f}")

    def save_metrics(self, filename: str = "metrics_agentic_dp.json"):
        final_budgets = {cid: budget for cid, budget in self.client_budgets.items()}
        
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': time.time() - self.start_time,
            'total_rounds': len(self.metrics_history['round']),
            'final_accuracy': self.metrics_history['accuracy'][-1] if self.metrics_history['accuracy'] else 0,
            'total_epsilon_used': sum([x.get('total', 0) for x in self.metrics_history['privacy_consumption']]),
            'avg_fairness': np.mean(self.metrics_history['fairness']) if self.metrics_history['fairness'] else 0,
            'avg_efficiency': np.mean(self.metrics_history['efficiency']) if self.metrics_history['efficiency'] else 0,
            'final_budgets': final_budgets,
            'client_epsilons': self.metrics_history['client_epsilons'],
            'budget_utilization': self.metrics_history['budget_utilization'],
            'gradient_importance': self.metrics_history['gradient_importance'],
            'metrics_history': self.metrics_history,
            'strategy': 'agentic-dp'
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"💾 Métricas Agentic DP salvas em {filename}")

def main():
    strategy = AgenticDPServer(total_rounds=15)
    
    try:
        print("🌐 Iniciando servidor Agentic DP na porta 8084...")
        fl.server.start_server(
            server_address="0.0.0.0:8084",
            config=fl.server.ServerConfig(num_rounds=15),
            strategy=strategy,
        )
    except Exception as e:
        print(f"❌ Erro no servidor: {e}")
    finally:
        strategy.save_metrics()
        print("✅ Servidor Agentic DP finalizado")

if __name__ == "__main__":
    main()
