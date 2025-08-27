import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
import os
from torch.utils.data import TensorDataset, DataLoader

class OptimizedNet(nn.Module):
    """Rede neural otimizada para MNIST"""
    def __init__(self):
        super(OptimizedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class AgenticDPClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.client_type = "agentic-dp"
        self.net = OptimizedNet()
        self.trainloader, self.testloader = self.load_data(client_id)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_history = []
        self.accuracy_history = []
        self.gradient_importance_history = []
        
        print(f"🚀 Cliente Agentic DP {client_id} inicializado")

    def load_data(self, client_id):
        """Carrega dados MNIST com fallback para dados dummy"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        try:
            train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
            print(f"✅ Cliente {client_id} - MNIST carregado")
        except:
            dummy_data = torch.randn(1000, 1, 28, 28)
            dummy_targets = torch.randint(0, 10, (1000,))
            train_dataset = TensorDataset(dummy_data, dummy_targets)
            test_dataset = TensorDataset(dummy_data, dummy_targets)
            print(f"📦 Cliente {client_id} - Usando dados dummy")
        
        train_size = len(train_dataset) // 4
        test_size = len(test_dataset) // 4
        
        train_start = client_id * train_size
        train_end = min((client_id + 1) * train_size, len(train_dataset))
        test_start = client_id * test_size
        test_end = min((client_id + 1) * test_size, len(test_dataset))
        
        train_subset = torch.utils.data.Subset(train_dataset, range(train_start, train_end))
        test_subset = torch.utils.data.Subset(test_dataset, range(test_start, test_end))
        
        print(f"📊 Cliente {client_id} - Treino: {len(train_subset)}, Teste: {len(test_subset)}")
        
        return (
            DataLoader(train_subset, batch_size=64, shuffle=True),
            DataLoader(test_subset, batch_size=64)
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.net.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.net.state_dict().keys(), parameters)}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Treinamento com Agentic DP"""
        self.set_parameters(parameters)
        
        epsilon = config.get("epsilon", 8.0)
        delta = config.get("delta", 1e-5)
        clip_norm = config.get("clip_norm", 1.5)
        learning_rate = config.get("learning_rate", 0.001)
        epochs = config.get("epochs", 1)
        round_idx = config.get("round", 0)
        allocation_reason = config.get("allocation_reason", "N/A")
        gradient_importance = config.get("gradient_importance", 0.5)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        self.net.train()
        total_loss = 0
        
        for data, target in self.trainloader:
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            
            # Aplicar DP com ruído adaptativo baseado na importância
            self.apply_adaptive_dp(clip_norm, epsilon, delta, gradient_importance)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.trainloader)
        self.loss_history.append(avg_loss)
        self.gradient_importance_history.append(gradient_importance)
        
        print(f"🔒 Cliente {self.client_id} - R{round_idx}: "
              f"ε={epsilon:.2f} ({allocation_reason}), "
              f"Importância={gradient_importance:.2f}, Loss={avg_loss:.4f}")
        
        if round_idx % 2 == 0 and len(self.loss_history) > 1:
            loss_change = self.loss_history[-2] - self.loss_history[-1]
            trend = "↘️" if loss_change > 0 else "↗️" if loss_change < 0 else "➡️"
            print(f"   📉 Progresso: Loss {avg_loss:.1f} {trend} (Δ{loss_change:+.1f})")
        
        return self.get_parameters(config), len(self.trainloader.dataset), {
            "client_type": self.client_type,
            "epsilon_used": epsilon,
            "loss": avg_loss,
            "client_id": self.client_id,
            "allocation_reason": allocation_reason,
            "gradient_importance": gradient_importance
        }

    def apply_adaptive_dp(self, clip_norm: float, epsilon: float, delta: float, importance: float):
        """Aplica DP adaptativo baseado na importância dos gradientes"""
        if epsilon <= 0:
            return
        
        # Clipping de gradientes
        total_norm = 0.0
        for p in self.net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.net.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        # Ajustar ruído baseado na importância (mais importância = menos ruído)
        importance_factor = 1.0 - (importance * 0.3)  # 0.7 to 1.0 range
        effective_epsilon = epsilon * importance_factor
        
        # Calcular e aplicar ruído
        sigma = math.sqrt(2 * math.log(1.25 / delta)) / effective_epsilon
        
        for p in self.net.parameters():
            if p.grad is not None:
                noise = torch.normal(0, sigma * clip_norm, p.grad.shape)
                p.grad.data.add_(noise)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.net.eval()
        
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in self.testloader:
                output = self.net(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        test_loss /= len(self.testloader.dataset)
        accuracy = correct / len(self.testloader.dataset)
        self.accuracy_history.append(accuracy)
        
        print(f"📊 Cliente {self.client_id} - Avaliação: "
              f"Loss={test_loss:.4f}, Acc={accuracy:.3f}")
        
        if len(self.accuracy_history) > 1:
            acc_change = self.accuracy_history[-1] - self.accuracy_history[-2]
            trend = "↗️" if acc_change > 0 else "↘️" if acc_change < 0 else "➡️"
            print(f"   📈 Accuracy: {accuracy:.3f} {trend} (Δ{acc_change:+.3f})")
        
        return test_loss, len(self.testloader.dataset), {"accuracy": accuracy}

def main():
    import sys
    if len(sys.argv) != 2:
        print("Uso: python client_agentic_dp.py <client_id>")
        return
    
    client_id = int(sys.argv[1])
    
    try:
        client = AgenticDPClient(client_id)
        print(f"🌐 Conectando cliente {client_id} ao servidor Agentic DP...")
        fl.client.start_client(
            server_address="0.0.0.0:8084",
            client=client.to_client(),
        )
        print(f"✅ Cliente {client_id} finalizado")
    except Exception as e:
        print(f"❌ Erro no cliente {client_id}: {e}")

if __name__ == "__main__":
    main()
