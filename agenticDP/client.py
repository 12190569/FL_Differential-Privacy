#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import random
import sys
from dataclasses import dataclass
from typing import Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# ------------------------
# Util & Modelo
# ------------------------

def _hdr(client_id: int) -> None:
    print(f"### AGENTIC-DP CLIENT v4 ### PYTHON= {sys.executable} FLWR= {fl.__version__}")
    print(f"✅ Cliente {client_id} - inicializando...")
    sys.stdout.flush()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x


def partition_dataset(dataset, cid: int, num_clients: int):
    n = len(dataset)
    shard = n // num_clients
    start = cid * shard
    end = (cid + 1) * shard if cid < num_clients - 1 else n
    subset = torch.utils.data.Subset(dataset, list(range(start, end)))
    return subset


@dataclass
class AgentConfig:
    eps_budget_total: float = 18.0
    eps_max_round: float = 2.5
    eps_min_train: float = 0.20       # abaixo disso, pula treino
    clip_norm: float = 1.0
    lr: float = 1e-3                  # Adam base
    batch_size: int = 128
    local_epochs: int = 1
    k_nm: float = 0.5                 # k para noise_multiplier ~ k/eps
    nm_min: float = 0.05
    nm_max: float = 1.5
    seed: int = 42


class PrivacyAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.budget_rem = cfg.eps_budget_total

    def _probe_importance(self, probe_loss: float) -> float:
        # Importância ∈ [0.2, 0.8] mapeada a partir da perda de sonda
        x = max(0.0, min(3.0, probe_loss - 1.5))
        imp = 0.2 + 0.6 * (math.tanh(x) * 0.5 + 0.5)
        return float(max(0.0, min(1.0, imp)))

    def decide_epsilon(self, round_idx: int, num_rounds: int, probe_loss: float) -> float:
        """Decisão stateful com limitador de ritmo de gasto por rodada restante."""
        if self.budget_rem <= 0.0:
            return 0.0

        imp = self._probe_importance(probe_loss)
        urgency = 0.4 + 0.6 * (round_idx / max(1, num_rounds))  # 0.4..1.0
        r = max(0.05, min(1.0, self.budget_rem / self.cfg.eps_budget_total))

        base = (0.9 * imp + 0.6 * urgency) * self.cfg.eps_max_round
        candidate = base * r

        # Limitador de ritmo: não gastar muito acima da média restante
        rounds_left = max(1, num_rounds - round_idx + 1)
        pace = self.budget_rem / rounds_left
        candidate = min(candidate, 2.0 * pace)  # no máx. 2x da média restante

        # Limites finais
        candidate = max(0.0, min(candidate, self.cfg.eps_max_round, self.budget_rem))
        return float(candidate)

    def spend(self, eps: float) -> None:
        self.budget_rem = float(max(0.0, self.budget_rem - eps))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def noise_std_from_epsilon(
    clip_norm: float, eps: float, batch_size: int, k_nm: float, nm_min: float, nm_max: float
) -> float:
    """DP-SGD style: std = (nm * clip_norm) / batch_size, nm ~ k/eps (clamped)."""
    eps = max(eps, 1e-3)
    nm = k_nm / eps
    nm = max(nm_min, min(nm_max, nm))
    return float((nm * clip_norm) / max(1, batch_size))


# ------------------------
# Cliente Flower
# ------------------------

class AgenticClient(fl.client.NumPyClient):
    def __init__(self, cid: int, server_addr: str, num_clients: int, cfg: AgentConfig, device: torch.device):
        self.cid = cid
        self.server_addr = server_addr
        self.num_clients = num_clients
        self.cfg = cfg
        self.device = device

        set_seed(cfg.seed + cid)

        # Dados
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_ds_full = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_ds_full = datasets.MNIST("./data", train=False, download=True, transform=transform)
        self.train_ds = partition_dataset(train_ds_full, cid, num_clients)
        self.test_ds = partition_dataset(test_ds_full, cid, num_clients)

        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=512, shuffle=False)

        # Modelo/otimizador
        self.model = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-4)

        # Agente stateful (mantém budget)
        self.agent = PrivacyAgent(self.cfg)

        self.last_importance = 0.5

        print(f"📊 Cliente {cid} - Treino: {len(self.train_ds)}, Teste: {len(self.test_ds)}")
        print(f"🚀 Cliente Agentic DP {cid} inicializado")
        print(f"🌐 Conectando cliente {cid} ao servidor {self.server_addr}...")
        sys.stdout.flush()

    # Parâmetros
    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for (k, _), p in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(p, dtype=state_dict[k].dtype)
        self.model.load_state_dict(state_dict)

    # Avaliação simples
    def _eval_loss_acc(self, loader) -> Tuple[float, float]:
        self.model.eval()
        total, loss_sum, correct = 0, 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                loss_sum += float(loss.item()) * x.size(0)
                total += x.size(0)
                pred = out.argmax(dim=1)
                correct += int((pred == y).sum().item())
        return loss_sum / max(1, total), correct / max(1, total)

    def _train_with_dp(self, epsilon: float) -> float:
        """1 época local com clipping + ruído gaussiano (escala por batch_size)."""
        self.model.train()
        loss_last = 0.0

        # Escalar LR pela “qualidade” do passo (ε relativo ao máx.)
        lr_scale = float(max(1e-3, epsilon / self.cfg.eps_max_round))
        for g in self.optimizer.param_groups:
            g["lr"] = max(1e-5, self.cfg.lr * lr_scale)

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()

            # Clip global
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.clip_norm)

            # Ruído DP por batch
            std = noise_std_from_epsilon(
                self.cfg.clip_norm, epsilon, self.cfg.batch_size, self.cfg.k_nm, self.cfg.nm_min, self.cfg.nm_max
            )
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.normal(mean=0.0, std=std, size=p.grad.shape,
                                         device=p.grad.device, dtype=p.grad.dtype)
                    p.grad.add_(noise)

            self.optimizer.step()
            loss_last = float(loss.item())

        return loss_last

    # Métodos Flower
    def fit(self, parameters, config):
        round_idx = int(config.get("round", 0))
        num_rounds = int(config.get("num_rounds", 1))

        # Recebe parâmetros globais
        self.set_parameters(parameters)

        # "Sonda" de importância
        probe_loss, _ = self._eval_loss_acc(self.train_loader)
        imp = self.agent._probe_importance(probe_loss)

        # Decide ε com orçamento stateful e pace limit
        epsilon = self.agent.decide_epsilon(round_idx, num_rounds, probe_loss)

        # Pular treino se ε muito baixo (mas reportar num_examples > 0)
        if epsilon <= self.cfg.eps_min_train or self.agent.budget_rem <= 0.0:
            self.last_importance = imp
            print(f"⚠️ Cliente {self.cid} - R{round_idx}: ε≈0 -> pulando treino "
                  f"(budget_restante={self.agent.budget_rem:.2f})")
            return self.get_parameters(config), len(self.train_ds), {
                "epsilon": 0.0,
                "loss": float(probe_loss),
                "importance": float(imp),
                "budget_remaining": float(self.agent.budget_rem),
            }

        # Treina com DP
        last_loss = self._train_with_dp(epsilon)
        # Consome orçamento
        self.agent.spend(epsilon)
        self.last_importance = imp

        why = []
        if self.agent.budget_rem > self.cfg.eps_budget_total * 0.5:
            why.append("orçamento alto")
        if imp >= 0.6:
            why.append("importância alta")
        if round_idx / max(1, num_rounds) >= 0.5:
            why.append("urgência")
        why_s = " / ".join(why) if why else "normal"

        print(
            f"🔒 Cliente {self.cid} - R{round_idx}: ε={epsilon:.2f} ({why_s}), "
            f"Imp={imp:.2f}, Loss={last_loss:.4f}"
        )
        sys.stdout.flush()

        return self.get_parameters(config), len(self.train_ds), {
            "epsilon": float(epsilon),
            "loss": float(last_loss),
            "importance": float(imp),
            "budget_remaining": float(self.agent.budget_rem),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = self._eval_loss_acc(self.test_loader)
        print(f"📊 Cliente {self.cid} - Avaliação: Loss={loss:.4f}, Acc={acc:.3f}")
        sys.stdout.flush()
        return float(loss), len(self.test_ds), {"accuracy": float(acc)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server", type=str, default="127.0.0.1:8084")
    p.add_argument("--cid", type=int, required=True)
    p.add_argument("--num_clients", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    _hdr(args.cid)
    device = torch.device("cpu")
    cfg = AgentConfig()

    client = AgenticClient(
        cid=args.cid,
        server_addr=args.server,
        num_clients=args.num_clients,
        cfg=cfg,
        device=device,
    )

    # Usa .to_client() para remover o warning deprecação do tipo
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()

