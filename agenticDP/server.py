#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Metrics, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _hdr() -> None:
    print(f"### AGENTIC-DP SERVER v4 ### PYTHON= {sys.executable}")
    print("🚀 Servidor Agentic DP inicializado")
    print("🎯 Estratégia: Alocação descentralizada (client-side) de ε + métricas agregadas")
    sys.stdout.flush()


def weighted_avg_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics) or 1
    keys = set().union(*[m.keys() for _, m in metrics])
    return {k: sum(n * float(m.get(k, 0.0)) for n, m in metrics) / total for k in keys}


class AgenticFedAvg(FedAvg):
    """FedAvg com logs e salvaguardas contra soma de exemplos igual a zero."""

    def __init__(self, *, num_rounds: int, log_dir: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.log_dir = log_dir
        self.metrics_path = os.path.join(log_dir, "metrics_agentic.json")
        self.history: Dict[str, list] = {"rounds": []}

    def configure_fit(  # type: ignore[override]
        self, server_round: int, parameters, client_manager: ClientManager
    ):
        print(f"\n🔧 RODADA {server_round} - Agentic DP Allocation (client-side)")
        sys.stdout.flush()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(  # type: ignore[override]
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[ClientProxy, fl.common.FitRes]],
    ):
        # Salvaguarda: evita ZeroDivisionError quando todos retornam num_examples=0
        total_examples = sum(fr.num_examples for _, fr in results)
        if total_examples == 0:
            print("⚠️  Todos os clientes reportaram num_examples=0 nesta rodada; "
                  "mantendo parâmetros globais anteriores.")
            # Log por cliente, mesmo sem agregar
            for client_proxy, fitres in results:
                cid = getattr(client_proxy, "cid", "N/A")
                m = fitres.metrics or {}
                def fmt(x, p): 
                    try: return f"{float(x):.{p}f}"
                    except Exception: return str(x)
                print(
                    f"   📊 Cliente {cid}: ε={fmt(m.get('epsilon', None),2)}, "
                    f"Loss={fmt(m.get('loss', None),4)}, "
                    f"Imp={fmt(m.get('importance', None),2)}, "
                    f"Budget_rem={fmt(m.get('budget_remaining', None),2)}"
                )
            sys.stdout.flush()

            self.history["rounds"].append(
                {
                    "round": server_round,
                    "clients": [
                        {
                            "cid": getattr(cp, "cid", "N/A"),
                            **(fr.metrics or {}),
                            "num_examples": fr.num_examples,
                        }
                        for cp, fr in results
                    ],
                }
            )
            try:
                with open(self.metrics_path, "w") as f:
                    json.dump(self.history, f, indent=2)
                print("💾 Métricas parciais salvas em metrics_agentic.json")
            except Exception as e:
                print(f"⚠️ Falha ao salvar métricas: {e}")

            # Retorna None para manter parâmetros atuais
            return None, {}

        # Caminho normal
        agg = super().aggregate_fit(server_round, results, failures)

        print(f"🧮 RODADA {server_round} - Agregando {len(results)} resultados")
        for client_proxy, fitres in results:
            cid = getattr(client_proxy, "cid", "N/A")
            m = fitres.metrics or {}
            def fmt(x, p): 
                try: return f"{float(x):.{p}f}"
                except Exception: return str(x)
            print(
                f"   📊 Cliente {cid}: ε={fmt(m.get('epsilon', None),2)}, "
                f"Loss={fmt(m.get('loss', None),4)}, "
                f"Imp={fmt(m.get('importance', None),2)}, "
                f"Budget_rem={fmt(m.get('budget_remaining', None),2)}"
            )
        sys.stdout.flush()

        self.history["rounds"].append(
            {
                "round": server_round,
                "clients": [
                    {
                        "cid": getattr(cp, "cid", "N/A"),
                        **(fr.metrics or {}),
                        "num_examples": fr.num_examples,
                    }
                    for cp, fr in results
                ],
            }
        )
        try:
            with open(self.metrics_path, "w") as f:
                json.dump(self.history, f, indent=2)
            print("💾 Métricas parciais salvas em metrics_agentic.json")
        except Exception as e:
            print(f"⚠️ Falha ao salvar métricas: {e}")

        return agg


def on_fit_config_fn_factory(num_rounds: int):
    def _fn(server_round: int):
        return {"round": server_round, "num_rounds": num_rounds}
    return _fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8084)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--logdir", type=str, default=".")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _hdr()
    address = f"{args.host}:{args.port}"
    print(f"🌐 Iniciando servidor Agentic DP na porta {args.port}... FLWR= {fl.__version__}")

    strategy = AgenticFedAvg(
        num_rounds=args.rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        on_fit_config_fn=on_fit_config_fn_factory(args.rounds),
        evaluate_metrics_aggregation_fn=weighted_avg_metrics,
        fit_metrics_aggregation_fn=weighted_avg_metrics,
        log_dir=args.logdir,
    )

    start = datetime.now()
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    end = datetime.now()
    print(f"\n[SUMMARY] Rodadas: {args.rounds} | Duração: {(end-start).total_seconds():.2f}s")


if __name__ == "__main__":
    main()

