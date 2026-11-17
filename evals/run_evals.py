"""
Runner de Avaliação e Benchmark de Evals (Engenharia de IA)
Executa as 15 perguntas técnicas sobre o voo de telemetria,
mensura acurácia de execução de código, taxa de auto-correção (Self-Healing),
latência média e conformidade de saídas.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Garante path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.profiler import TelemetryProfiler
from src.agent import TelemetryCopilotAgent


BASE_DIR = Path(__file__).resolve().parent.parent

def run_benchmark(
    queries_file: str = str(BASE_DIR / "evals" / "test_queries.json"),
    csv_file: str = str(BASE_DIR / "data" / "sample_flight.csv"),
    output_file: str = str(BASE_DIR / "evals" / "eval_results.json"),
    provider: str = "gemini",
    model: str = "gemini-3.5-flash-lite",
    max_queries: int = 15
) -> Dict[str, Any]:
    print(f"=== INICIANDO BENCHMARK DE EVALS ({provider} / {model}) ===")
    print(f"Dataset: {csv_file}")
    print(f"Queries: {queries_file}")

    df = pd.read_csv(csv_file)
    profiler = TelemetryProfiler(df)
    agent = TelemetryCopilotAgent(provider=provider, model=model)

    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if max_queries:
        queries = queries[:max_queries]

    results: List[Dict[str, Any]] = []
    passed = 0
    self_healed_count = 0
    total_latency_ms = 0.0

    for idx, item in enumerate(queries, 1):
        qid = item["id"]
        q_text = item["query"]
        req_plot = item.get("requires_plot", False)
        print(f"\n[{idx}/{len(queries)}] Executando {qid}: '{q_text}'...")

        t_start = time.perf_counter()
        resp = agent.run_query(q_text, df, profiler)
        lat = (time.perf_counter() - t_start) * 1000
        total_latency_ms += lat

        exec_success = resp.execution_result.get("success", False)
        has_fig = resp.has_figure
        metrics = resp.execution_result.get("metrics", {})

        # Critério de sucesso: execução sem erro e geração de figura (se requerida) ou métricas
        is_pass = exec_success and (not req_plot or has_fig)
        if is_pass:
            passed += 1

        if resp.self_healed:
            self_healed_count += 1

        print(f" -> Resultado: {'PASS' if is_pass else 'FAIL'} | Latência: {lat:.1f}ms | Auto-corrigido: {resp.self_healed} | Métricas: {len(metrics)}")

        results.append({
            "id": qid,
            "category": item.get("category", "General"),
            "query": q_text,
            "passed": is_pass,
            "execution_success": exec_success,
            "has_figure": has_fig,
            "self_healed": resp.self_healed,
            "retries": resp.retries_count,
            "latency_ms": round(lat, 2),
            "metrics_extracted": metrics,
            "error": resp.execution_result.get("error")
        })

    total = len(queries)
    pass_rate = round((passed / total) * 100, 2) if total > 0 else 0.0
    avg_lat = round(total_latency_ms / total, 2) if total > 0 else 0.0

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
        "model": agent.model_name,
        "total_queries": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate_pct": pass_rate,
        "self_healed_count": self_healed_count,
        "avg_latency_ms": avg_lat,
        "queries_evaluated": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"BENCHMARK CONCLUÍDO:")
    print(f"Taxa de Acerto (Pass Rate): {passed}/{total} ({pass_rate}%)")
    print(f"Auto-correções com Sucesso: {self_healed_count}")
    print(f"Latência Média: {avg_lat:.1f} ms")
    print(f"Relatório gravado em: {output_file}")
    print("=" * 50)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=str(BASE_DIR / "evals" / "test_queries.json"))
    parser.add_argument("--csv", default=str(BASE_DIR / "data" / "sample_flight.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "evals" / "eval_results.json"))
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--max", type=int, default=5)  # default quick run
    args = parser.parse_args()

    run_benchmark(
        queries_file=args.queries,
        csv_file=args.csv,
        output_file=args.output,
        provider=args.provider,
        model=args.model,
        max_queries=args.max
    )
