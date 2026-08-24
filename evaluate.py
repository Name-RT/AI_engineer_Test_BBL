"""
evaluate.py — Automated RAG Benchmark & Performance Evaluation Suite
Bangkok Bank AI Policy Assistant

Evaluates:
1. Retrieval Accuracy (Context Recall @ Top-K against expected ground-truth sections)
2. Response Groundedness Rate (% of answers fully grounded without hallucination)
3. Guardrail Precision (% of off-topic queries correctly caught and rejected)
4. End-to-End System Latency (per-query and average)
5. Exports results to JSON & Markdown report for submission
"""
import os
import sys
import time
import json
import logging
from typing import Dict, Any, List

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from config.settings import load_config, setup_logging
from agents.graph import create_graph

console = Console()
logger = logging.getLogger(__name__)


def load_benchmark_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load evaluation test cases."""
    if not os.path.isabs(dataset_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_dir, dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_benchmark(config: Dict[str, Any], limit: int = None) -> Dict[str, Any]:
    """Execute evaluation benchmark across all test cases."""
    dataset_rel_path = config.get("evaluation", {}).get("golden_dataset_path", "evaluation/golden_dataset.json")
    dataset = load_benchmark_dataset(dataset_rel_path)

    if limit and limit > 0:
        dataset = dataset[:limit]

    console.print(Panel(
        f"[bold cyan]🏦 BANGKOK BANK AI POLICY ASSISTANT[/bold cyan]\n"
        f"[bold white]Automated RAG Evaluation & Benchmark Suite[/bold white]\n"
        f"Dataset: [green]{dataset_rel_path}[/green] ({len(dataset)} Test Cases)\n"
        f"Retrieval Mode: [yellow]{config.get('search', {}).get('mode', 'chroma')}[/yellow]",
        border_style="blue"
    ))

    graph = create_graph(config)

    results = []
    total_latency = 0.0
    retrieval_hits = 0
    grounded_hits = 0
    guardrail_tested = 0
    guardrail_correct = 0

    table = Table(title="📊 Test Case Execution Details", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Query", style="cyan", width=32)
    table.add_column("Expected Section", style="magenta", width=22)
    table.add_column("Retrieved Section(s)", style="yellow", width=24)
    table.add_column("Recall", justify="center", width=8)
    table.add_column("Grounded", justify="center", width=10)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Time", justify="right", style="green", width=8)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        eval_task = progress.add_task("[cyan]Evaluating queries...", total=len(dataset))

        for idx, item in enumerate(dataset, start=1):
            query = item["question"]
            ground_truth = item.get("ground_truth", "")
            expected_sections = item.get("expected_sections", [])
            is_rejection_expected = "rejected" in ground_truth.lower() or not expected_sections

            initial_state = {
                "query": query,
                "expanded_query": "",
                "is_valid": True,
                "rejection_reason": "",
                "retrieved_documents": [],
                "retrieval_confidence": 0.0,
                "retrieval_attempts": 0,
                "generated_report": "",
                "is_grounded": False,
                "generation_attempts": 0,
                "final_answer": "",
                "error": "",
            }

            start_t = time.time()
            try:
                run_res = graph.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": f"eval_{time.time():.0f}_{idx}"}}
                )
                latency = time.time() - start_t
            except Exception as exc:
                latency = time.time() - start_t
                run_res = {
                    "final_answer": f"ERROR: {exc}",
                    "retrieved_documents": [],
                    "retrieval_confidence": 0.0,
                    "is_grounded": False,
                    "is_valid": False,
                }

            total_latency += latency
            retrieved_docs = run_res.get("retrieved_documents", [])
            confidence = run_res.get("retrieval_confidence", 0.0)
            is_grounded = run_res.get("is_grounded", False)
            is_valid = run_res.get("is_valid", True)
            final_answer = run_res.get("final_answer", "")

            # Extract retrieved section headers
            retrieved_sections = []
            for doc in retrieved_docs:
                content = doc.get("content", "")
                first_line = content.strip().split("\n")[0]
                if "===" in first_line:
                    header = first_line.replace("===", "").strip()
                    if header not in retrieved_sections:
                        retrieved_sections.append(header)

            # Check Context Recall
            if is_rejection_expected:
                guardrail_tested += 1
                if not is_valid or "ไม่สามารถประมวลผล" in final_answer or "rejected" in final_answer.lower():
                    guardrail_correct += 1
                    recall_ok = True
                else:
                    recall_ok = False
            else:
                if any(any(exp.lower() in ret.lower() for ret in retrieved_sections) for exp in expected_sections):
                    retrieval_hits += 1
                    recall_ok = True
                else:
                    recall_ok = False

            if is_grounded:
                grounded_hits += 1

            recall_badge = "[green]PASS[/green]" if recall_ok else "[red]FAIL[/red]"
            grounded_badge = "[green]YES[/green]" if is_grounded else "[yellow]N/A[/yellow]"

            exp_str = ", ".join(expected_sections) if expected_sections else "None (Reject)"
            ret_str = ", ".join(retrieved_sections[:2]) if retrieved_sections else "None"

            table.add_row(
                str(idx),
                query[:30] + ("..." if len(query) > 30 else ""),
                exp_str[:20] + ("..." if len(exp_str) > 20 else ""),
                ret_str[:22] + ("..." if len(ret_str) > 22 else ""),
                recall_badge,
                grounded_badge,
                f"{confidence:.2f}",
                f"{latency:.2f}s"
            )

            results.append({
                "test_id": idx,
                "query": query,
                "expected_sections": expected_sections,
                "retrieved_sections": retrieved_sections,
                "ground_truth": ground_truth,
                "final_answer": final_answer,
                "confidence": round(confidence, 4),
                "is_grounded": is_grounded,
                "recall_passed": recall_ok,
                "latency_seconds": round(latency, 3),
            })

            progress.update(eval_task, advance=1)

    console.print(table)

    # ── Summary Statistics ─────────────────────────────────────────────────────
    total_queries = len(dataset)
    non_rejected_queries = total_queries - guardrail_tested
    retrieval_recall = (retrieval_hits / non_rejected_queries * 100) if non_rejected_queries > 0 else 100.0
    grounded_rate = (grounded_hits / non_rejected_queries * 100) if non_rejected_queries > 0 else 100.0
    guardrail_accuracy = (guardrail_correct / guardrail_tested * 100) if guardrail_tested > 0 else 100.0
    avg_latency = total_latency / total_queries if total_queries > 0 else 0.0

    summary_table = Table(title="🏆 Overall Benchmark Summary", show_header=True)
    summary_table.add_column("Benchmark Metric", style="cyan", width=35)
    summary_table.add_column("Score / Result", justify="right", style="bold green", width=20)
    summary_table.add_column("Target Threshold", justify="right", style="dim", width=18)

    summary_table.add_row("Total Test Cases Evaluated", f"{total_queries} queries", "—")
    summary_table.add_row("Context Retrieval Recall @ Top-K", f"{retrieval_recall:.1f}% ({retrieval_hits}/{non_rejected_queries})", ">= 85.0%")
    summary_table.add_row("Factual Groundedness Rate", f"{grounded_rate:.1f}% ({grounded_hits}/{non_rejected_queries})", ">= 90.0%")
    summary_table.add_row("Guardrail Off-Topic Catch Rate", f"{guardrail_accuracy:.1f}% ({guardrail_correct}/{guardrail_tested})", "100.0%")
    summary_table.add_row("Average Pipeline Latency", f"{avg_latency:.2f} seconds", "< 5.0s")

    console.print(summary_table)

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": total_queries,
        "metrics": {
            "retrieval_recall_percent": round(retrieval_recall, 2),
            "groundedness_rate_percent": round(grounded_rate, 2),
            "guardrail_accuracy_percent": round(guardrail_accuracy, 2),
            "avg_latency_seconds": round(avg_latency, 3),
        },
        "detailed_results": results
    }

    # Save to JSON
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation", "evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    console.print(f"📁 Detailed report exported to: [bold green]{json_path}[/bold green]")

    return summary_report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated RAG Benchmark Suite")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test queries (default: run all)")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg.get("app", {}).get("log_level", "INFO"))
    run_benchmark(cfg, limit=args.limit)
