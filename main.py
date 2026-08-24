"""
main.py — CLI Entry Point
"""
import sys
import argparse
import logging
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import load_config, setup_logging
from agents.graph import create_graph

console = Console()
logger = logging.getLogger(__name__)

def run_query(graph, query: str) -> str:
    start_time = time.time()
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
    
    try:
        # Use thread_id to allow checkpointing if needed
        result = graph.invoke(initial_state, config={"configurable": {"thread_id": str(time.time())}})
        
        elapsed = time.time() - start_time
        final_answer = result.get("final_answer", "")
        docs = result.get("retrieved_documents", [])
        confidence = result.get("retrieval_confidence", 0.0)
        is_grounded = result.get("is_grounded", False)
        
        console.print(Panel(f"[bold cyan]🔍 Query:[/bold cyan] {query}"))
        
        if not final_answer:
            if result.get("error"):
                final_answer = result["error"]
            elif not result.get("is_valid", True):
                final_answer = f"Rejected: {result.get('rejection_reason', 'Unknown')}"
        
        info_str = f"📄 Retrieved: {len(docs)} chunks\n🎯 Confidence: {confidence:.2f}"
        console.print(Panel(info_str))
        
        console.print(Panel(f"[bold green]📋 Answer:[/bold green]\n{final_answer}"))
        
        grounded_str = "Yes" if is_grounded else "No (or N/A)"
        stats_str = f"✅ Grounded: {grounded_str}\n⏱️  Time: {elapsed:.2f}s"
        console.print(Panel(stats_str))
        
        return final_answer

    except Exception as e:
        logger.error(f"Error running query: {e}")
        console.print(Panel(f"[bold red]Error:[/bold red] {e}"))
        return str(e)

def run_demo(graph):
    queries = [
        "What is the policy on international travel?",
        "How does the remote work policy work?",
        "What are the rules for expense reimbursement?",
        "What is the weather today?", # off-topic -> should reject
        "Tell me about password requirements and IT security"
    ]
    
    for q in queries:
        console.rule(f"[bold yellow]Testing Query[/bold yellow]")
        run_query(graph, q)
        time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="RAG_BBL Agentic System")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--demo", action="store_true", help="Run predefined demo queries")
    parser.add_argument("--evaluate", action="store_true", help="Run RAGAS evaluation")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config["app"]["log_level"])
    
    logger.info("Initializing Agentic RAG System...")
    graph = create_graph(config)
    
    if args.evaluate:
        from evaluate import run_benchmark
        run_benchmark(config)
        return
        
    if args.demo:
        run_demo(graph)
        return
        
    if args.query:
        run_query(graph, args.query)
        return
        
    # Interactive mode
    console.print(Panel.fit("[bold green]Welcome to RAG_BBL CLI[/bold green]\nType 'exit' or 'quit' to stop."))
    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
            run_query(graph, user_input)
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
