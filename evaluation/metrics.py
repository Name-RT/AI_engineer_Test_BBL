"""
evaluation/metrics.py — RAGAS Evaluation Pipeline
"""
import os
import json
import logging
import time
from typing import Dict, Any
from rich.console import Console
from rich.table import Table

# RAGAS specific imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from config.settings import get_llm, get_embeddings
from agents.graph import create_graph

logger = logging.getLogger(__name__)
console = Console()

def run_evaluation(config: Dict[str, Any]) -> Dict[str, float]:
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["evaluation"]["golden_dataset_path"])
    
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load golden dataset: {e}")
        return {}

    logger.info(f"Loaded {len(golden_data)} Q&A pairs for evaluation.")
    
    graph = create_graph(config)
    
    # Prepare RAGAS dataset format
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    console.print("[bold yellow]Running queries for evaluation...[/bold yellow]")
    for item in golden_data:
        question = item["question"]
        expected = item["ground_truth"]
        
        # Skip expected off-topic rejections for RAGAS eval 
        # (RAGAS metrics don't work well on rejected queries)
        if "rejected" in expected.lower():
            continue
            
        initial_state = {
            "query": question,
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
            result = graph.invoke(initial_state, config={"configurable": {"thread_id": str(time.time())}})
            answer = result.get("final_answer", "")
            docs = [d["content"] for d in result.get("retrieved_documents", [])]
            
            if not answer or not docs:
                logger.warning(f"Skipping '{question}' due to missing answer or docs.")
                continue
                
            eval_data["question"].append(question)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append(docs)
            eval_data["ground_truth"].append(expected)
        except Exception as e:
            logger.error(f"Error running pipeline for eval on '{question}': {e}")
            continue

    if not eval_data["question"]:
        logger.error("No valid data points collected for evaluation.")
        return {}

    dataset = Dataset.from_dict(eval_data)
    
    # Configure RAGAS wrappers
    llm = get_llm(temperature=0.0)
    embeddings = get_embeddings()
    
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    
    # Note: Ragas v0.2.x evaluate signature might vary slightly, 
    # but passing llm and embeddings generally works or defaults can be overridden per metric.
    for m in [faithfulness, answer_relevancy, context_precision]:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_emb

    console.print("[bold yellow]Running RAGAS evaluation...[/bold yellow]")
    max_retries = 3
    results = None
    
    for attempt in range(max_retries):
        try:
            results = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=ragas_llm,
                embeddings=ragas_emb
            )
            break
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
            
    if not results:
        logger.error("RAGAS evaluation completely failed after retries.")
        return {}
        
    scores = results.scores if hasattr(results, "scores") else results

    table = Table(title="RAGAS Evaluation Results")
    table.add_column("Metric", justify="left", style="cyan")
    table.add_column("Score", justify="right", style="green")
    
    # Extract mean scores across the dataset
    metric_names = ["faithfulness", "answer_relevancy", "context_precision"]
    final_scores = {}
    
    if isinstance(scores, dict):
        # Output is often a dict-like object with metric names
        for m in metric_names:
            if m in scores:
                table.add_row(m, f"{scores[m]:.4f}")
                final_scores[m] = scores[m]
    else:
        # If it's a pandas dataframe or similar
        try:
            df = results.to_pandas()
            for m in metric_names:
                if m in df.columns:
                    mean_score = df[m].mean()
                    table.add_row(m, f"{mean_score:.4f}")
                    final_scores[m] = mean_score
        except:
            table.add_row("Results", str(scores))
            
    console.print(table)
    return final_scores
