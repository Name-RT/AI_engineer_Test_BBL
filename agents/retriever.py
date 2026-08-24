"""
agents/retriever.py — Data Retriever Agent Node
"""
import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from .state import AgentState

logger = logging.getLogger(__name__)

def create_retriever_node(search_tool, config: Dict[str, Any]):
    """
    Factory creating the Data Retriever Agent node for LangGraph.
    
    The Retriever Agent searches the knowledge base and extracts relevant raw snippets
    without altering or summarizing them, computing a weighted-average relevance score
    across the top-K retrieved chunks.
    """
    # Weights for top-K positions: top-1 contributes most, diminishing for lower ranks.
    # Sum = 1.0 so the output is a weighted average in the same unit as raw logits.
    TOP_K_WEIGHTS = [0.40, 0.25, 0.15, 0.12, 0.08]
    
    def retriever_node(state: AgentState) -> dict:
        query = state.get("expanded_query") or state.get("query", "")
        
        logger.info(f"Retrieving documents for query: {query}")
        results = search_tool.search(query)
        
        if not results:
            score = 0.0
        else:
            # Weighted average of top-K raw logits gives a more robust
            # quality measure than relying on a single top-1 score.
            weights = TOP_K_WEIGHTS[:len(results)]
            weight_sum = sum(weights)
            score = sum(w * float(r["score"]) for w, r in zip(weights, results)) / weight_sum
            
        logger.info(f"Retrieved {len(results)} documents with weighted relevance score {score:.4f}")
        
        return {
            "retrieved_documents": results,
            "retrieval_score": score,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }
    return retriever_node

def create_query_rewriter_node(llm):
    """
    Factory creating the Query Rewriter Agent node for LangGraph.
    
    Triggered when retrieval confidence falls below the threshold. Uses the LLM to
    reformulate the user query with policy-oriented synonyms and improved phrasing.
    """
    def query_rewriter_node(state: AgentState) -> dict:
        original = state.get("query", "")
        previous = state.get("expanded_query", "")
        
        logger.info(f"Rewriting query due to low confidence: {original}")
        
        prompt = (
            "The following search query returned poor results. Rewrite it to be "
            "more specific and use different keywords that might appear in a "
            "company policy document.\n"
            f"Original query: {original}\n"
            f"Previous attempt: {previous}\n"
            "Rewrite the query (output ONLY the new query, nothing else):"
        )
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            new_query = response.content.strip()
            logger.info(f"Rewritten query: {new_query}")
            return {"expanded_query": new_query}
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return {"expanded_query": original} # fallback
            
    return query_rewriter_node
