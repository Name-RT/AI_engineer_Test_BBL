"""
agents/state.py — LangGraph State Definition
"""
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict, total=False):
    """
    Centralized LangGraph State Schema.
    
    This immutable data structure is passed through the entire LangGraph directed cyclic graph,
    tracking query evolution, retrieved context, confidence scores, and validation flags.
    """
    # === Input Field ===
    query: str                          # Original user query string
    
    # === Intermediate Processing & Routing Fields ===
    expanded_query: str                 # Query after synonym expansion or LLM rewriting
    is_valid: bool                      # Boolean flag from input validator (True = allowed, False = rejected)
    rejection_reason: str               # Explicit reason if query is rejected (e.g. 'off-topic', 'too short')
    retrieved_documents: List[Dict[str, Any]] # Extracted chunks: [{'chunk_id': int, 'content': str, 'score': float}]
    retrieval_confidence: float         # Top-1 retrieval confidence score (0.0 to 1.0)
    retrieval_attempts: int             # Number of retrieval/rewrite attempts executed
    
    # === Synthesis & Quality Assurance Fields ===
    generated_report: str               # Candidate report drafted by Report Generator Agent
    is_grounded: bool                   # Fact-check verdict (True = verified against source docs, False = hallucinated)
    generation_attempts: int            # Number of generation retry cycles executed
    
    # === Terminal Outputs ===
    final_answer: str                   # Final response formatted for delivery to the end user
    error: str                          # Error details if an unrecoverable failure occurred
