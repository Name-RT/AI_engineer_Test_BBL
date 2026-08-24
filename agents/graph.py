"""
agents/graph.py — LangGraph StateGraph Definition
"""
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config.settings import get_llm
from tools.search import KnowledgeBaseSearchTool
from .state import AgentState
from .retriever import create_retriever_node, create_query_rewriter_node
from .generator import create_generator_node, create_rejection_response_node, create_max_attempts_fallback_node

from guardrails.input_validator import create_input_validator_node
from guardrails.output_validator import create_output_validator_node

logger = logging.getLogger(__name__)

def create_graph(config: Dict[str, Any]):
    """
    Constructs and compiles the complete LangGraph StateGraph pipeline.
    
    Architecture & Flow:
    1. Input Validation: Screens query (fast TF-IDF check -> LLM classification).
       - If invalid -> Routes to `rejection_response` -> END
       - If valid -> Routes to `retriever`
    2. Data Retrieval: Searches Knowledge Base using TF-IDF / Hybrid vector similarity.
       - If confidence >= threshold -> Routes to `generator`
       - If confidence < threshold and attempts < max -> Routes to `query_rewriter` -> `retriever`
       - If max attempts reached -> Routes to `generator`
    3. Report Generation: Synthesizes retrieved chunks into a cited, non-redundant markdown report.
    4. Output Validation: Verifies factual groundedness against retrieved chunks.
       - If grounded -> END
       - If ungrounded and attempts < max -> Retries `generator`
       - If max attempts reached -> Routes to `max_attempts_fallback` -> END
       
    Args:
        config (Dict[str, Any]): Application configuration dictionary from config.yaml.
        
    Returns:
        CompiledStateGraph: The compiled LangGraph application with MemorySaver checkpointer.
    """
    # 1. Initialize dependencies once (Singleton-like lifecycle per graph instance)
    llm = get_llm(temperature=config["llm"]["temperature"])
    search_tool = KnowledgeBaseSearchTool(config)
    
    # 2. Create node functions using closure factory pattern
    input_validator_node = create_input_validator_node(llm, search_tool, config)
    retriever_node = create_retriever_node(search_tool, config)
    query_rewriter_node = create_query_rewriter_node(llm)
    generator_node = create_generator_node(llm, config)
    output_validator_node = create_output_validator_node(llm, config)
    rejection_response_node = create_rejection_response_node(config)
    max_attempts_fallback_node = create_max_attempts_fallback_node()
    
    # 3. Create Graph
    graph = StateGraph(AgentState)
    
    # Add Nodes
    graph.add_node("input_validator", input_validator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("generator", generator_node)
    graph.add_node("output_validator", output_validator_node)
    graph.add_node("rejection_response", rejection_response_node)
    graph.add_node("max_attempts_fallback", max_attempts_fallback_node)
    
    # 4. Routing logic
    def route_input(state: AgentState) -> str:
        if state.get("is_valid", True):
            return "valid"
        return "invalid"
        
    def route_confidence(state: AgentState) -> str:
        threshold = config["guardrails"]["confidence_threshold"]
        max_attempts = config["llm"]["max_retries"]
        confidence = state.get("retrieval_confidence", 0.0)
        
        if confidence >= threshold:
            return "generator"
        elif state.get("retrieval_attempts", 0) < max_attempts:
            return "query_rewriter"
        elif confidence > 0.20:
            return "generator"
        else:
            return "rejection_response"
            
    def route_output_validation(state: AgentState) -> str:
        if state.get("is_grounded", False):
            return "end"
        elif state.get("generation_attempts", 0) < config["guardrails"]["max_generation_attempts"]:
            return "generator"
        else:
            return "max_attempts_fallback"
            
    # Add Edges
    graph.set_entry_point("input_validator")
    
    graph.add_conditional_edges(
        "input_validator",
        route_input,
        {
            "valid": "retriever",
            "invalid": "rejection_response"
        }
    )
    
    graph.add_edge("rejection_response", END)
    
    graph.add_conditional_edges(
        "retriever",
        route_confidence,
        {
            "generator": "generator",
            "query_rewriter": "query_rewriter",
            "rejection_response": "rejection_response"
        }
    )
    
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("generator", "output_validator")
    
    graph.add_conditional_edges(
        "output_validator",
        route_output_validation,
        {
            "end": END,
            "generator": "generator",
            "max_attempts_fallback": "max_attempts_fallback"
        }
    )
    
    graph.add_edge("max_attempts_fallback", END)
    
    # 5. Compile with MemorySaver
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
