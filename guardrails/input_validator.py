"""
guardrails/input_validator.py — Input Validation & Off-topic Detection
"""
import os
import logging
from typing import Dict, Any, TYPE_CHECKING
from langchain_core.messages import SystemMessage, HumanMessage
if TYPE_CHECKING:
    from agents.state import AgentState
from config.settings import count_tokens
from .security_shield import SecurityShield

logger = logging.getLogger(__name__)

def create_input_validator_node(llm, search_tool, config: Dict[str, Any]):
    """
    Factory creating the Input Validator node for LangGraph.
    
    Screens user query across security, length, token limits, and off-topic checks:
    1. Security Shield: Checks for jailbreaks, prompt leakage, code injection, and masks PII.
    2. Length & Tokens: Validates non-empty query within allowed token and character bounds.
    3. Off-Topic Filtering: Two-stage check (fast TF-IDF similarity -> LLM binary classifier).
    
    Args:
        llm: Language model instance for fallback off-topic classification.
        search_tool: KnowledgeBaseSearchTool instance for fast lexical similarity check.
        config (Dict[str, Any]): Application configuration dictionary.
        
    Returns:
        Callable[[AgentState], dict]: Node function returning validation status and sanitized query.
    """
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "input_validator_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        logger.error(f"Failed to load input validator prompt: {e}")
        system_prompt = "Is this query related to company policies? Answer ONLY 'yes' or 'no'."

    security_shield = SecurityShield(config)

    def input_validator_node(state: "AgentState") -> dict:
        """Processes the query in the state through all validation filters."""
        raw_query = state.get("query", "").strip()
        min_length = config["guardrails"]["min_query_length"]
        max_length = config["guardrails"]["max_query_length"]
        
        # Step 0: Foundational Security Shield (Rate Limit, Jailbreak, Prompt Leakage, Code Injection, PII Masking)
        sec_res = security_shield.validate_and_sanitize_input(raw_query)
        if not sec_res["is_safe"]:
            logger.warning(f"Security Shield rejection: {sec_res['rejection_reason']}")
            return {
                "is_valid": False,
                "rejection_reason": sec_res["rejection_reason"]
            }

        # Use PII-sanitized query
        query = sec_res["sanitized_query"]
        
        # Step 1: Basic Length & Token Validation
        if not query:
            logger.warning("Empty query rejected.")
            return {"is_valid": False, "rejection_reason": "Query cannot be empty."}
            
        if len(query) < min_length:
            logger.warning("Query too short rejected.")
            return {"is_valid": False, "rejection_reason": f"Query too short (min {min_length} chars)."}
            
        if count_tokens(query) > 500 or len(query) > max_length:
            logger.warning("Query too long rejected.")
            return {"is_valid": False, "rejection_reason": f"Query too long, max {max_length} characters."}

        # Step 2: Off-topic Detection (Fast Check via TF-IDF)
        results = search_tool.search(query, top_k=1)
        fast_check_passed = False
        if results and results[0]["score"] > 0.05: # very low threshold
            fast_check_passed = True
            
        if fast_check_passed:
            logger.info("Passed fast TF-IDF off-topic check.")
            return {"is_valid": True, "rejection_reason": "", "query": query}

        # Step 3: Off-topic Detection (LLM-based)
        logger.info("Fast check failed or uncertain, falling back to LLM off-topic check.")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ]
        
        try:
            response = llm.invoke(messages)
            answer = response.content.strip().lower()
            if "yes" in answer:
                return {"is_valid": True, "rejection_reason": "", "query": query}
            else:
                logger.warning("LLM detected off-topic query.")
                return {"is_valid": False, "rejection_reason": "off-topic"}
        except Exception as e:
            logger.error(f"Error during LLM validation: {e}")
            # Fail open if LLM fails
            return {"is_valid": True, "rejection_reason": "", "query": query}

    return input_validator_node
