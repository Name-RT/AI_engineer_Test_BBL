"""
guardrails/output_validator.py — Hallucination Check & Output Validation
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

def create_output_validator_node(llm, config: Dict[str, Any]):
    """
    Factory creating the Output Validator node for LangGraph.
    
    Verifies factual groundedness of the generated report against retrieved chunks
    and ensures sensitive information is sanitized before final output.
    
    Args:
        llm: Language model instance used as evaluator judge.
        config (Dict[str, Any]): Application configuration dictionary.
        
    Returns:
        Callable[[AgentState], dict]: Node function returning groundedness status and final answer.
    """
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "output_validator_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        logger.error(f"Failed to load output validator prompt: {e}")
        system_prompt = "Is every claim in the report supported by the source documents? Answer ONLY 'yes' or 'no'."

    security_shield = SecurityShield(config)

    def output_validator_node(state: "AgentState") -> dict:
        """Verifies report groundedness against source documents and sanitizes output."""
        report = state.get("generated_report", "").strip()
        docs = state.get("retrieved_documents", [])
        
        # Log token count of output
        logger.info(f"Output tokens: {count_tokens(report)}")
        
        # Step 1: Empty check
        if not report or "Error generating report" in report:
            logger.warning("Empty or error report, marking as ungrounded.")
            return {"is_grounded": False, "final_answer": ""}
            
        if not docs:
            # If no docs were retrieved but a report was generated, it might be hallucinated
            # or it might be a valid "No information found" response.
            if "No relevant information found" in report:
                sanitized_report = security_shield.sanitize_output(report)
                return {"is_grounded": True, "final_answer": sanitized_report}
            return {"is_grounded": False, "final_answer": ""}

        # Step 2: Groundedness Check (LLM-based)
        context_text = "\n\n".join([f"Document Chunk #{i+1}:\n{doc['content']}" for i, doc in enumerate(docs)])
        
        user_message_content = (
            f"Source Documents:\n{context_text}\n\n"
            f"Generated Report to Verify:\n{report}\n\n"
            "Question: Is the generated report faithful to and supported by the source documents? "
            "Answer ONLY 'yes' or 'no'."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message_content)
        ]
        
        try:
            logger.info("Running hallucination check...")
            response = llm.invoke(messages)
            answer = response.content.strip().lower()
            logger.info(f"Hallucination check response: {answer}")
            
            # Robust affirmative checking
            if "yes" in answer or answer.startswith("y") or "faithful" in answer or "supported" in answer:
                logger.info("Report is grounded.")
                sanitized_report = security_shield.sanitize_output(report)
                return {"is_grounded": True, "final_answer": sanitized_report}
            else:
                logger.warning(f"Report contains unsupported claims (hallucination detected). LLM verdict: {answer}")
                return {"is_grounded": False, "final_answer": ""}
        except Exception as e:
            logger.error(f"Error during hallucination check: {e}")
            return {"is_grounded": False, "final_answer": ""}

    return output_validator_node
