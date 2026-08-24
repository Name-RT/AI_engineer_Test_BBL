"""
agents/generator.py — Report Generator Agent Node
"""
import os
import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from .state import AgentState
from config.settings import count_tokens, truncate_to_token_limit

logger = logging.getLogger(__name__)

def create_generator_node(llm, config: Dict[str, Any]):
    """
    Factory creating the Report Generator Agent node for LangGraph.
    
    The Report Generator Agent synthesizes retrieved snippets into a cohesive,
    non-redundant, well-formatted answer with explicit source citations.
    """
    # Load system prompt
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "generator_system.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        logger.error(f"Failed to load generator prompt: {e}")
        system_prompt = "You are a helpful assistant."
        
    def generator_node(state: AgentState) -> dict:
        documents = state.get("retrieved_documents", [])
        query = state.get("query", "")
        
        if not documents:
            logger.info("No documents retrieved, returning empty answer.")
            return {
                "generated_report": "No relevant information found in the knowledge base.",
                "generation_attempts": state.get("generation_attempts", 0) + 1,
            }
            
        context_text = "\n\n".join([f"Snippet from section {doc.get('chunk_id', 'unknown')}:\n{doc['content']}" for doc in documents])
        
        # Token limit check
        max_tokens = config["llm"]["max_tokens"]
        total = count_tokens(system_prompt + context_text + query)
        
        if total > max_tokens * 0.8:
            logger.warning(f"Context too long ({total} tokens), truncating.")
            # Approximation for max context tokens
            max_context = int(max_tokens * 0.7)
            context_text = truncate_to_token_limit(context_text, max_context)
            
        logger.info(f"Token usage: {total} tokens")
        
        # Retry logic: if we failed hallucination check previously, instruct to stick strictly
        extra_instruction = ""
        if state.get("generation_attempts", 0) > 0 and not state.get("is_grounded", True):
            extra_instruction = "\n\nIMPORTANT: Stick strictly to the provided documents. Do not hallucinate."
            
        messages = [
            SystemMessage(content=system_prompt + extra_instruction),
            HumanMessage(content=f"Documents:\n{context_text}\n\nQuestion: {query}")
        ]
        
        try:
            response = llm.invoke(messages)
            report = response.content.strip()
            return {
                "generated_report": report,
                "generation_attempts": state.get("generation_attempts", 0) + 1,
            }
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "generated_report": f"Error generating report: {str(e)}",
                "generation_attempts": state.get("generation_attempts", 0) + 1,
            }
            
    return generator_node

def create_rejection_response_node(config: Dict[str, Any]):
    """
    Factory creating the Rejection Response node for LangGraph.
    
    Generates polite and informative refusal messages when queries are rejected by
    the Input Validator (e.g. security violations, off-topic, empty, or length bounds).
    
    Args:
        config (Dict[str, Any]): Application configuration dictionary.
        
    Returns:
        Callable[[AgentState], dict]: Node function returning formatted final rejection answer.
    """
    def rejection_response_node(state: AgentState) -> dict:
        reason = state.get("rejection_reason", "")
        max_length = config["guardrails"]["max_query_length"]
        
        logger.info(f"Generating rejection response for reason: {reason}")
        
        # 1. Security Violations (OWASP LLM Defense)
        if "jailbreak" in reason.lower() or "prompt injection" in reason.lower():
            message = ("⚠️ ขออภัย ระบบตรวจพบรูปแบบคำสั่งที่ไม่ได้รับอนุญาต (Prompt Injection / Jailbreak) "
                       "กรุณาสอบถามข้อมูลเกี่ยวกับระเบียบนโยบายของธนาคารตามปกติครับ")
        elif "system prompt" in reason.lower():
            message = ("🔒 ขออภัย ข้อมูลคำสั่งภายในระบบ (System Prompt) เป็นความลับของระบบ "
                       "คุณสามารถสอบถามข้อมูลระเบียบนโยบายด้าน HR/IT ขององค์กรได้โดยตรงครับ")
        elif "rate limit" in reason.lower():
            message = ("⏱️ มีการส่งคำถามถี่เกินกำหนด (Rate Limit Exceeded) "
                       "กรุณารอสักครู่แล้วลองส่งคำถามใหม่อีกครั้งครับ")
        elif "profanity" in reason.lower() or "toxic" in reason.lower():
            message = ("🚫 ขออภัย ระบบตรวจพบข้อความที่ไม่สุภาพ กรุณาใช้คำถามที่สุภาพในการสอบถามข้อมูลครับ")
        elif "code" in reason.lower() or "sql" in reason.lower():
            message = ("🛡️ ขออภัย ระบบตรวจพบคำสั่งที่มีความเสี่ยงด้านความปลอดภัย (Code/SQL Injection)")
            
        # 2. Scope & Guardrail Violations
        elif "off-topic" in reason.lower():
            message = ("I'm sorry, but your question appears to be outside the scope "
                       "of our company knowledge base. I can only answer questions "
                       "related to company policies, HR, IT security, travel, expenses, "
                       "leave, and workplace guidelines. Please try rephrasing your "
                       "question or ask about a specific company policy.")
        elif "empty" in reason.lower():
            message = "Please provide a question. Your query cannot be empty."
        elif "too long" in reason.lower():
            message = f"Your query is too long. Please keep it under {max_length} characters."
        elif "too short" in reason.lower():
            message = "Your query is too short. Please provide more detail."
        else:
            message = f"Your query could not be processed. Reason: {reason}"
            
        return {"final_answer": message}
    return rejection_response_node

def create_max_attempts_fallback_node():
    """
    Factory creating the Maximum Attempts Fallback node for LangGraph.
    
    Acts as a safety net: when the Generator fails hallucination verification after
    maximum attempts, returns verified raw document snippets rather than crashing.
    """
    def max_attempts_fallback_node(state: AgentState) -> dict:
        logger.warning("Max generation attempts reached. Output could not be verified.")
        docs = state.get("retrieved_documents", [])
        
        fallback_msg = (
            "I found relevant information but was unable to generate "
            "a fully verified answer. Here are the raw snippets I found:\n\n"
        )
        fallback_msg += "\n\n".join(doc["content"] for doc in docs)
        
        return {
            "final_answer": fallback_msg,
            "error": "Max generation attempts reached. Output could not be verified."
        }
    return max_attempts_fallback_node
