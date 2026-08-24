import pytest
import time
from agents.graph import create_graph
import config.settings

@pytest.fixture
def test_graph(mock_llm, sample_config, search_tool, monkeypatch):
    import agents.graph
    monkeypatch.setattr(agents.graph, "get_llm", lambda *args, **kwargs: mock_llm)
    monkeypatch.setattr(agents.graph, "KnowledgeBaseSearchTool", lambda config: search_tool)
    
    return create_graph(sample_config)

def test_end_to_end_valid_query(test_graph, mock_llm):
    # Setup mock responses:
    # 1. input validation LLM (skipped if fast check passes, travel policy will pass)
    # 2. generator LLM
    # 3. output validation LLM
    mock_llm.responses = ["This is a test report about travel.", "yes"]
    
    state = {
        "query": "travel policy",
        "retrieval_attempts": 0,
        "generation_attempts": 0
    }
    
    result = test_graph.invoke(state, config={"configurable": {"thread_id": "test1"}})
    
    assert result["is_valid"] is True
    assert len(result["retrieved_documents"]) > 0
    assert result["is_grounded"] is True
    assert result["final_answer"] == "This is a test report about travel."

def test_end_to_end_off_topic(test_graph, mock_llm):
    # LLM responses: 1. input validation -> 'no'
    mock_llm.responses = ["no"]
    
    state = {
        "query": "quantum computing",
        "retrieval_attempts": 0,
        "generation_attempts": 0
    }
    
    result = test_graph.invoke(state, config={"configurable": {"thread_id": "test2"}})
    
    assert result["is_valid"] is False
    assert "off-topic" in result["rejection_reason"].lower() or "off-topic" in result["final_answer"].lower() or "scope" in result["final_answer"].lower()

def test_end_to_end_no_results(test_graph, mock_llm):
    # Change threshold so no results match
    import tools.search
    
    state = {
        "query": "xyzabc_no_match",
        "retrieval_attempts": 0,
        "generation_attempts": 0
    }
    
    # input validation LLM -> yes (let it pass to retriever)
    # query rewrite LLM -> "xyzabc not in kb rewritten"
    # generator LLM -> empty or standard response
    # output validator LLM -> "yes"
    mock_llm.responses = ["yes", "qwerty_no_match", "No relevant information found", "yes"]
    
    result = test_graph.invoke(state, config={"configurable": {"thread_id": "test3"}})
    
    assert len(result["retrieved_documents"]) == 0
    # It might retry once because of low confidence
    assert result["retrieval_attempts"] > 0
    assert "No relevant information" in result["final_answer"]

def test_query_rewrite_on_low_confidence(test_graph, mock_llm):
    mock_llm.responses = ["yes", "better query about travel", "Good report.", "yes"]
    
    state = {
        "query": "trvl",
        "retrieval_attempts": 0,
        "generation_attempts": 0
    }
    
    result = test_graph.invoke(state, config={"configurable": {"thread_id": "test4"}})
    
    assert result.get("expanded_query", "") != ""
    assert result["retrieval_attempts"] > 1

def test_max_attempts_fallback_execution(test_graph, mock_llm):
    # Setup mock to fail hallucination check repeatedly (answering 'no')
    mock_llm.responses = ["Hallucinated report 1", "no", "Hallucinated report 2", "no", "Hallucinated report 3", "no"]
    
    state = {
        "query": "travel policy",
        "retrieval_attempts": 0,
        "generation_attempts": 0
    }
    
    result = test_graph.invoke(state, config={"configurable": {"thread_id": "test5"}})
    
    assert result["generation_attempts"] >= 2
    assert "raw snippets" in result["final_answer"].lower() or "unable to generate" in result["final_answer"].lower()
