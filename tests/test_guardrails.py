import pytest
from guardrails.input_validator import create_input_validator_node
from guardrails.output_validator import create_output_validator_node

def test_empty_query_rejected(mock_llm, search_tool, sample_config):
    node = create_input_validator_node(mock_llm, search_tool, sample_config)
    result = node({"query": ""})
    assert result["is_valid"] is False
    assert "empty" in result["rejection_reason"].lower()

def test_too_short_query_rejected(mock_llm, search_tool, sample_config):
    node = create_input_validator_node(mock_llm, search_tool, sample_config)
    result = node({"query": "hi"})
    assert result["is_valid"] is False
    assert "too short" in result["rejection_reason"].lower()

def test_too_long_query_rejected(mock_llm, search_tool, sample_config):
    node = create_input_validator_node(mock_llm, search_tool, sample_config)
    result = node({"query": "a" * 150})
    assert result["is_valid"] is False
    assert "too long" in result["rejection_reason"].lower()

def test_valid_query_accepted(mock_llm, search_tool, sample_config):
    # Should pass TF-IDF fast check
    node = create_input_validator_node(mock_llm, search_tool, sample_config)
    result = node({"query": "travel policy"})
    assert result["is_valid"] is True
    assert result["rejection_reason"] == ""

def test_off_topic_detected(mock_llm, search_tool, sample_config):
    # Will fail fast check (low score) and hit LLM which replies "no"
    mock_llm.responses = ["no"]
    node = create_input_validator_node(mock_llm, search_tool, sample_config)
    result = node({"query": "quantum mechanics"})
    assert result["is_valid"] is False
    assert "off-topic" in result["rejection_reason"].lower()

def test_on_topic_accepted(mock_llm, search_tool, sample_config):
    # Force fast check to fail by using obscure words, but LLM says yes
    mock_llm.responses = ["yes"]
    node = create_input_validator_node(mock_llm, search_tool, sample_config)
    result = node({"query": "obscure hr rule"})
    assert result["is_valid"] is True

def test_output_grounded_passes(mock_llm, sample_config):
    mock_llm.responses = ["yes"]
    node = create_output_validator_node(mock_llm, sample_config)
    state = {
        "generated_report": "Employees can travel.",
        "retrieved_documents": [{"content": "Employees can travel."}]
    }
    result = node(state)
    assert result["is_grounded"] is True
    assert result["final_answer"] == "Employees can travel."

def test_output_hallucinated_fails(mock_llm, sample_config):
    mock_llm.responses = ["no"]
    node = create_output_validator_node(mock_llm, sample_config)
    state = {
        "generated_report": "Employees can fly to the moon.",
        "retrieved_documents": [{"content": "Employees can travel."}]
    }
    result = node(state)
    assert result["is_grounded"] is False
    assert result["final_answer"] == ""
