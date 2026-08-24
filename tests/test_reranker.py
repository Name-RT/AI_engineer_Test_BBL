"""
tests/test_reranker.py — Unit Tests for Two-Stage Re-ranking Engine
"""
import pytest
from unittest.mock import MagicMock, patch
from tools.search import KnowledgeBaseSearchTool


@pytest.fixture
def mock_config():
    return {
        "search": {
            "mode": "tfidf",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "top_k": 5,
            "similarity_threshold": 0.01,
            "knowledge_base_path": "knowledge_base.txt",
            "persist_directory": "./chroma_db",
            "collection_name": "rag_bbl_policies",
        },
        "reranking": {
            "enabled": True,
            "model_name": "BAAI/bge-reranker-v2-m3",
            "candidate_top_k": 5,
            "final_top_k": 2,
            "device": "cpu",
        },
        "embedding": {
            "model_name": "intfloat/multilingual-e5-small",
            "device": "cpu",
        }
    }


def test_reranker_disabled_behavior(mock_config):
    """Verify that when reranking is disabled, stage 1 results are returned directly."""
    mock_config["reranking"]["enabled"] = False
    tool = KnowledgeBaseSearchTool(mock_config)
    
    results = tool.search("vacation annual leave", top_k=3)
    assert len(results) <= 3
    assert tool.reranker is None  # Never loaded when disabled


def test_reranker_scoring_reorders_chunks(mock_config):
    """Verify that rerank() applies cross-encoder scores and reorders candidates."""
    tool = KnowledgeBaseSearchTool(mock_config)
    
    # Mock CrossEncoder
    mock_encoder = MagicMock()
    # Return higher score for second candidate
    mock_encoder.predict.return_value = [0.1, 4.5, -2.0]
    tool.reranker = mock_encoder
    
    candidates = [
        {"chunk_id": 1, "content": "Generic policy info", "score": 0.9},
        {"chunk_id": 2, "content": "Exact answer to user question", "score": 0.5},
        {"chunk_id": 3, "content": "Irrelevant appendix text", "score": 0.3},
    ]
    
    reranked = tool.rerank("user question", candidates, top_k=2)
    
    assert len(reranked) == 2
    # Chunk 2 should be ranked #1 because its predict score was 4.5
    assert reranked[0]["chunk_id"] == 2
    assert reranked[0]["score"] > reranked[1]["score"]
    assert "stage1_score" in reranked[0]
    assert reranked[0]["stage1_score"] == 0.5


def test_reranker_raw_logit_score_ordering(mock_config):
    """Verify that raw logit scores from CrossEncoder are directly preserved and sorted."""
    tool = KnowledgeBaseSearchTool(mock_config)
    
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [10.5, -8.2, 0.4]
    tool.reranker = mock_encoder
    
    candidates = [
        {"chunk_id": 1, "content": "Doc 1", "score": 0.8},
        {"chunk_id": 2, "content": "Doc 2", "score": 0.7},
        {"chunk_id": 3, "content": "Doc 3", "score": 0.6},
    ]
    
    reranked = tool.rerank("query", candidates, top_k=3)
    
    # Doc 1 got 10.5, Doc 3 got 0.4, Doc 2 got -8.2
    assert len(reranked) == 3
    assert reranked[0]["chunk_id"] == 1
    assert reranked[0]["score"] == 10.5
    assert reranked[1]["chunk_id"] == 3
    assert reranked[1]["score"] == 0.4
    assert reranked[2]["chunk_id"] == 2
    assert reranked[2]["score"] == -8.2


def test_reranker_graceful_fallback_on_exception(mock_config):
    """Verify that if CrossEncoder raises an error, search falls back to candidates."""
    tool = KnowledgeBaseSearchTool(mock_config)
    
    mock_encoder = MagicMock()
    mock_encoder.predict.side_effect = RuntimeError("GPU Out of Memory")
    tool.reranker = mock_encoder
    
    candidates = [
        {"chunk_id": 1, "content": "Doc 1", "score": 0.9},
        {"chunk_id": 2, "content": "Doc 2", "score": 0.8},
    ]
    
    # Should not raise, but return candidates safely
    results = tool.rerank("query", candidates, top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == 1


def test_two_stage_search_end_to_end(mock_config):
    """Verify full search() execution with Two-Stage retrieval flow."""
    tool = KnowledgeBaseSearchTool(mock_config)
    
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [1.2, 3.4, 0.5, 2.1, -1.0]
    tool.reranker = mock_encoder
    
    results = tool.search("travel expenses reimbursement", top_k=2)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
