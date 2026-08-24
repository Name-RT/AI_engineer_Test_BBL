import pytest
from tools.search import KnowledgeBaseSearchTool

def test_tfidf_search_returns_relevant_chunks(search_tool):
    results = search_tool.search("travel 30 days")
    assert len(results) > 0
    assert "travel 30 days" in results[0]["content"].lower()

def test_tfidf_search_empty_query(search_tool):
    results = search_tool.search("")
    assert results == []
    
    results = search_tool.search("   ")
    assert results == []

def test_tfidf_search_no_match(search_tool):
    search_tool.threshold = 0.5
    results = search_tool.search("quantum physics and black holes")
    assert results == []

def test_synonym_expansion(search_tool):
    expanded = search_tool._expand_synonyms("WFH")
    expanded_lower = expanded.lower()
    assert "remote" in expanded_lower
    assert "work" in expanded_lower

def test_search_respects_top_k(search_tool):
    results = search_tool.search("work", top_k=1)
    assert len(results) <= 1

def test_search_scores_are_sorted(search_tool):
    results = search_tool.search("travel remote")
    assert len(results) >= 2, "Expected at least 2 search results to verify sorting"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

def test_chunking_preserves_content(search_tool):
    # Our test_kb has two chunks
    assert len(search_tool.chunks) == 2
    content_combined = " ".join([c["content"] for c in search_tool.chunks])
    assert "Travel" in content_combined
    assert "Remote" in content_combined

def test_chroma_search_integration(sample_config, sample_knowledge_base, monkeypatch, tmp_path):
    sample_config["search"]["mode"] = "chroma"
    sample_config["search"]["knowledge_base_path"] = str(sample_knowledge_base)
    sample_config["search"]["persist_directory"] = str(tmp_path / "test_chroma_db")
    sample_config["search"]["collection_name"] = "test_collection"
    sample_config["search"]["similarity_threshold"] = 0.0
    
    # Mock embeddings for fast offline testing
    class FastMockEmbeddings:
        def embed_documents(self, texts):
            return [[float(i % 4 + 1) for i in range(8)] for _ in texts]
        def embed_query(self, text):
            return [float(i % 4 + 1) for i in range(8)]
            
    import config.settings
    monkeypatch.setattr(config.settings, "get_embeddings", lambda: FastMockEmbeddings())
    
    tool = KnowledgeBaseSearchTool(sample_config)
    assert tool.collection is not None
    assert tool.collection.count() == 2
    
    results = tool.search("travel", top_k=2)
    assert len(results) > 0
    assert "chunk_id" in results[0]
    assert "content" in results[0]
    assert "score" in results[0]

def test_hybrid_rrf_search_integration(sample_config, sample_knowledge_base, monkeypatch):
    sample_config["search"]["mode"] = "hybrid"
    sample_config["search"]["knowledge_base_path"] = str(sample_knowledge_base)
    sample_config["search"]["similarity_threshold"] = 0.0
    
    class FastMockEmbeddings:
        def embed_documents(self, texts):
            return [[float(i % 4 + 1) for i in range(8)] for _ in texts]
        def embed_query(self, text):
            return [float(i % 4 + 1) for i in range(8)]
            
    import config.settings
    monkeypatch.setattr(config.settings, "get_embeddings", lambda: FastMockEmbeddings())
    
    tool = KnowledgeBaseSearchTool(sample_config)
    assert tool.embed_matrix is not None
    
    results = tool.search("travel policy", top_k=2)
    assert len(results) > 0
    assert "chunk_id" in results[0]
    assert "content" in results[0]
    assert "rrf_score" in results[0]
