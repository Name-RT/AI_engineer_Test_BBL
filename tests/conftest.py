import pytest
from typing import Any, List, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from tools.search import KnowledgeBaseSearchTool

class MockLLM(BaseChatModel):
    responses: List[str] = Field(default_factory=list)
    current_index: int = Field(default=0)
    
    @property
    def _llm_type(self) -> str:
        return "mock_llm"
        
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any):
        if self.current_index < len(self.responses):
            response = self.responses[self.current_index]
            self.current_index += 1
        else:
            response = self.responses[-1] if self.responses else "Mock response"
        return type("ChatResult", (), {"generations": [type("ChatGeneration", (), {"message": AIMessage(content=response)})()]})()
        
    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> BaseMessage:
        if self.current_index < len(self.responses):
            response = self.responses[self.current_index]
            self.current_index += 1
        else:
            response = self.responses[-1] if self.responses else "Mock response"
        return AIMessage(content=response)

@pytest.fixture
def mock_llm():
    return MockLLM(responses=["yes", "This is a mocked report.", "yes"])

@pytest.fixture
def sample_config():
    return {
        "app": {"log_level": "INFO"},
        "search": {
            "mode": "tfidf",
            "chunk_size": 100,
            "chunk_overlap": 0,
            "top_k": 2,
            "similarity_threshold": 0.01,
            "knowledge_base_path": "tests/test_kb.txt"
        },
        "llm": {
            "temperature": 0.0,
            "max_tokens": 100,
            "max_retries": 2
        },
        "guardrails": {
            "min_query_length": 3,
            "max_query_length": 100,
            "confidence_threshold": 0.1,
            "max_generation_attempts": 2
        }
    }

@pytest.fixture
def sample_knowledge_base(tmp_path):
    kb_path = tmp_path / "test_kb.txt"
    content = "=== Section 1: Travel ===\nEmployees must book travel 30 days in advance.\n\n=== Section 2: WFH ===\nRemote work is allowed 3 days a week."
    kb_path.write_text(content)
    return kb_path

@pytest.fixture
def search_tool(sample_config, sample_knowledge_base, monkeypatch):
    # Override KB path to point to our temp file
    sample_config["search"]["knowledge_base_path"] = str(sample_knowledge_base)
    import os
    orig_join = os.path.join
    def mock_join(*args):
        if len(args) > 1 and "test_kb.txt" in args[-1]:
            return args[-1]
        return orig_join(*args)
    
    import tools.search
    monkeypatch.setattr(tools.search.os.path, "join", mock_join)
    return KnowledgeBaseSearchTool(sample_config)
