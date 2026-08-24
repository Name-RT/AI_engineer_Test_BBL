import pytest
import os
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import load_config, get_llm, count_tokens, truncate_to_token_limit, ChatAzureAPIM


def test_load_config():
    config = load_config()
    assert "app" in config
    assert "search" in config
    assert "llm" in config


def test_get_llm_invalid_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "invalid_provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm()


def test_get_llm_deepseek(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    llm = get_llm()
    assert llm is not None


def test_get_llm_azure_apim(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure_apim")
    monkeypatch.setenv("AZURE_APIM_KEY", "test-apim-key")
    llm = get_llm()
    assert isinstance(llm, ChatAzureAPIM)
    assert llm.model_name == "gpt-5-mini"


def test_chat_azure_apim_mock_invoke():
    model = ChatAzureAPIM(api_key="test-key", model_name="gpt-5-mini")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "This is a response from Azure APIM."}
    
    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = model.invoke([SystemMessage(content="You are helpful."), HumanMessage(content="Hello")])
        assert res.content == "This is a response from Azure APIM."
        mock_post.assert_called_once()


def test_count_tokens():
    text = "Hello world"
    count = count_tokens(text)
    assert count > 0


def test_truncate_to_token_limit():
    text = "This is a very long sentence that needs to be truncated."
    truncated = truncate_to_token_limit(text, 5)
    assert len(truncated) < len(text)
    assert count_tokens(truncated) <= 5
