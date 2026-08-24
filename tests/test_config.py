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
    assert count_tokens("") == 0
    text = "Hello world"
    count = count_tokens(text)
    assert count == 2


def test_truncate_to_token_limit():
    text = "This is a very long sentence that needs to be truncated."
    truncated = truncate_to_token_limit(text, 5)
    assert len(truncated) < len(text)
    assert count_tokens(truncated) <= 5


def test_chat_azure_apim_rate_limit_retry():
    model = ChatAzureAPIM(api_key="test-key", model_name="gpt-5-mini")
    
    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.headers = {"Retry-After": "0.01"}
    
    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {"response": "Recovered after 429 rate limit."}
    
    with patch("requests.post", side_effect=[mock_429, mock_200]) as mock_post, patch("time.sleep") as mock_sleep:
        res = model.invoke([HumanMessage(content="Test retry")])
        assert res.content == "Recovered after 429 rate limit."
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()
