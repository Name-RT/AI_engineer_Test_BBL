"""
config/settings.py — Configuration loader & LLM Factory
Supported Providers: DeepSeek & Azure APIM Gateway (BBL Candidate Endpoint)
"""
import os
import sys
import time
import yaml
import logging
import warnings
import requests
import tiktoken
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Suppress harmless HuggingFace Hub unauthenticated notices
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Load .env variables (without logging sensitive values)
load_dotenv()


class ChatAzureAPIM(SimpleChatModel):
    """
    Custom LangChain Chat Model for Azure API Management (APIM) LLM Gateway.
    
    Compatible with:
    Endpoint: https://apimsdbxcandidate01.azure-api.net/llm/responses
    Header: api-key: <key>
    Payload: {"model": "gpt-5-mini", "input": "..."}
    """
    endpoint: str = "https://apimsdbxcandidate01.azure-api.net/llm/responses"
    api_key: str = ""
    model_name: str = "gpt-5-mini"
    temperature: float = 0.3
    timeout: float = 60.0

    @property
    def _llm_type(self) -> str:
        return "azure_apim"

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Executes a call to the Azure APIM LLM gateway by aggregating messages into an input prompt.
        """
        if not self.api_key:
            raise ValueError(
                "AZURE_APIM_KEY is not set. Please provide your API key in .env (AZURE_APIM_KEY=...)"
            )

        # Format chat history into a coherent prompt string
        formatted_dialogue = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_dialogue.append("[System Instruction]\n" + str(msg.content))
            elif isinstance(msg, HumanMessage):
                formatted_dialogue.append("[User]\n" + str(msg.content))
            elif isinstance(msg, AIMessage):
                formatted_dialogue.append("[Assistant]\n" + str(msg.content))
            else:
                formatted_dialogue.append(f"[{msg.type}]\n" + str(msg.content))

        input_text = "\n\n".join(formatted_dialogue)

        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "input": input_text
        }

        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                # Handle 429 Too Many Requests / TPM Quota Throttling (1000 tokens/min limit)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        sleep_time = float(retry_after) if retry_after else (backoff * (2 ** attempt))
                    except (ValueError, TypeError):
                        sleep_time = backoff * (2 ** attempt)

                    sleep_time = min(max(sleep_time, 2.0), 65.0)
                    logging.warning(
                        f"Azure APIM 429 Rate Limit exceeded (1000 TPM limit). "
                        f"Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(sleep_time)
                    continue

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Azure APIM Gateway returned HTTP {response.status_code}: {response.text}"
                    )

                data = response.json()
                parsed_text = self._extract_text_from_apim_response(data)
                if parsed_text:
                    return parsed_text
                return str(data)

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Network error calling Azure APIM: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    continue
                logging.error(f"Network error calling Azure APIM at {self.endpoint}: {e}")
                raise RuntimeError(f"Azure APIM Connection Error: {e}")

        raise RuntimeError("Azure APIM failed after maximum rate limit retries.")

    @staticmethod
    def _extract_text_from_apim_response(data: Any) -> str:
        """
        Extracts clean text from diverse API schemas including OpenAI / Responses API,
        skipping raw encrypted reasoning blocks.
        """
        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            # 1. Direct text fields
            for field in ["output_text", "text", "content"]:
                if field in data and isinstance(data[field], str) and data[field].strip():
                    return data[field].strip()

            # 2. Nested output / response fields
            for field in ["output", "response", "messages", "results"]:
                if field in data:
                    extracted = ChatAzureAPIM._extract_text_from_apim_response(data[field])
                    if extracted:
                        return extracted

            # 3. Standard OpenAI choices format
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    if "message" in choice:
                        return ChatAzureAPIM._extract_text_from_apim_response(choice["message"])
                    elif "text" in choice and isinstance(choice["text"], str):
                        return choice["text"].strip()

            # 4. Content array format
            if "content" in data:
                if isinstance(data["content"], list):
                    texts = []
                    for item in data["content"]:
                        if isinstance(item, dict):
                            if item.get("type") == "text" and "text" in item:
                                texts.append(str(item["text"]))
                            elif "text" in item:
                                texts.append(str(item["text"]))
                        elif isinstance(item, str):
                            texts.append(item)
                    if texts:
                        return "\n".join(texts).strip()
                elif isinstance(data["content"], str):
                    return data["content"].strip()

            # Fallback for single text value
            for val in data.values():
                if isinstance(val, str) and len(val.strip()) > 0:
                    return val.strip()

        elif isinstance(data, list):
            text_parts = []
            for item in data:
                if isinstance(item, dict):
                    # Skip internal/encrypted reasoning chunks
                    item_type = item.get("type", "")
                    if item_type == "reasoning":
                        continue

                    # Extract message text
                    if item_type == "message" or "content" in item:
                        content = item.get("content")
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and "text" in c:
                                    text_parts.append(str(c["text"]))
                                elif isinstance(c, str):
                                    text_parts.append(c)
                        elif isinstance(content, str):
                            text_parts.append(content)
                    elif "text" in item and isinstance(item["text"], str):
                        text_parts.append(item["text"])
                    elif "output" in item:
                        text_parts.append(ChatAzureAPIM._extract_text_from_apim_response(item["output"]))
                elif isinstance(item, str):
                    text_parts.append(item)

            if text_parts:
                return "\n".join([t for t in text_parts if t.strip()]).strip()

        return ""


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config.yaml: {e}")
        raise


def get_llm(temperature: float = None) -> BaseChatModel:
    """
    Get LLM instance based on provider in .env.
    Supported Providers:
    1. 'deepseek' — ChatDeepSeek (official API)
    2. 'azure_apim' / 'azure' — ChatAzureAPIM (BBL Candidate APIM Gateway)
    """
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower().strip()
    config = load_config()
    temp = temperature if temperature is not None else config["llm"]["temperature"]
    max_tokens = config["llm"]["max_tokens"]
    max_retries = config["llm"]["max_retries"]

    try:
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            return ChatDeepSeek(
                model=model,
                api_key=api_key,
                temperature=temp,
                max_tokens=max_tokens,
                max_retries=max_retries
            )

        elif provider in ["azure_apim", "azure", "apim"]:
            endpoint = os.getenv(
                "AZURE_APIM_ENDPOINT",
                "https://apimsdbxcandidate01.azure-api.net/llm/responses"
            )
            api_key = os.getenv("AZURE_APIM_KEY", os.getenv("AZURE_API_KEY", ""))
            model = os.getenv("AZURE_APIM_MODEL", os.getenv("AZURE_DEPLOYMENT", "gpt-5-mini"))
            return ChatAzureAPIM(
                endpoint=endpoint,
                api_key=api_key,
                model_name=model,
                temperature=temp
            )

        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: '{provider}'. Supported providers are: 'deepseek', 'azure_apim'."
            )

    except Exception as e:
        logging.error(f"Error initializing LLM (provider='{provider}'): {e}")
        raise


def get_embeddings() -> Embeddings:
    """Return HuggingFaceEmbeddings using config model name."""
    config = load_config()
    model_name = config["embedding"]["model_name"]
    model_kwargs = {'device': config["embedding"]["device"]}
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
    except Exception as e:
        logging.error(f"Error initializing embeddings: {e}")
        raise


def setup_logging(level: str) -> None:
    """Configure logging module."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count number of tokens in text."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate text to max_tokens."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])
