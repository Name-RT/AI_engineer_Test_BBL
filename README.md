# AI ENGINEER TEST | AI Policy Assistant

[![CI / Automated Testing](https://github.com/bangkokbank/rag-bbl/actions/workflows/ci.yml/badge.svg)](https://github.com/bangkokbank/rag-bbl/actions/workflows/ci.yml)
[![Tests Passing](https://img.shields.io/badge/tests-58%2F58%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen.svg)](tests/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Docker Ready](https://img.shields.io/badge/docker-ready-2496ED.svg)](docker-compose.yml)

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/LangGraph-FF9900?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/ChromaDB-FC521F?style=for-the-badge&logo=chroma&logoColor=white" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/BGE_Reranker_v2_M3-0055FF?style=for-the-badge&logo=huggingface&logoColor=white" alt="BGE-Reranker-M3" />
  <img src="https://img.shields.io/badge/DeepSeek-4D6BFE?style=for-the-badge&logo=deepseek&logoColor=white" alt="DeepSeek" />
  <img src="https://img.shields.io/badge/Azure_APIM-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white" alt="Azure APIM" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" alt="GitHub Actions" />
</p>

A multi-agent RAG system for querying and synthesizing company policies (HR, IT Security, Finance, Operations). Built with **LangGraph**, **ChromaDB** vector search, **Two-Stage Cross-Encoder Re-ranking**, and **Azure APIM (GPT-5-Mini) / DeepSeek** backend. Supports Thai and English.

---

## System Architecture

<p align="center">
  <a href="architecture_flowchart.drawio">
    <img src="architecture_flowchart.svg" alt="AI Policy Assistant RAG Flowchart" width="100%" />
  </a>
  <br/>
  <em>RAG Pipeline Flowchart — Editable via <code>architecture_flowchart.drawio</code> in <a href="https://app.diagrams.net">draw.io</a></em>
</p>

---

## Features

### Multi-Agent Pipeline (LangGraph)
- **Data Retriever** — pulls policy snippets from `knowledge_base.txt` with ChromaDB / Hybrid search and calculates weighted-average relevance scores.
- **Report Generator** — produces a structured 4-section answer with aggregated policy citations at the bottom. Strictly preserves the question's language (English queries $\rightarrow$ English headers & content; Thai queries $\rightarrow$ Thai headers & content).
- **Query Rewriter** — rewrites the query and retries when retrieval relevance score falls below threshold (`< 0.15`).
- **Fallback** — if hallucination checks keep failing, returns the raw verified snippets instead of guessing.

### Two-Stage Retrieval & Noise Filtering (`tools/search.py`)
- **Stage 1 (recall):** Candidate retrieval (Top-10) using ChromaDB dense vectors (`multilingual-e5-small`) + TF-IDF with Reciprocal Rank Fusion.
- **Stage 2 (precision & noise filtering):** Cross-Encoder re-ranking with `BAAI/bge-reranker-v2-m3` featuring Level 1 chunk noise filtering (`min_chunk_score >= 0.0`), delivering Top-5 refined chunks sorted by raw relevance logit.
- **Auto-Ingestion:** Automatically syncs and persists all 32 chunks into ChromaDB (`./chroma_db/`) on startup.
- **Synonym mapping** for colloquial terms (*WFH* → *remote work*, *ลาพักร้อน* → *annual leave*, etc.)

### Azure APIM & Rate Limit Resilience (`config/settings.py`)
- **OpenAI Responses API Integration:** Native parser for APIM `/llm/responses` endpoint, extracting text content while filtering raw encrypted reasoning blocks.
- **1,000 TPM Exponential Backoff:** Automatic HTTP 429 detection and retry handling honoring `Retry-After` headers.

### Security (OWASP LLM Top 10)
- **Jailbreak guard** — catches adversarial prompts like *"Ignore previous instructions"*, *"DAN Mode"*, *"ลืมกฎทั้งหมด"*.
- **Prompt leakage defense** — blocks attempts to extract system prompts or internal config.
- **PII redaction** — masks Thai National IDs, credit card numbers, phone numbers, emails, and API keys on both input and output.
- **Injection filtering** — strips `<script>` tags, SQL injection (`DROP TABLE`, `UNION SELECT`), and shell payloads.
- **Rate limiting** — sliding window, 30 requests/minute per session.
- **Profanity filter** — rejects abusive input in Thai and English.

### Interfaces
- **Web UI** (`gradio_app.py`) — token streaming with TTFT < 0.1s, real-time node stage progression, dark theme, input locking during processing, live query stop/cancellation button, suggestion pills, confidence gauge, and collapsible sources. Runs on port 7861.
- **CLI** (`main.py`) — interactive REPL with Rich formatting, or pass `--query` for one-shot use.

### Evaluation & CI
- **Benchmark suite** (`evaluate.py`) — runs against `golden_dataset.json`, reports recall, groundedness, guardrail precision, and latency.
- **Docker** — `Dockerfile` + `docker-compose.yml` with healthchecks.
- **GitHub Actions** — runs tests and coverage on every push.

---

## Tech Stack

| Layer | Stack | Notes |
|:---|:---|:---|
| **Framework** | ![LangGraph](https://img.shields.io/badge/LangGraph-FF9900?style=flat-square&logo=langchain&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white) | Cyclic StateGraph with checkpointer |
| **LLM** | ![DeepSeek](https://img.shields.io/badge/DeepSeek-4D6BFE?style=flat-square&logo=deepseek&logoColor=white) ![Azure APIM](https://img.shields.io/badge/Azure_APIM-0078D4?style=flat-square&logo=microsoftazure&logoColor=white) | DeepSeek API or Azure APIM Gateway (BBL endpoint) |
| **Vector DB** | ![ChromaDB](https://img.shields.io/badge/ChromaDB-FC521F?style=flat-square&logo=chroma&logoColor=white) ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black) | Cosine similarity with `multilingual-e5-small` |
| **Re-ranking** | ![BGE Reranker](https://img.shields.io/badge/BGE_Reranker-v2_M3-0055FF?style=flat-square&logo=huggingface&logoColor=white) | `BAAI/bge-reranker-v2-m3` Cross-Encoder |
| **Retrieval** | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | TF-IDF + dense vectors + RRF |
| **Security** | ![OWASP](https://img.shields.io/badge/OWASP-LLM_Top_10-000000?style=flat-square&logo=owasp&logoColor=white) | PII masking, jailbreak guard, rate limiter |
| **Testing** | ![Pytest](https://img.shields.io/badge/Pytest-57_Tests-0A9EDC?style=flat-square&logo=pytest&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=flat-square&logo=githubactions&logoColor=white) | 57 tests, ~80% coverage |
| **UI / Deploy** | ![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=flat-square&logo=gradio&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) | Dark theme UI + Docker Compose |

---

## Getting Started

### Docker (recommended)

```bash
git clone https://github.com/Name-RT/AI_engineer_Test_BBL.git
cd AI_engineer_Test_BBL

cp .env.example .env
# fill in your API keys

docker compose up -d
# open http://localhost:7861
```

### Local setup

```bash
git clone https://github.com/Name-RT/AI_engineer_Test_BBL.git
cd AI_engineer_Test_BBL

conda create -n RAG_BBL python=3.11 -y
conda activate RAG_BBL
pip install -r requirements.txt

cp .env.example .env
# set DEEPSEEK_API_KEY or AZURE_APIM_KEY
```

```bash
# Web UI
python gradio_app.py

# CLI (interactive)
python main.py

# CLI (single query)
python main.py --query "What is the policy on international travel?"

# Demo pipeline
python main.py --demo
```

---

## Benchmark Results

```bash
python evaluate.py
python evaluate.py --limit 5
```

| Metric | Score | Target |
|:---|:---:|:---:|
| Context Recall @ Top-K | 100.0% (20/20) | >= 85% |
| Factual Groundedness | 100.0% (20/20) | >= 90% |
| Guardrail Catch Rate | 100.0% | 100% |
| Avg. Latency | ~6.8s | < 10s |

Full results exported to [`evaluation/evaluation_report.json`](evaluation/evaluation_report.json).

---

## Tests

```bash
python -m pytest tests/ -v --cov=agents --cov=tools --cov=guardrails --cov=config --cov-report=term-missing --tb=short
```

**58/58 passing:**

| File | Count | Covers |
|:---|:---:|:---|
| `test_config.py` | 8 | Config loading, DeepSeek/Azure APIM provider switching, 429 retry, token utils |
| `test_reranker.py` | 5 | Two-stage re-ranking, sigmoid scores, fallback on error |
| `test_guardrails.py` | 8 | Query length checks, off-topic detection, hallucination guard |
| `test_integration.py` | 4 | End-to-end StateGraph flow, query rewrite loops |
| `test_search.py` | 8 | ChromaDB search, TF-IDF, synonym expansion, Top-K ordering |
| `test_security_guardrails.py` | 25 | PII masking, jailbreak, prompt leakage, injection, rate limiting |

See [`TESTING.md`](TESTING.md) for detailed documentation on all test cases and scenarios.

---

## Project Structure

```text
AI_engineer_Test_BBL/
├── main.py                             # CLI entrypoint
├── gradio_app.py                       # Web UI (Gradio, port 7861)
├── evaluate.py                         # Benchmark runner
├── config.yaml                         # All settings (thresholds, models, reranker, security)
├── knowledge_base.txt                  # Policy document (15 sections)
├── TESTING.md                          # Test suite documentation (57 test cases)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example                        # API key template (DeepSeek / Azure APIM)
│
├── agents/
│   ├── state.py                        # AgentState TypedDict
│   ├── retriever.py                    # Retriever & query rewriter nodes
│   ├── generator.py                    # Generator & fallback nodes
│   └── graph.py                        # StateGraph wiring
│
├── tools/
│   └── search.py                       # Search tool (ChromaDB, TF-IDF, BGE reranker, synonyms)
│
├── guardrails/
│   ├── security_shield.py              # Jailbreak, PII masking, rate limiter, injection filter
│   ├── input_validator.py              # Query validation (security → TF-IDF → LLM)
│   └── output_validator.py             # Groundedness check & output PII scrub
│
├── config/
│   └── settings.py                     # LLM factory (DeepSeek / Azure APIM), embeddings, token utils
│
├── prompts/
│   ├── retriever_system.txt
│   ├── generator_system.txt
│   ├── input_validator_prompt.txt
│   └── output_validator_prompt.txt
│
├── evaluation/
│   ├── golden_dataset.json             # 20 benchmark Q&A pairs
│   ├── evaluation_report.json
│   └── metrics.py
│
├── tests/                              # 57 tests, ~80% coverage
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_reranker.py
│   ├── test_search.py
│   ├── test_guardrails.py
│   ├── test_integration.py
│   └── test_security_guardrails.py
│
└── .github/workflows/
    └── ci.yml
```

---

## Security & Privacy

- **PII compliance** — Thai IDs, credit cards, phone numbers, and emails are scrubbed before processing and in output.
- **No key leakage** — API keys, tokens, and system prompts are detected and redacted.
- **Hallucination check** — every response is validated against source chunks before delivery.

---

## License & Project Info

Developed for the **AI ENGINEER TEST \| AI Policy Assistant** project.  
Built with LangGraph, ChromaDB, Azure APIM (GPT-5-Mini), and DeepSeek.
