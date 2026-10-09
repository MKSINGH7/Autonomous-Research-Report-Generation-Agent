# Autonomous Research & Report Generation Agent

A Retrieval-Augmented Generation (RAG) research assistant built with **LangGraph**,
**Chroma**, **Sentence-Transformers** and **Streamlit**. Upload PDF/TXT/DOCX files,
ask questions, and get grounded answers, structured research reports and the exact
source passages used. Runs fully offline with **Ollama**, or online with **Gemini**.

## Features
- PDF / TXT / DOCX ingestion with validation and page-level metadata
- Recursive chunking with configurable size and overlap
- Local embeddings (`all-MiniLM-L6-v2`, 384-dim) — no paid API
- Persistent Chroma vector store (survives restarts, re-index replaces old chunks)
- Cosine-similarity top-K retrieval with a relevance threshold
- Two LangGraph workflows: ingestion and research (with conditional routing)
- Grounded answers with inline `[Source n]` citations
- 8-section structured report (Sources section generated from metadata)
- Switch between Ollama (local) and Gemini (online) with one setting
- Friendly handling of missing Ollama / model / API key / rate limits / empty input

## Architecture
```
Ingestion: START → load_documents → clean_text → chunk_documents → index_chunks → END
Research:  START → retrieve_context ─┬─ (nothing relevant) → no_context → END
                                     └─ generate_answer ─┬─ generate_report → END
                                                         └─ (answer only)  → END
```

## Technologies
Python 3.14.0 · LangGraph · langchain-core · Chroma · sentence-transformers ·
Ollama · Google Gemini · pypdf · python-docx · Streamlit · pytest

## Project structure
```
main.py            Streamlit UI
src/config.py     settings from .env
src/utils.py      exceptions, text cleaning, cosine similarity
src/loaders.py    validation + text extraction
src/chunking.py   chunking + metadata
src/embeddings.py local embedding model
src/vectorstore.py persistent Chroma wrapper
src/retriever.py  search, context, sources
src/llm.py        provider registry (Ollama/Gemini)
src/prompts.py    RAG + report prompts
src/report_generator.py  report assembly
src/graph.py      LangGraph workflows
tests/test_pipeline.py
```

## Installation (Windows, PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run main.py
```
(Command Prompt: `.venv\Scripts\activate.bat`.
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)

## Local LLM setup (Ollama)
1. Install Ollama from https://ollama.com and check `ollama --version`
2. `ollama pull llama3.2:1b`  then test with `ollama run llama3.2:1b`
3. In `.env`: `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=llama3.2:1b`

## Online API setup (Gemini)
1. Create a key at https://aistudio.google.com
2. In `.env`: `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...`
3. Model names change (e.g. `gemini-2.5-flash` is scheduled to shut down on
   16 Oct 2026). If you get a 404, update `GEMINI_MODEL`.

## Configuration
See `.env.example` — every setting is documented there
(`TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_SIMILARITY`, `EMBEDDING_MODEL`, …).

## Usage
1. Upload documents → **Process / Index documents**
2. Enter a research question → **Research**
3. Read the Answer, Report and Sources tabs; download the report as Markdown

## Screenshots
_Add screenshots here (e.g. `docs/ui.png`)._

## Testing
```powershell
python -m pytest -v
# also test the real embedding model (downloads it once):
$env:RUN_SLOW_TESTS="1"; python -m pytest -v
```

## Deployment
- Local: `streamlit run main.py`
- Streamlit Community Cloud: push to GitHub, add `LLM_PROVIDER` and
  `GEMINI_API_KEY` under *Secrets*. **Ollama does not run on Streamlit Cloud**, and
  its filesystem is ephemeral, so the vector store may reset.

## Security
Never commit `.env`. It is listed in `.gitignore`. If a key leaks, revoke it first.

## Troubleshooting
| Problem | Fix |
|---|---|
| Cannot reach Ollama | Start Ollama / `ollama serve` |
| Model not installed | `ollama pull <model>` |
| API key rejected | Re-check `GEMINI_API_KEY` in `.env` |
| No relevant results | Lower *Minimum similarity*, raise *Top-K* |
| Scanned PDF has no text | OCR is not included |
| `ModuleNotFoundError: src` | Run from the project root; use `python -m pytest` |

## Future improvements
Hybrid BM25 + vector search · cross-encoder reranking · MMR · query rewriting ·
OCR for scanned PDFs · streaming responses · evaluation with RAGAS · multi-user
collections · hosted vector DB for cloud deployment.
