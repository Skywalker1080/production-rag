# Simple PDF RAG — LangChain + FastAPI + Qdrant (Docker) + Bedrock GLM 5

PDF-only RAG. Upload PDFs → chunk → Bedrock Titan embeddings → Qdrant → ask via **GLM 5 on Bedrock** (`zai.glm-5`, Converse API via `langchain-aws` `ChatBedrock`).

## 1. Prereqs

- Python 3.11+, Docker, AWS account with Bedrock **Model Access** enabled for:
  - `zai.glm-5` (LLM)
  - `amazon.titan-embed-text-v2:0` (embeddings)
- Supported region, e.g. `us-east-1` (or `us-west-2` per AWS docs).

## 2. Setup

```powershell
cd C:\Projects\RAG

# Qdrant
docker compose up -d
# check: http://localhost:6333/dashboard

# Python env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Env — paste your Bedrock API key
copy .env.example .env
# edit .env: AWS_BEARER_TOKEN_BEDROCK=<your bedrockapi key> + AWS_REGION
```

One-time model warmup (pre-downloads layout + table models so uploads
never wait on the network — cached permanently in `~/.cache/huggingface`):

```powershell
uv run python -m app.warmup
```

`.env` keys:

| Key | Default | Notes |
|---|---|---|
| `AWS_BEARER_TOKEN_BEDROCK` | — | **your Bedrock API key (no access ID needed)** |
| `AWS_REGION` | `us-east-1` | region where GLM 5 is enabled |
| `BEDROCK_MODEL_ID` | `zai.glm-5` | GLM 5 on Bedrock |
| `BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | embeddings |
| `QDRANT_URL` | `http://localhost:6333` | docker Qdrant |
| `QDRANT_COLLECTION` | `pdf_docs` | auto-created (1024-dim, cosine) |

## 3. Run

```powershell
uvicorn app.main:app --reload --port 8000
```

- UI: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## 4. Use

1. Open UI → **Upload PDF** (only `.pdf` accepted, stored in `./data/`).
2. **Ask** a question → answer + sources (file + page + snippet).
3. `DELETE /api/documents` (UI button) clears the Qdrant collection.

## 5. API

| Method | Path | Body | Purpose |
|---|---|---|---|
| GET | `/health` | — | backend + Qdrant + chunk count |
| POST | `/api/upload` | `multipart: file=<pdf>` | **202 + `{job_id}`**, ingestion runs in background |
| GET | `/api/jobs/{job_id}` | — | poll job: `queued/running/done/failed` + current `step` + `detail` |
| POST | `/api/ask` | `{"question": "...", "top_k": 4}` | RAG query |
| GET | `/api/stats` | — | chunk count + uploaded files |
| DELETE | `/api/documents` | — | wipe collection |

## 6. How it works

```
PDF upload → UnstructuredPDFLoader (elements, layout-aware for multi-column)
  → RecursiveCharacterTextSplitter (1000/200)
  → Ollama nomic-embed-text (local, 768-dim) → Qdrant (pdf_docs)
Ask → similarity_search(top_k=4) → prompt + context
  → ChatBedrock(zai.glm-5) → answer + sources
```

Embeddings default to **local Ollama** (`nomic-embed-text`, needs
`ollama serve` + `ollama pull nomic-embed-text`) — no cloud throttling.
Set `EMBED_PROVIDER=bedrock` in `.env` to use Titan instead. Switching
providers auto-recreates the Qdrant collection (dims differ: 768 vs 1024).

`UNSTRUCTURED_STRATEGY` (in `.env`, default `fast`): `fast` (seconds, plain
text order) | `hi_res` (minutes on CPU, best multi-column reading order,
needs poppler + tesseract) | `auto`. Unstructured's own partition logs
(strategy, pages, fallbacks) are routed into `./logs/rag.log` alongside
pipeline steps.

## 7. Troubleshooting

- Logs: every step is logged to `./logs/rag.log` (rotated, 2 MB × 3) and the
  console. Format: `timestamp | LEVEL | rag | step | message`. Ingest steps:
  `ensure_collection → load_pdf → split → upsert`; query steps:
  `retrieve → generate`. Failures log `FAILED step=<name>` with a full
  traceback, so you can see exactly which step broke.
- `Qdrant not reachable` → `docker compose up -d`, check port 6333.
- Bedrock `AccessDenied` → enable Model Access in Bedrock console (both models), check region + keys.
- `ValidationException: 1024-dim mismatch` → you changed embed model after ingesting; `DELETE /api/documents` and re-upload.
