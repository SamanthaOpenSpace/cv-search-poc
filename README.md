# cv-search

Local-first candidate search & planning with **SQLite + FAISS** and a Streamlit UI.  
Built as a `src/`-layout Python package and managed with **uv**.

- **UI**: `app.py` + `pages/`
- **CLI**: `main.py` (Click)
- **Vectors**: FAISS file at `data/cv_search.faiss`
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Config**: `.env` → `Settings`

---

## 🔧 Prerequisites

- **Python**: 3.11+
- **uv**: environment & package manager
- (Optional) **rclone**: only required for the Google Drive ingestion flow

> Copy `.env.example` → `.env` and fill the keys you actually use.

---

## ⚙️ Setup (uv-first)

```bash
# Create a local virtual environment for Python 3.11
uv venv -p 3.11

# Install project deps as defined in pyproject.toml
uv sync
```

> `uv sync` installs the package in editable mode (src layout) because the project sets `[tool.uv].package = true`.

---

## 🔐 Configure environment

Create `.env` in the repo root (start from `.env.example`) and set one provider:

**Standard OpenAI**
```ini
OPENAI_API_KEY=sk-...
# Optional overrides
# OPENAI_MODEL=gpt-4.1-mini
# OPENAI_EMBED_MODEL=text-embedding-3-large
```

**Azure OpenAI**
```ini
USE_AZURE_OPENAI=True
OPENAI_API_KEY=...
AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_API_VERSION=2024-02-01
# In Azure, these are deployment names:
OPENAI_MODEL=<chat-deployment>
OPENAI_EMBED_MODEL=<embed-deployment>
```

**Search defaults (optional)**
```ini
# "hybrid" | "lexical" | "semantic"
SEARCH_MODE=hybrid
```

**Google Drive sync (optional)**
```ini
GDRIVE_REMOTE_NAME=gdrive
GDRIVE_SOURCE_DIR=CV_Inbox
# GDRIVE_LOCAL_DEST_DIR=data/gdrive_inbox
# GDRIVE_RCLONE_CONFIG_PATH=<custom rclone.conf path if not default>
```

---

## ✅ Bootstrapping checks

```bash
# Show detected env & settings (API key masked)
uv run python main.py env-info

# Initialize (or re-initialize) the database schema
uv run python main.py init-db

# Verify DB tables and FTS support
uv run python main.py check-db
```

---

## 📥 Ingestion flows

> **Read this first:** The current codebase shows a few WIP gaps in ingestion (see **Known gaps** below).  
> You can still use the UI/CLI for parsing/planning/search once your DB & FAISS are seeded.

### Option A — Mock data (JSON)

```bash
uv run python main.py ingest-mock
```

- Intention: rebuild `cvsearch.db` and the FAISS index from `data/test/mock_cvs.json`.
- **Current status:** this command references a non-existent `Settings.test_data_dir` and will error out until the attribute is added or the code is patched. See **Known gaps**.

### Option B — Google Drive inbox (via rclone)

1) Install and configure **rclone** (`rclone config`) with a remote matching `GDRIVE_REMOTE_NAME` in `.env`.
2) Pull `.pptx` CVs locally, then parse & ingest:

```bash
uv run python main.py sync-gdrive     # downloads from Drive → local inbox
uv run python main.py ingest-gdrive   # parses PPTX → JSON, upserts DB & FAISS
uv run python main.py ingest-gdrive --file "test.pptx"  # parses PPTX → JSON, upserts DB & FAISS

```

- **Current status:** ingestion calls `get_or_create_faiss_id(...)` which is not present in `CVDatabase`; implement this helper before this flow can succeed. See **Known gaps**.

---

## 🔎 Search & planning (CLI)

> **Tip:** Run search after you have data + FAISS in place. Semantic search warns if the index is missing.

### Parse a free-text brief → normalized criteria JSON
```bash
uv run python main.py parse-request   --text "Greenfield healthtech app; 2 senior .NET + 1 React; Kafka, Postgres; Playwright"
```

### Single-seat search (from criteria JSON)
```bash
uv run python main.py search-seat   --criteria ./data/test/criteria.json   --topk 3   --mode hybrid   --justify
```

### Project search (multi-seat)

From text (derives seats deterministically):
```bash
uv run python main.py project-search   --text "Mobile + web app; React/Next.js + .NET; Kafka; healthtech"   --topk 3   --justify
```

From explicit criteria:
```bash
uv run python main.py project-search   --criteria ./data/test/criteria.json   --topk 3   --justify
```

### Presale planning (stateless roles from a brief)
```bash
uv run python main.py presale-plan   --text "Mobile + web app; Flutter or React Native; LLM chatbot; analytics dashboards"
```

---

## 🖥️ Streamlit UI

```bash
uv run streamlit run app.py
```

Pages:
- **Project Search** – derive seats from a brief or upload a criteria JSON; run per‑seat search (optionally with LLM justifications).
- **Single Seat Search** – build a one-seat query from widgets; run lexical/semantic/hybrid ranking.
- **Admin & System** – status metrics, mock re‑ingest button.

> **Note:** The “Upload New CVs” button in **Admin & Ingest** references a method removed from the ingestion pipeline. Use the CLI once ingestion helpers are restored. See **Known gaps**.

---

## 🧠 Retrieval modes

- **Lexical**: weighted SQL over normalized tags (role/tech/domain/seniority) in SQLite.
- **Semantic**: FAISS search over a synthesized per-candidate document; requires the local index.
- **Hybrid**: late fusion (`w_lex`, `w_sem`) with per-seat diagnostics and artifacts (optionally saved under `runs/<timestamp>/`).

On first run, `sentence-transformers` downloads the model (`all-MiniLM-L6-v2`). Subsequent runs use cache.

---

## 🛠️ Known gaps & temporary guidance

These are present in the current repo snapshot and affect ingestion/justification paths:

1. **`ingest-mock`** references `Settings.test_data_dir`, which does not exist; add this attribute (e.g., defaulting to `REPO_ROOT / "data" / "test"`) or adapt the call site to use `settings.data_dir / "test"`.
2. **FAISS ID mapping**: ingestion calls `CVDatabase.get_or_create_faiss_id(...)`, but this helper is not implemented; add it to persist a stable `faiss_id` per `candidate_id` in `faiss_id_map`.
3. **UI upload ingestion**: Streamlit’s Admin page calls `pipeline.run_ingestion_from_list(...)`, which was explicitly deleted during refactor; reintroduce a compatible method or wire the UI to `upsert_cvs(...)`.
4. **LLM justification context**: both the UI and `JustificationService` expect `CVDatabase.get_full_candidate_context(...)` which doesn’t exist; implement a query that returns `summary_text`, `experience_text`, and `tags_text` from `candidate_doc`.

Until those are addressed:
- Prefer running **search flows** against a DB you’ve already seeded (once the above helpers are added).
- If you need semantic results, ensure the FAISS index exists; otherwise, run in **lexical** mode to test ranking behavior.

---
