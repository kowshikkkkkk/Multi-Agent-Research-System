# Multi-Agent Financial Research System
An autonomous AI pipeline that researches, analyzes, and generates 
fact-verified financial reports using 5 specialized agents — with 
programmatic hallucination detection, automated quality control, 
and systematic RAG evaluation.

---

## What It Does
You ask a question like *"Analyze Infosys 2024 financial risks"* and the system:
1. Breaks it into structured research subtasks
2. Pulls verified numbers from a SQL database AND narrative context from a vector database simultaneously
3. Cross-validates both sources programmatically — flags contradictions before they reach the report
4. Generates a professional report, scores it, and revises automatically if quality is insufficient
5. Tracks every run in MLflow for systematic improvement

---

## Architecture
User Query
↓
┌─────────────┐
│   Planner   │  Breaks query into structured subtasks
└──────┬──────┘
↓
┌─────────────────────────────────────────┐
│           Researcher (Dual-Source)      │
│  SQLite ──┐                             │
│           ├──→ Cross-Validation ──→ LLM │
│  ChromaDB ┘                             │
└──────┬──────────────────────────────────┘
↓
┌─────────────┐
│   Analyst   │  Pattern recognition, risk identification
└──────┬──────┘
↓
┌─────────────┐
│    Writer   │  Professional report generation
└──────┬──────┘
↓
┌─────────────┐     NEEDS REVISION (max 2x)
│    Critic   │ ────────────────────────────→ Researcher
└──────┬──────┘     (critique-aware loop)
↓
Final Report

---

## Key Features

**Dual-Source Retrieval**
SQL database for verified financial figures. ChromaDB vector store 
(1,605 chunks, all-MiniLM-L6-v2) for narrative context from the 
annual report. Both sources retrieved simultaneously.

**Programmatic Cross-Validation**
Numbers extracted from RAG output are verified against SQL ground 
truth with 5% tolerance. Contradictions flagged before reaching the 
Writer — no LLM involved in fact checking.

**Critique-Aware Revision Loop**
When the Critic says NEEDS REVISION, the Researcher reads exactly 
why it was sent back and targets those specific gaps. Max 2 revision 
cycles enforced by loop guard. Full revision history preserved in state.

**RAG Evaluation Suite**
Custom evaluation measuring retrieval relevance, faithfulness, and 
answer completeness across 5 structured test cases. A/B tested k=3 
vs k=5 retrieval — k=5 improved completeness from 0.20 to 0.46 at 
the cost of 3.5x latency increase. All runs tracked in MLflow.

**19/19 Unit Tests Passing**
Pytest suite covering RAG retrieval, SQL queries, planner output, 
critic verdicts, loop guard logic, and end-to-end pipeline execution.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Agent Orchestration | LangGraph |
| Vector Store | ChromaDB (HNSW indexing) |
| Embeddings | all-MiniLM-L6-v2 (384 dimensions) |
| Structured Data | SQLite |
| Experiment Tracking | MLflow |
| Evaluation | Custom RAG eval suite |
| Testing | Pytest (19/19 passing) |
| API | FastAPI (async) |
| UI | Streamlit |

---

## Evaluation Results

| Run | What Changed | Overall Score |
|---|---|---|
| Run 1 | Baseline | 0.387 ❌ |
| Run 2 | Fixed duplicate chunks (17,655 → 1,605) | 0.523 🔶 |
| Run 3 | Fixed ground truth currency mismatch | 0.624 ⚠️ |
| Run 4 | Increased retrieval k=3 → k=5 | 0.664 ⚠️ |

MLflow tracked every parameter and metric across all runs.

---

## Setup

```bash
# Clone
git clone https://github.com/kowshikkkkkk/Multi-Agent-Research-System.git
cd Multi-Agent-Research-System

# Environment
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt

# API Key
echo GROQ_API_KEY=your_key_here > .env

# Build vector store and database
python create_db.py
python -c "from utils.rag import build_vector_store; build_vector_store()"

# Run pipeline
python main.py
```

---

## Running the Full Stack

```bash
# Terminal 1 — API
uvicorn api:app --reload

# Terminal 2 — UI
streamlit run app.py

# Terminal 3 — Experiment tracking
mlflow ui

# Run evaluation suite
python evaluate.py

# Run tests
pytest tests/test_pipeline.py -v
```

---

## Project Structure
├── agents/
│   ├── planner.py       # Query decomposition
│   ├── researcher.py    # Dual-source retrieval + cross-validation
│   ├── analyst.py       # Pattern recognition
│   ├── writer.py        # Report generation
│   └── critic.py        # Quality scoring + revision trigger
├── tests/
│   └── test_pipeline.py # 19 unit tests
├── utils/
│   ├── rag.py           # ChromaDB pipeline
│   └── logger.py        # MLflow logging
├── cross_validate.py    # Programmatic SQL-RAG fact checker
├── evaluate.py          # RAG evaluation suite
├── graph.py             # LangGraph state graph
├── sql_query.py         # Intent-based SQL interface
├── create_db.py         # Database setup
├── api.py               # FastAPI endpoint
└── app.py               # Streamlit UI

---

## Known Limitations and Roadmap

**Current limitations:**
- Hardcoded for Infosys — SQL schema and ChromaDB not yet 
  parameterized for multi-company support
- Critic uses LLM-as-judge with no external ground truth for 
  narrative queries
- Single worker FastAPI — not production-ready for concurrent load

**Planned improvements:**
- Metadata filtering in ChromaDB for multi-company support
- Text-to-SQL replacing intent-based routing for scalability
- Redis queue + multiple workers for production concurrency
- FinBERT embeddings for better financial domain retrieval
