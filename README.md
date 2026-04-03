# Multi-Agent AI Research & Report Automation System

An end-to-end agentic AI pipeline that autonomously researches, analyzes, 
and generates professional reports from documents using multiple specialized AI agents.

## Architecture
User Query
↓
Planner Agent    → breaks query into structured subtasks
↓
Researcher Agent → retrieves real data from PDF via RAG (ChromaDB)
↓
Analyst Agent    → reasons over findings, identifies patterns
↓
Writer Agent     → produces professional business report
↓
Critic Agent     → scores report, loops back if NEEDS REVISION

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Agent Orchestration | LangGraph |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Experiment Tracking | MLflow |
| API | FastAPI |
| UI | Streamlit |

## Key Features

- **Multi-agent orchestration** using LangGraph state graph
- **RAG pipeline** grounding agents in real document data
- **Conditional edges** with automatic back-looping on low quality reports
- **Loop guard** preventing infinite revision cycles
- **MLflow logging** tracking latency and outputs per agent
- **FastAPI endpoint** exposing pipeline as REST API
- **Streamlit UI** for interactive demos

## Setup
```bash
# Clone the repo
git clone https://github.com/kowshikkkkkk/Multi-Agent-Research-System.git
cd Multi-Agent-Research-System

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# Run the pipeline
python main.py
```

## Running the Full Stack
```bash
# Terminal 1 - FastAPI
uvicorn api:app --reload

# Terminal 2 - Streamlit
streamlit run app.py

# Terminal 3 - MLflow UI
mlflow ui
```

## Project Structure
├── agents/
│   ├── planner.py      # Breaks query into subtasks
│   ├── researcher.py   # RAG-based document retrieval
│   ├── analyst.py      # Reasoning over findings
│   ├── writer.py       # Report generation
│   └── critic.py       # Quality control
├── utils/
│   ├── rag.py          # ChromaDB vector store pipeline
│   └── logger.py       # MLflow logging
├── graph.py            # LangGraph state graph wiring
├── api.py              # FastAPI endpoint
└── app.py              # Streamlit UI
