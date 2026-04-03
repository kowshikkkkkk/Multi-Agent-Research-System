from fastapi import FastAPI
from pydantic import BaseModel
from graph import build_graph

app = FastAPI(
    title="Multi-Agent Research System",
    description="AI-powered research and report generation pipeline",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

class ReportResponse(BaseModel):
    query: str
    plan: str
    research: str
    analysis: str
    report: str
    critique: str
    revision_count: int

@app.get("/")
def root():
    return {"message": "Multi-Agent Research System is running!"}

@app.post("/analyze", response_model=ReportResponse)
def analyze(request: QueryRequest):
    pipeline = build_graph(request.query)
    
    initial_state = {
        "query": request.query,
        "plan": "",
        "research": "",
        "analysis": "",
        "report": "",
        "critique": "",
        "revision_count": 0
    }
    
    final_state = pipeline.invoke(initial_state)
    return final_state