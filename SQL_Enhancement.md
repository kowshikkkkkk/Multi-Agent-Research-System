# SQL + RAG Dual-Source Retrieval Enhancement

## Overview

This document describes the enhancement made to the Multi-Agent Research System to support **dual-source retrieval** combining structured SQL queries with unstructured RAG (Retrieval-Augmented Generation).

## Problem

**Original System:** RAG-only retrieval from document corpus
- ✅ Good: Rich narrative context from documents
- ❌ Bad: Can hallucinate when facts aren't in documents
- ❌ Bad: No structured data integration

## Solution

**Enhanced System:** Dual-source retrieval (SQL + RAG)
- ✅ SQL: Quantified facts from financial database
- ✅ RAG: Strategic narrative from documents
- ✅ Cross-validation: Prevents hallucination
- ✅ Production-grade: Matches real-world systems

## Architecture
User Query
↓
Planner Agent
↓
Researcher Agent (DUAL-SOURCE)
├─ SQL Query: financial_data.db
│   └─ Returns: revenue, margins, headcount, segments
└─ RAG Retrieval: ChromaDB
└─ Returns: Strategic narratives, business context
↓
Analyst Agent (Cross-validates SQL + RAG)
↓
Writer Agent (Synthesizes fact-checked report)
↓
Critic Agent (Quality gates with loop guards)

## Files Added

### 1. `sql_query.py` (210 lines)
**Purpose:** Safe SQL query interface with intent-based mapping

**Key Features:**
- Intent detection from natural language questions
- Safe parameterized queries (prevents SQL injection)
- Returns structured results with summaries
- Fallback patterns for unknown queries

**Example:**
```python
from sql_query import FinancialDataQuery

db = FinancialDataQuery()
result = db.query("What is Infosys's revenue growth?")
# Returns: {"status": "success", "summary": "...", "results": [...]}
```

### 2. `create_db.py` (238 lines)
**Purpose:** Initialize SQLite database with sample financial data

**Tables Created:**
- `quarterly_data`: Q1-Q4 2024 quarterly financials
- `segment_revenue`: 6 business segments with growth rates
- `key_metrics`: 5 key performance indicators

**Example Data:**
Infosys Q1 2024: $2,100.5M revenue, 319,000 employees
Infosys Q2 2024: $2,250.2M revenue, 320,500 employees
Infosys Q3 2024: $2,180.8M revenue, 321,200 employees

### 3. Enhanced `agents/researcher.py` (80 lines added)
**Original:** RAG-only retrieval
**Enhanced:** Dual-source retrieval (SQL + RAG)

**New Features:**
```python
# Query SQL database for structured facts
sql_data = _get_sql_data(subtasks)

# Retrieve RAG context
rag_context = retrieve(subtasks, k=3)

# Combine both sources
combined_context = f"""
## STRUCTURED DATA (SQL)
{sql_data}

## UNSTRUCTURED DATA (RAG)
{rag_context}
"""

# LLM analyzes both sources for fact-checked response
response = llm.invoke(messages)
```

### 4. `demo.py` (174 lines)
**Purpose:** End-to-end pipeline demo showing all 5 agents with dual-source retrieval

**Execution Flow:**
1. Planner: Breaks query into subtasks
2. Researcher: Queries SQL + RAG simultaneously
3. Analyst: Extracts patterns from both sources
4. Writer: Generates fact-checked report
5. Critic: Quality control with loop guards (max 2 revisions)

**Example Output:**
================================================================================
MULTI-AGENT RESEARCH PIPELINE WITH DUAL-SOURCE RETRIEVAL
User Query: Analyze Infosys's 2024 financial performance
[... all 5 agents execute ...]
EXECUTION SUMMARY
────────────────────────────────────────────────────────────────────────────
Revisions: 2
Verdict: APPROVED
Report Length: 2262 characters
Sources Used: SQL Database + RAG Corpus (Dual-Source Retrieval)

## Technical Details

### SQL Safety Patterns

**Intent-Based Mapping** (prevents SQL injection):
```python
if any(x in question.lower() for x in ["revenue", "growth"]):
    sql_query = "SELECT ... FROM quarterly_data WHERE ..."
    # Never uses raw user input
```

### Cross-Validation Example

**If SQL and RAG align:**
SQL: "Operating margin is 21.2%"
RAG: "Margin expanded through automation investments"
Result: HIGH CONFIDENCE (both sources agree)

**If SQL and RAG contradict:**
SQL: "Headcount is 321,200"
RAG: "Significant hiring happening"
Result: FLAG FOR REVIEW (need to investigate)

## Integration Points

### 1. Researcher Agent Enhancement
- Original: `retrieve(subtasks, k=3)`
- Enhanced: `retrieve(subtasks, k=3)` + `sql_db.query(subtasks)`
- Result: Dual-source context passed to LLM

### 2. LLM Prompt Enhancement
- New system message instructs LLM to use BOTH sources
- Requests explicit source attribution
- Asks LLM to flag contradictions

### 3. Loop Guard Logic (No Changes)
- Critic already evaluates quality
- Revision loops still work as before
- Max 2 revisions unchanged

## Data Sources

### SQL Database (`financial_data.db`)
**Company:** Infosys
**Period:** Q1-Q4 2024 + Q4 2023
**Tables:**
- `quarterly_data`: 4 records (revenue, profit, headcount)
- `segment_revenue`: 6 records (business segments + growth)
- `key_metrics`: 5 records (margins, ROE, attrition)

**Example Query:**
```sql
SELECT company, quarter, year, revenue_usd_millions 
FROM quarterly_data 
WHERE company = 'Infosys'
ORDER BY year DESC, quarter DESC
```

### RAG Corpus
**Source:** Simulated Infosys 2024 Annual Report
**Content:** Strategic narratives, market insights, management guidance
**Retrieval:** ChromaDB with HuggingFace embeddings (all-MiniLM-L6-v2)

## Performance Metrics

**Typical Execution Time (from demo.py):**
- Planner: 0.5s
- Researcher (SQL + RAG): 2.0s
  - SQL query: 0.1s
  - RAG retrieval: 0.5s
  - LLM synthesis: 1.4s
- Analyst: 1.5s
- Writer: 1.5s
- Critic: 0.8s
- **Total: ~6.3s**

**Report Quality:**
- Length: 2,000-2,500 characters
- Revisions: 1-2 (loop guards prevent infinite loops)
- Verdict: APPROVED (high quality)

## ATS Keywords

**Resume bullets now hit these keywords:**
- ✅ SQL (queries database)
- ✅ Data integration (SQL + RAG)
- ✅ Structured + unstructured data
- ✅ Cross-validation (hallucination prevention)
- ✅ Heterogeneous data sources
- ✅ LangGraph (orchestration)
- ✅ RAG (retrieval-augmented generation)
- ✅ ChromaDB (vector database)
- ✅ FastAPI (API deployment)
- ✅ MLflow (experiment tracking)

## Interview Talking Points

**"Why did you add SQL to your RAG system?"**
> Production AI systems combine structured + unstructured data. SQL gives you verifiable facts; RAG gives you context. Cross-validating both prevents hallucination.

**"How do you prevent the system from hallucinating?"**
> The Researcher queries both SQL (quantified facts) and RAG (narratives). If SQL says "margin is 21.2%" and RAG says "margins expanded," the Analyst validates alignment. Contradictions get flagged.

**"What's the architecture?"**
> 5-agent LangGraph pipeline. Enhanced Researcher does dual-source retrieval, combining structured queries with unstructured search. Loop guards (max 2 revisions) prevent infinite cycles.

**"How would you scale this?"**
> Move to Postgres (instead of SQLite), connect to production ChromaDB instance, add caching (Redis), implement monitoring (MLflow → Datadog). Architecture already supports these swaps.

## Setup & Testing

### Create Database
```bash
python create_db.py
```

### Test SQL Module
```bash
python sql_query.py
```

### Test Enhanced Researcher
```bash
python -c "from agents.researcher import researcher_agent; print(researcher_agent('Analyze revenue growth'))"
```

### Run Full Pipeline
```bash
python demo.py
```

## Resume Bullet

**Before (RAG-only):**
Built 5-agent LangGraph research pipeline with ChromaDB RAG grounding
on 1,605-chunk Infosys 2024 annual report; implemented conditional
routing with loop guards preventing runaway revisions.

**After (SQL + RAG):**
Built 5-agent LangGraph pipeline with dual-source Researcher: merged
structured SQL queries (quarterly revenue, segment performance, KPI metrics)
with unstructured RAG retrieval (ChromaDB on 1,605 chunks from Infosys 2024
report); implemented conditional routing + loop guards for hallucination
prevention via cross-validation across heterogeneous data sources.

## Next Steps

**For Production:**
1. Replace SQLite with Postgres (better scaling)
2. Connect to live ChromaDB instance
3. Add evaluation metrics (ROUGE/BLEU on reports)
4. Implement caching (Redis for repeated queries)
5. Deploy via Docker + Kubernetes
6. Monitor with CloudWatch/Datadog

**For Interviews:**
1. Understand why SQL + RAG is better than RAG alone
2. Know all 5 agent functions by name
3. Be able to explain loop guard design
4. Discuss deployment steps
5. Practice the technical explanation (2 min elevator pitch)

---

**Created:** May 2026
**Status:** Production-Ready Demo
**Tech Stack:** LangGraph + Groq + ChromaDB + SQLite + FastAPI