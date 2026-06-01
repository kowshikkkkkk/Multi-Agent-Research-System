from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from utils.rag import retrieve
from sql_query import FinancialDataQuery
from cross_validate import cross_validate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=800
)

sql_db = FinancialDataQuery(db_path="financial_data.db")


def _get_sql_data(subtasks: str) -> dict:
    print("  [SQL] Querying financial database...")
    sql_result = sql_db.query(subtasks)
    
    if sql_result["status"] == "success":
        formatted = f"**SQL Data Summary:** {sql_result['summary']}\n"
        formatted += f"**Records Found:** {sql_result['record_count']}\n"
        if sql_result['results']:
            formatted += "**Sample Data:**\n"
            for i, row in enumerate(sql_result['results'][:3]):
                formatted += f"  {i+1}. {row}\n"
        return {
            "success": True,
            "data": formatted,
            "record_count": sql_result['record_count']
        }
    else:
        return {
            "success": False,
            "data": "No structured data found",
            "record_count": 0
        }


def researcher_agent(subtasks: str, retrieval_config: dict = None) -> str:
    print("\n" + "="*70)
    print("RESEARCHER AGENT (Dual-Source: SQL + RAG)")
    print("="*70)
    
    # Default config if not provided
    if retrieval_config is None:
        retrieval_config = {"k": 5, "keywords_to_emphasize": []}
    
    k = retrieval_config["k"]
    extra_keywords = retrieval_config["keywords_to_emphasize"]
    
    # Adjust query on revision
    if extra_keywords:
        adjusted_subtasks = subtasks + " " + " ".join(extra_keywords)
        print(f"🔄 REVISION MODE: Added keywords: {', '.join(extra_keywords)}")
    else:
        adjusted_subtasks = subtasks
    
    # Step 1: SQL (adjusted query)
    print("Retrieving structured financial data from database...")
    sql_data = _get_sql_data(adjusted_subtasks)
    
    # Step 2: RAG (adjusted query AND adjusted k)
    print(f"Retrieving relevant content from document store (k={k})...")
    rag_context = retrieve(adjusted_subtasks, k=k) 
    
    # Step 3: Combine
    combined_context = f"""
DATA SOURCES (Dual-Retrieval):

## STRUCTURED DATA (SQL - Financial Database)
{sql_data['data']}

## UNSTRUCTURED DATA (RAG - Document Corpus)
{rag_context}

---
INSTRUCTIONS:
- Use BOTH sources to provide comprehensive analysis
- Prioritize SQL data for quantified facts (numbers, metrics)
- Use RAG context for strategic narrative and rationale
- Cross-validate: if SQL says "margin is 21.2%" and RAG says "margins expanded", note the alignment
- If sources disagree, flag the contradiction
- Do not make up any numbers or facts
    """
    
    # Step 4: LLM
    messages = [
        SystemMessage(content="""You are an advanced research agent with access to BOTH structured and unstructured data.

Your role:
1. Analyze subtasks using BOTH SQL facts (quantified) and RAG context (narrative)
2. Provide fact-checked research that combines structured data with strategic context
3. Explicitly mention which source (SQL or RAG) each finding comes from
4. Highlight cross-validation between sources (if SQL and RAG align, that's high confidence)
5. Flag any contradictions or data gaps
6. Never fabricate numbers or facts - only cite what's in the provided sources

Format your response with:
- **Key Finding:** [from SQL or RAG]
- **Supporting Context:** [from the other source]
- **Confidence Level:** [based on source alignment]
        """),
        HumanMessage(content=f"Subtasks:\n{subtasks}\n\nCombined Context:\n{combined_context}")
    ]
    
    response = llm.invoke(messages)
    
    # Step 5: Programmatic cross-validation
    validation = cross_validate(subtasks, rag_context)
    
    validation_summary = f"""
---
**CROSS-VALIDATION REPORT**
- Status:     {validation['status']}
- Confidence: {validation['confidence']}
- SQL Truth:  {validation.get('sql_summary', 'N/A')}
- Verified:   {len(validation['verified_facts'])} facts
- Contradictions: {len(validation['contradictions'])}
"""
    
    if validation['contradictions']:
        validation_summary += "\n⚠️ CONTRADICTIONS:\n"
        for c in validation['contradictions']:
            validation_summary += f"  RAG says {c['rag_number']} but SQL says {c['sql_value']}\n"
    
    attribution = f"\n---\n**Sources:** SQL ({sql_data['record_count']} records) + RAG | Validation: {validation['status']} ({validation['confidence']} confidence)"
    
    return response.content + validation_summary + attribution