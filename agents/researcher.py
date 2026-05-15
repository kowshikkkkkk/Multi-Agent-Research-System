from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from utils.rag import retrieve
from sql_query import FinancialDataQuery
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=800
)

# Initialize SQL query interface
sql_db = FinancialDataQuery(db_path="financial_data.db")


def _get_sql_data(subtasks: str) -> dict:
    """
    Query structured financial database for facts.
    """
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


def researcher_agent(subtasks: str) -> str:
    """
    Enhanced researcher agent with dual-source retrieval (SQL + RAG).
    
    Args:
        subtasks: Research task/subtask
    
    Returns:
        str: Combined research findings from SQL + RAG sources
    """
    print("\n" + "="*70)
    print("RESEARCHER AGENT (Dual-Source: SQL + RAG)")
    print("="*70)
    
    # Step 1: Get structured data from SQL
    print("Retrieving structured financial data from database...")
    sql_data = _get_sql_data(subtasks)
    
    # Step 2: Get unstructured data from RAG
    print("Retrieving relevant content from document store...")
    rag_context = retrieve(subtasks, k=3)
    
    # Step 3: Combine both sources for the LLM
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
    
    # Step 4: Invoke LLM with combined context
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
    
    # Step 5: Add source attribution to response
    attribution = f"\n\n---\n**Sources Used:**\n- SQL Database: {sql_data['record_count']} records retrieved\n- RAG Corpus: Document context extracted\n- Method: Dual-source cross-validation for hallucination prevention"
    
    return response.content + attribution