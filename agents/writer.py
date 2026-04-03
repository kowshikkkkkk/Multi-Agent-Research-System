from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.6,
    max_tokens=2000
)

def writer_agent(analysis: str, original_query: str) -> str:
    messages = [
        SystemMessage(content="""You are a professional report writing agent.
        You will be given an analysis and the original research question.
        Your job is to produce a clean, structured, professional report with:
        - An executive summary (2-3 sentences)
        - Key findings section
        - Risk assessment section  
        - Conclusions and recommendations
        Write clearly for a business audience. 
        Only use information provided in the analysis.
        Do not add any new facts or data."""),
        HumanMessage(content=f"Original Question:\n{original_query}\n\nAnalysis:\n{analysis}")
    ]
    
    response = llm.invoke(messages)
    return response.content