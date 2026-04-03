from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=1200
)

def analyst_agent(research_findings: str) -> str:
    messages = [
        SystemMessage(content="""You are a financial analyst agent.
        You will be given research findings from a real document.
        Your job is to:
        1. Identify patterns and connections across the findings
        2. Assess the severity of each risk identified
        3. Draw meaningful conclusions based only on the provided findings
        4. Highlight what is concerning and what is reassuring
        Do not make up any data. Only reason over what is provided."""),
        HumanMessage(content=f"Research Findings:\n{research_findings}")
    ]
    
    response = llm.invoke(messages)
    return response.content