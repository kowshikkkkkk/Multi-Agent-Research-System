from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=600
)

def critic_agent(report: str, original_query: str) -> str:
    messages = [
        SystemMessage(content="""You are a quality control agent.
        You will be given a research report and the original question.
        Your job is to:
        1. Check if the report fully answers the original question
        2. Identify any gaps or missing information
        3. Flag any claims that seem unsupported
        4. Give an overall quality score out of 10
        5. State clearly: APPROVED or NEEDS REVISION
        Be strict but fair."""),
        HumanMessage(content=f"Original Question:\n{original_query}\n\nReport:\n{report}")
    ]
    
    response = llm.invoke(messages)
    return response.content