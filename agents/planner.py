from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=500
)

def planner_agent(user_query: str) -> str:
    messages = [
        SystemMessage(content="""You are a planning agent. 
        Your only job is to break the user's research question into 3-5 clear subtasks.
        Do not answer the question yourself.
        Output ONLY a numbered list of subtasks. Nothing else."""),
        HumanMessage(content=user_query)
    ]
    
    response = llm.invoke(messages)
    return response.content