from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from utils.rag import retrieve
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=800
)

def researcher_agent(subtasks: str) -> str:
    print("Retrieving relevant content from document store...")
    context = retrieve(subtasks, k=3)
    
    messages = [
        SystemMessage(content="""You are a research agent.
        You will be given subtasks and relevant context retrieved from a real document.
        Summarize the key findings relevant to each subtask.
        Answer ONLY from the provided context. 
        If the context doesn't contain enough information, say 'Insufficient context.'
        Do not make up any numbers or facts."""),
        HumanMessage(content=f"Subtasks:\n{subtasks}\n\nContext:\n{context}")
    ]
    
    response = llm.invoke(messages)
    return response.content