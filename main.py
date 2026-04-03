from agents.planner import planner_agent

query = "Analyze the financial risks of Infosys based on their 2024 annual report"

print("=== PLANNER AGENT ===")
result = planner_agent(query)
print(result)

from utils.rag import build_vector_store

print("Building vector store from Infosys 2024 Annual Report...")
build_vector_store()

from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.analyst import analyst_agent
from agents.writer import writer_agent
from agents.critic import critic_agent

query = "Analyze the financial risks of Infosys based on their 2024 annual report"

print("=== PLANNER AGENT ===")
plan = planner_agent(query)
print(plan)

print("\n=== RESEARCHER AGENT ===")
research = researcher_agent(plan)
print(research)

print("\n=== ANALYST AGENT ===")
analysis = analyst_agent(research)
print(analysis)

print("\n=== WRITER AGENT ===")
report = writer_agent(analysis, query)
print(report)

print("\n=== CRITIC AGENT ===")
critique = critic_agent(report, query)
print(critique)

from graph import build_graph

query = "Analyze the financial risks of Infosys based on their 2024 annual report"

app = build_graph(query)

initial_state = {
    "query": query,
    "plan": "",
    "research": "",
    "analysis": "",
    "report": "",
    "critique": "",
    "revision_count": 0
}

print("🚀 Starting Multi-Agent Research Pipeline...")
final_state = app.invoke(initial_state)
print("\n✅ Pipeline Complete!")