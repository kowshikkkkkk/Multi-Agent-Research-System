from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.analyst import analyst_agent
from agents.writer import writer_agent
from agents.critic import critic_agent
from utils.logger import start_run, log_agent, end_run
import time

class ResearchState(TypedDict):
    query: str
    plan: str
    research: str
    analysis: str
    report: str
    critique: str
    revision_count: int

def planner_node(state: ResearchState) -> ResearchState:
    print("\n=== PLANNER AGENT ===")
    t = time.time()
    plan = planner_agent(state["query"])
    print(plan)
    log_agent("planner", state["query"], plan, t)
    return {"plan": plan}

def researcher_node(state: ResearchState) -> ResearchState:
    print("\n=== RESEARCHER AGENT ===")
    t = time.time()
    research = researcher_agent(state["plan"])
    print(research)
    log_agent("researcher", state["plan"], research, t)
    return {"research": research}

def analyst_node(state: ResearchState) -> ResearchState:
    print("\n=== ANALYST AGENT ===")
    t = time.time()
    analysis = analyst_agent(state["research"])
    print(analysis)
    log_agent("analyst", state["research"], analysis, t)
    return {"analysis": analysis}

def writer_node(state: ResearchState) -> ResearchState:
    print("\n=== WRITER AGENT ===")
    t = time.time()
    report = writer_agent(state["analysis"], state["query"])
    print(report)
    log_agent("writer", state["analysis"], report, t)
    return {"report": report}

def critic_node(state: ResearchState) -> ResearchState:
    print("\n=== CRITIC AGENT ===")
    t = time.time()
    critique = critic_agent(state["report"], state["query"])
    print(critique)
    log_agent("critic", state["report"], critique, t)
    approved = "APPROVED" in critique
    end_run(
        final_score="6/10" if not approved else "8+/10",
        approved=approved
    )
    return {
        "critique": critique,
        "revision_count": state["revision_count"] + 1
    }

def route_after_critic(state: ResearchState) -> str:
    if "APPROVED" in state["critique"]:
        print("\n✅ Report APPROVED — pipeline complete!")
        return "end"
    elif state["revision_count"] >= 2:
        print("\n⚠️ Max revisions reached — delivering best report.")
        return "end"
    else:
        print("\n🔄 NEEDS REVISION — looping back to researcher...")
        return "researcher"

def build_graph(query: str):
    start_run(query)
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher": "researcher",
            "end": END
        }
    )
    return graph.compile()