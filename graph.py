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
    revision_history: list  # stores all previous reports

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
    
    # Determine retrieval strategy based on revision
    retrieval_config = {
        "k": 5,
        "keywords_to_emphasize": []
    }
    
    if state["revision_count"] > 0 and state["critique"]:
        # PARSE critique to adjust strategy
        critique_analysis = extract_missing_from_critique(state["critique"])
        retrieval_config["k"] = critique_analysis["adjusted_k"]
        retrieval_config["keywords_to_emphasize"] = critique_analysis["keywords"]
        
        enhanced_plan = f"""
Original Plan: {state["plan"]}

REVISION #{state["revision_count"]} REQUIRED.
Critic Feedback: {state["critique"]}

ADJUSTED RETRIEVAL:
- Searching with emphasis on: {', '.join(retrieval_config['keywords_to_emphasize'])}
- Retrieving {retrieval_config['k']} chunks (expanded from 5)
- Focus specifically on addressing the critique above.
"""
    else:
        enhanced_plan = state["plan"]
    
    # Pass config to researcher_agent
    research = researcher_agent(
        enhanced_plan,
        retrieval_config=retrieval_config
    )
    print(research)
    log_agent("researcher", enhanced_plan, research, t)
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
    
    # Store current report in history before potentially overwriting
    history = state.get("revision_history", [])
    history.append({ 
        "revision": state["revision_count"],
        "report": state["report"],
        "critique": critique,
        "approved": approved
    })
    
    return {
        "critique": critique,
        "revision_count": state["revision_count"] + 1,
        "revision_history": history
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

def extract_missing_from_critique(critique: str) -> dict:
    """Parse critique to identify what's missing"""
    missing_keywords = []
    adjusted_k = 5  # default
    
    # Common patterns in critiques
    if any(word in critique.lower() for word in ["missing", "lacks", "incomplete", "insufficient"]):
        adjusted_k = 8  # Get more chunks
        missing_keywords.append("comprehensive analysis")
    
    if "risk" in critique.lower() and "risk" not in missing_keywords:
        missing_keywords.append("risk assessment challenges threats")
    
    if "financial" in critique.lower() and "financial" not in missing_keywords:
        missing_keywords.append("revenue margins profitability")
    
    if "strategy" in critique.lower() and "strategy" not in missing_keywords:
        missing_keywords.append("strategy future outlook plans")
    
    if "competitive" in critique.lower():
        missing_keywords.append("competition market position")
    
    return {
        "keywords": missing_keywords,
        "adjusted_k": adjusted_k
    }


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