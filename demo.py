"""
Demo: Complete Multi-Agent Pipeline with Dual-Source Researcher
Shows all 5 agents working together with SQL + RAG retrieval
"""

from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.analyst import analyst_agent
from agents.writer import writer_agent
from agents.critic import critic_agent
import time


def run_pipeline(user_query: str, max_revisions: int = 2):
    """
    Execute the complete multi-agent research pipeline.
    
    Flow:
    Planner → Researcher (SQL+RAG) → Analyst → Writer → Critic → [Loop if needed]
    
    Args:
        user_query: User's research question
        max_revisions: Maximum revision loops (prevents infinite loops)
    
    Returns:
        dict: Final report and metadata
    """
    
    print("\n" + "="*80)
    print("MULTI-AGENT RESEARCH PIPELINE WITH DUAL-SOURCE RETRIEVAL")
    print("="*80)
    print(f"\nUser Query: {user_query}\n")
    
    revision_count = 0
    state = {
        "query": user_query,
        "plan": "",
        "research": "",
        "analysis": "",
        "report": "",
        "critique": "",
        "revision_count": 0
    }
    
    while True:
        # ────────────────────────────────────────────────────────────────
        # STEP 1: PLANNER
        # ────────────────────────────────────────────────────────────────
        print("\n" + "="*80)
        print("STEP 1: PLANNER AGENT")
        print("="*80)
        t = time.time()
        
        state["plan"] = planner_agent(state["query"])
        
        elapsed = time.time() - t
        print(f"\n✓ Plan created in {elapsed:.2f}s")
        print(f"Plan preview: {state['plan'][:200]}...\n")
        
        # ────────────────────────────────────────────────────────────────
        # STEP 2: RESEARCHER (DUAL-SOURCE: SQL + RAG)
        # ────────────────────────────────────────────────────────────────
        print("\n" + "="*80)
        print("STEP 2: RESEARCHER AGENT (DUAL-SOURCE: SQL + RAG)")
        print("="*80)
        t = time.time()
        
        state["research"] = researcher_agent(state["plan"])
        
        elapsed = time.time() - t
        print(f"\n✓ Research completed in {elapsed:.2f}s")
        print(f"Research preview: {state['research'][:300]}...\n")
        
        # ────────────────────────────────────────────────────────────────
        # STEP 3: ANALYST
        # ────────────────────────────────────────────────────────────────
        print("\n" + "="*80)
        print("STEP 3: ANALYST AGENT")
        print("="*80)
        t = time.time()
        
        state["analysis"] = analyst_agent(state["research"])
        
        elapsed = time.time() - t
        print(f"\n✓ Analysis completed in {elapsed:.2f}s")
        print(f"Analysis preview: {state['analysis'][:300]}...\n")
        
        # ────────────────────────────────────────────────────────────────
        # STEP 4: WRITER
        # ────────────────────────────────────────────────────────────────
        print("\n" + "="*80)
        print("STEP 4: WRITER AGENT")
        print("="*80)
        t = time.time()
        
        state["report"] = writer_agent(state["analysis"], state["query"])
        
        elapsed = time.time() - t
        print(f"\n✓ Report written in {elapsed:.2f}s")
        print(f"Report preview: {state['report'][:300]}...\n")
        
        # ────────────────────────────────────────────────────────────────
        # STEP 5: CRITIC
        # ────────────────────────────────────────────────────────────────
        print("\n" + "="*80)
        print("STEP 5: CRITIC AGENT (Quality Control)")
        print("="*80)
        t = time.time()
        
        state["critique"] = critic_agent(state["report"], state["query"])
        state["revision_count"] += 1
        
        elapsed = time.time() - t
        print(f"\n✓ Critique completed in {elapsed:.2f}s")
        print(f"Critique: {state['critique'][:300]}...\n")
        
        # ────────────────────────────────────────────────────────────────
        # DECISION: APPROVED or NEEDS REVISION?
        # ────────────────────────────────────────────────────────────────
        if "APPROVED" in state["critique"]:
            print("\n" + "="*80)
            print("✅ REPORT APPROVED")
            print("="*80)
            break
        elif state["revision_count"] >= max_revisions:
            print("\n" + "="*80)
            print(f"⚠️  MAX REVISIONS REACHED ({max_revisions})")
            print("="*80)
            print("Delivering best effort report...\n")
            break
        else:
            print("\n" + "="*80)
            print(f"🔄 REVISION NEEDED (Attempt {state['revision_count']}/{max_revisions})")
            print("="*80)
            print("Looping back to Researcher for more data...\n")
            # Continue loop
    
    # ────────────────────────────────────────────────────────────────
    # FINAL OUTPUT
    # ────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    
    print("\n" + "─"*80)
    print("FINAL REPORT")
    print("─"*80)
    print(state["report"])
    
    print("\n" + "─"*80)
    print("EXECUTION SUMMARY")
    print("─"*80)
    print(f"Revisions: {state['revision_count']}")
    print(f"Verdict: {'APPROVED' if 'APPROVED' in state['critique'] else 'BEST EFFORT'}")
    print(f"Report Length: {len(state['report'])} characters")
    print(f"Sources Used: SQL Database + RAG Corpus (Dual-Source Retrieval)")
    
    return state


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example queries
    queries = [
        "Analyze Infosys's 2024 financial performance and revenue growth trends",
        "Evaluate segment performance and identify the fastest-growing business lines",
        "Assess profitability metrics and operational efficiency improvements"
    ]
    
    # Run pipeline on first query
    result = run_pipeline(queries[0])
    
    print("\n" + "="*80)
    print("Demo completed! Pipeline is working with dual-source retrieval.")
    print("="*80)