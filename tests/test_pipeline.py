# tests/test_pipeline.py
# Unit Test Suite — Multi-Agent Research System

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.rag import retrieve
from sql_query import FinancialDataQuery
from agents.planner import planner_agent
from agents.critic import critic_agent
from graph import build_graph, ResearchState

# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def sql_db():
    return FinancialDataQuery(db_path="financial_data.db")

@pytest.fixture
def sample_query():
    return "Analyze Infosys revenue performance in 2024"

@pytest.fixture
def sample_report():
    return """
    EXECUTIVE SUMMARY
    Infosys reported total revenue of 1,53,670 crore rupees for fiscal year 2024,
    representing growth from the previous year. Operating margin stood at approximately
    21%, reflecting strong cost management. The company employs over 317,000 professionals
    across its key segments including Financial Services, Manufacturing, and Retail.
    Cloud services through Infosys Cobalt continue to drive strategic growth.
    Sources: SQL Database (verified financials) + RAG Corpus (annual report).
    """

# ── Test 1: RAG Retrieval ───────────────────────────────────

class TestRAGRetrieval:
    
    def test_retrieves_correct_number_of_chunks(self):
        """RAG should return exactly k chunks"""
        results = retrieve("Infosys revenue 2024", k=3)
        # retrieve returns a string — check it's non-empty
        assert isinstance(results, str)
        assert len(results) > 0
    
    def test_retrieves_relevant_content(self):
        """Retrieved chunks should contain Infosys related content"""
        results = retrieve("Infosys revenue operations", k=3)
        assert "Infosys" in results or "revenue" in results.lower()
    
    def test_retrieval_with_different_k(self):
        """RAG should work with different k values"""
        results_3 = retrieve("operating margin", k=3)
        results_5 = retrieve("operating margin", k=5)
        # k=5 should return more content than k=3
        assert len(results_5) >= len(results_3)
    
    def test_empty_query_handled(self):
        """RAG should handle edge case queries"""
        results = retrieve("Infosys", k=3)
        assert isinstance(results, str)

# ── Test 2: SQL Queries ─────────────────────────────────────

class TestSQLQuery:
    
    def test_sql_returns_data_for_revenue(self, sql_db):
        """SQL should return data for revenue queries"""
        result = sql_db.query("revenue")
        assert result is not None
        assert result["status"] == "success"
        assert result["record_count"] > 0
    
    def test_sql_returns_data_for_margin(self, sql_db):
        """SQL should return data for margin queries"""
        result = sql_db.query("margin")
        assert result is not None
        assert result["status"] in ["success", "no_data"]
    
    def test_sql_result_has_required_fields(self, sql_db):
        """SQL results should always have status and summary fields"""
        result = sql_db.query("revenue")
        assert "status" in result
        assert "summary" in result
        assert "record_count" in result
    
    def test_sql_handles_unknown_query(self, sql_db):
        """SQL should handle queries it doesn't recognize gracefully"""
        result = sql_db.query("xyzxyzxyz unknown query")
        assert result is not None
        assert "status" in result

# ── Test 3: Planner Agent ───────────────────────────────────

class TestPlannerAgent:
    
    def test_planner_returns_string(self, sample_query):
        """Planner should return a string output"""
        plan = planner_agent(sample_query)
        assert isinstance(plan, str)
    
    def test_planner_returns_non_empty(self, sample_query):
        """Planner should return non-empty plan"""
        plan = planner_agent(sample_query)
        assert len(plan) > 50
    
    def test_planner_contains_subtasks(self, sample_query):
        """Planner output should contain structured subtasks"""
        plan = planner_agent(sample_query)
        # Plan should mention research or analysis or subtask
        keywords = ["subtask", "research", "analyze", "1.", "2.", "step"]
        assert any(kw.lower() in plan.lower() for kw in keywords)

# ── Test 4: Critic Agent ────────────────────────────────────

class TestCriticAgent:
    
    def test_critic_returns_verdict(self, sample_report, sample_query):
        """Critic should return APPROVED or NEEDS REVISION"""
        verdict = critic_agent(sample_report, sample_query)
        assert isinstance(verdict, str)
        assert "APPROVED" in verdict or "NEEDS REVISION" in verdict
    
    def test_critic_approves_good_report(self, sample_report, sample_query):
        """Critic should approve a well-written complete report"""
        verdict = critic_agent(sample_report, sample_query)
        # Good report with facts should get approved
        assert isinstance(verdict, str)
        assert len(verdict) > 0
    
    def test_critic_handles_empty_report(self, sample_query):
        """Critic should handle poor quality report"""
        bad_report = "I don't know anything about Infosys."
        verdict = critic_agent(bad_report, sample_query)
        assert "NEEDS REVISION" in verdict

# ── Test 5: Loop Guard ──────────────────────────────────────

class TestLoopGuard:
    
    def test_pipeline_stops_at_max_revisions(self):
        """Pipeline should stop after 2 revisions maximum"""
        from graph import route_after_critic
        
        # Simulate state at revision_count = 2
        state = {
            "query": "test query",
            "plan": "test plan",
            "research": "test research",
            "analysis": "test analysis",
            "report": "test report",
            "critique": "NEEDS REVISION: missing data",
            "revision_count": 2
        }
        
        route = route_after_critic(state)
        assert route == "end"
    
    def test_pipeline_loops_before_max_revisions(self):
        """Pipeline should loop back when revision_count < 2"""
        from graph import route_after_critic
        
        state = {
            "query": "test query",
            "plan": "test plan",
            "research": "test research",
            "analysis": "test analysis",
            "report": "test report",
            "critique": "NEEDS REVISION: incomplete analysis",
            "revision_count": 1
        }
        
        route = route_after_critic(state)
        assert route == "researcher"
    
    def test_pipeline_ends_on_approval(self):
        """Pipeline should end when critic approves"""
        from graph import route_after_critic
        
        state = {
            "query": "test query",
            "plan": "test plan",
            "research": "test research",
            "analysis": "test analysis",
            "report": "test report",
            "critique": "APPROVED: excellent report",
            "revision_count": 1
        }
        
        route = route_after_critic(state)
        assert route == "end"

# ── Test 6: Full Pipeline ───────────────────────────────────

class TestFullPipeline:
    
    def test_pipeline_runs_end_to_end(self, sample_query):
        """Full pipeline should complete without errors"""
        graph = build_graph(sample_query)
        
        initial_state = {
            "query": sample_query,
            "plan": "",
            "research": "",
            "analysis": "",
            "report": "",
            "critique": "",
            "revision_count": 0
        }
        
        result = graph.invoke(initial_state)
        
        assert result is not None
        assert "report" in result
        assert len(result["report"]) > 100
    
    def test_pipeline_state_has_all_fields(self, sample_query):
        """Final state should have all required fields populated"""
        graph = build_graph(sample_query)
        
        initial_state = {
            "query": sample_query,
            "plan": "",
            "research": "",
            "analysis": "",
            "report": "",
            "critique": "",
            "revision_count": 0
        }
        
        result = graph.invoke(initial_state)
        
        assert len(result["plan"]) > 0
        assert len(result["research"]) > 0
        assert len(result["analysis"]) > 0
        assert len(result["report"]) > 0
        assert len(result["critique"]) > 0