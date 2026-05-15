import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, List, Any

class FinancialDataQuery:
    """
    Wrapper around SQLite financial database.
    Safe queries: predefined question→SQL mapping, prevents SQL injection.
    """
    
    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = db_path
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found at {db_path}. Run create_db.py first.")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Parse a natural language question and return structured SQL results.
        Returns: {"status": "success"|"error", "query": "...", "results": [...], "summary": "..."}
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            q_lower = question.lower()
            results = []
            summary = ""
            sql_query = ""
            
            # Intent 1: Revenue growth/trends
            if any(x in q_lower for x in ["revenue", "growth", "trend", "performance"]):
                sql_query = '''
                    SELECT company, quarter, year, revenue_usd_millions, employee_count
                    FROM quarterly_data
                    WHERE company = 'Infosys'
                    ORDER BY year DESC, 
                             CASE WHEN quarter='Q4' THEN 4 WHEN quarter='Q3' THEN 3 
                                  WHEN quarter='Q2' THEN 2 ELSE 1 END DESC
                '''
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                results = [dict(r) for r in rows]
                
                if results:
                    q1_rev = results[-1]['revenue_usd_millions']
                    latest_rev = results[0]['revenue_usd_millions']
                    growth = ((latest_rev - q1_rev) / q1_rev * 100) if q1_rev else 0
                    summary = f"Infosys Q1→Q3 2024 revenue trend: ${q1_rev:.1f}M → ${latest_rev:.1f}M ({growth:+.1f}% growth)"
            
            # Intent 2: Segment performance
            elif any(x in q_lower for x in ["segment", "business line", "line of business", "vertical"]):
                sql_query = '''
                    SELECT segment_name, revenue_usd_millions, growth_percent
                    FROM segment_revenue
                    WHERE company = 'Infosys' AND year = 2024
                    ORDER BY revenue_usd_millions DESC
                '''
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                results = [dict(r) for r in rows]
                
                if results:
                    top_segment = results[0]
                    summary = f"Top segment: {top_segment['segment_name']} at ${top_segment['revenue_usd_millions']:.1f}M"
            
            # Intent 3: Key metrics
            elif any(x in q_lower for x in ["margin", "profitability", "metric", "attrition", "roi"]):
                sql_query = '''
                    SELECT metric_name, value, unit
                    FROM key_metrics
                    WHERE company = 'Infosys' AND year = 2024
                    ORDER BY metric_name
                '''
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                results = [dict(r) for r in rows]
                
                if results:
                    summary = f"Retrieved {len(results)} key metrics (operating margin, net margin, ROE, etc.)"
            
            # Intent 4: Employee count
            elif any(x in q_lower for x in ["employee", "headcount", "staff", "workforce"]):
                sql_query = '''
                    SELECT quarter, year, employee_count
                    FROM quarterly_data
                    WHERE company = 'Infosys'
                    ORDER BY year DESC, 
                             CASE WHEN quarter='Q4' THEN 4 WHEN quarter='Q3' THEN 3 
                                  WHEN quarter='Q2' THEN 2 ELSE 1 END DESC
                    LIMIT 4
                '''
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                results = [dict(r) for r in rows]
                
                if results:
                    latest = results[0]
                    summary = f"Latest headcount: {latest['employee_count']:,} employees ({latest['quarter']} {latest['year']})"
            
            # Fallback
            else:
                sql_query = "SELECT company, quarter, year, revenue_usd_millions FROM quarterly_data LIMIT 5"
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                results = [dict(r) for r in rows]
                summary = f"Returned {len(results)} quarterly records (default fallback)"
            
            return {
                "status": "success",
                "query": sql_query.strip(),
                "results": results,
                "summary": summary,
                "record_count": len(results)
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": sql_query if sql_query else "N/A"
            }
        
        finally:
            conn.close()


if __name__ == "__main__":
    db = FinancialDataQuery()
    
    print("Testing SQL Query Module\n" + "="*60)
    
    sample_questions = [
        "What is Infosys's revenue growth trend?",
        "Tell me about business segments and their performance",
        "What are the key profitability metrics?",
        "How many employees work at Infosys?",
    ]
    
    for question in sample_questions:
        print(f"\nQuestion: {question}")
        result = db.query(question)
        print(f"Status: {result['status']}")
        print(f"Summary: {result['summary']}")
        print(f"Records returned: {result['record_count']}")
        if result['status'] == 'success' and result['results']:
            print(f"Sample result: {result['results'][0]}")