# cross_validate.py
# Programmatic SQL-RAG Cross Validation
# Replaces LLM-as-judge for numerical fact checking

import re
from sql_query import FinancialDataQuery

sql_db = FinancialDataQuery(db_path="financial_data.db")

# ── Number Extractor ────────────────────────────────────────

def extract_numbers(text: str) -> list:
    pattern = r'\d+(?:,\d{3})*(?:\.\d+)?%?'
    numbers = re.findall(pattern, text)
    cleaned = [n.replace(',', '') for n in numbers]
    # Filter out small integers — years, quarters, page numbers
    filtered = []
    for n in cleaned:
        try:
            val = float(n.replace('%', ''))
            # Keep percentages and numbers > 100
            if '%' in n or val > 100:
                filtered.append(n)
        except:
            continue
    return filtered

# ── Tolerance Checker ───────────────────────────────────────

def numbers_match(num1: str, num2: str, tolerance: float = 0.05) -> bool:
    """
    Checks if two numbers match within a tolerance.
    Handles percentages and regular numbers.
    tolerance=0.05 means 5% difference is acceptable.
    """
    try:
        # Remove % sign for comparison
        n1 = float(num1.replace('%', ''))
        n2 = float(num2.replace('%', ''))

        # Avoid division by zero
        if n2 == 0:
            return n1 == 0

        difference = abs(n1 - n2) / abs(n2)
        return difference <= tolerance

    except ValueError:
        return False

# ── Core Cross Validator ────────────────────────────────────

def cross_validate(query: str, rag_context: str) -> dict:
    """
    Validates RAG retrieved content against SQL verified facts.
    
    Returns:
        {
            "status": "CONSISTENT" | "CONTRADICTION" | "UNVERIFIABLE",
            "confidence": "HIGH" | "MEDIUM" | "LOW",
            "verified_facts": [...],
            "contradictions": [...],
            "unverified_numbers": [...],
            "summary": "..."
        }
    """
    # Step 1: Get SQL ground truth
    sql_result = sql_db.query(query)
    
    if sql_result["status"] != "success" or not sql_result["results"]:
        return {
            "status": "UNVERIFIABLE",
            "confidence": "LOW",
            "verified_facts": [],
            "contradictions": [],
            "unverified_numbers": [],
            "summary": "No SQL ground truth available for this query type"
        }
    
    # Step 2: Extract numbers from RAG context
    rag_numbers = extract_numbers(rag_context)
    
    if not rag_numbers:
        return {
            "status": "UNVERIFIABLE",
            "confidence": "LOW",
            "verified_facts": [],
            "contradictions": [],
            "unverified_numbers": [],
            "summary": "No numbers found in RAG context to verify"
        }
    
    # Step 3: Extract SQL ground truth numbers
    SKIP_FIELDS = ["year", "quarter"]
    sql_numbers = []
    for row in sql_result["results"]:
        for key, value in row.items():
            if key in SKIP_FIELDS:
                continue
            if isinstance(value, (int, float)):
            # Skip small integers — likely IDs or quarters
               if isinstance(value, int) and value < 100:
                   continue
               sql_numbers.append({
                  "field": key,
                  "value": str(value)
            })
    
    # Step 4: Cross check RAG numbers against SQL numbers
    verified_facts = []
    contradictions = []
    unverified = []
    
    for rag_num in rag_numbers:
        matched = False
        for sql_fact in sql_numbers:
            if numbers_match(rag_num, sql_fact["value"]):
                verified_facts.append({
                    "rag_number": rag_num,
                    "sql_field": sql_fact["field"],
                    "sql_value": sql_fact["value"],
                    "status": "VERIFIED"
                })
                matched = True
                break
        
        if not matched:
            # Check if it's a clear contradiction
            # (close to a SQL value but outside tolerance)
            contradiction_found = False
            for sql_fact in sql_numbers:
                try:
                    rag_val = float(rag_num.replace('%', ''))
                    sql_val = float(sql_fact["value"])
                    
                    # If same order of magnitude but different value
                    # it's likely a contradiction not just unrelated
                    if sql_val != 0:
                        ratio = rag_val / sql_val
                        if 0.5 <= ratio <= 2.0 and not numbers_match(rag_num, sql_fact["value"]):
                            contradictions.append({
                                "rag_number": rag_num,
                                "sql_field": sql_fact["field"],
                                "sql_value": sql_fact["value"],
                                "status": "CONTRADICTION"
                            })
                            contradiction_found = True
                            break
                except:
                    continue
            
            if not contradiction_found:
                unverified.append(rag_num)
    
    # Step 5: Determine overall status
    if contradictions:
        status = "CONTRADICTION"
        confidence = "LOW"
    elif verified_facts:
        coverage = len(verified_facts) / len(rag_numbers)
        if coverage >= 0.6:
            status = "CONSISTENT"
            confidence = "HIGH"
        else:
            status = "CONSISTENT"
            confidence = "MEDIUM"
    else:
        status = "UNVERIFIABLE"
        confidence = "LOW"
    
    # Step 6: Build summary
    summary_parts = []
    summary_parts.append(f"RAG numbers found: {len(rag_numbers)}")
    summary_parts.append(f"SQL verified: {len(verified_facts)}")
    
    if contradictions:
        summary_parts.append(f"Contradictions: {len(contradictions)}")
        for c in contradictions:
            summary_parts.append(
                f"  ⚠️ RAG says {c['rag_number']} but SQL says {c['sql_value']} for {c['sql_field']}"
            )
    
    if verified_facts:
        for v in verified_facts[:3]:  # show top 3
            summary_parts.append(
                f"  ✓ {v['rag_number']} verified against SQL {v['sql_field']}"
            )
    
    return {
        "status": status,
        "confidence": confidence,
        "verified_facts": verified_facts,
        "contradictions": contradictions,
        "unverified_numbers": unverified,
        "summary": "\n".join(summary_parts),
        "sql_summary": sql_result["summary"]
    }

# ── Pretty Printer ──────────────────────────────────────────

def print_validation_report(result: dict):
    print("\n" + "="*60)
    print("CROSS-VALIDATION REPORT")
    print("="*60)
    print(f"Status:     {result['status']}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nSQL Ground Truth: {result.get('sql_summary', 'N/A')}")
    print(f"\nDetails:\n{result['summary']}")
    
    if result['contradictions']:
        print("\n⚠️  CONTRADICTIONS DETECTED:")
        for c in result['contradictions']:
            print(f"   RAG: {c['rag_number']} vs SQL: {c['sql_value']} ({c['sql_field']})")
    
    if result['unverified_numbers']:
        print(f"\nUnverified numbers: {result['unverified_numbers']}")
    print("="*60)

# ── Test Runner ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Cross-Validation Module")
    print("="*60)
    
    # Test 1: Consistent data
    print("\nTest 1: Consistent RAG context")
    rag_context_good = """
    Infosys reported quarterly revenue of 2,100.5 million USD in Q1 2024.
    The company showed growth with Q3 revenue reaching 2,180.8 million.
    Employee count stood at 317,240 professionals.
    """
    result = cross_validate("revenue growth trend", rag_context_good)
    print_validation_report(result)
    
    # Test 2: Contradicting data
    print("\nTest 2: Contradicting RAG context")
    rag_context_bad = """
    Infosys reported quarterly revenue of 2,800.0 million USD in Q1 2024.
    Operating margin was 35% showing exceptional performance.
    """
    result = cross_validate("revenue growth trend", rag_context_bad)
    print_validation_report(result)
    
    # Test 3: No numbers
    print("\nTest 3: No numbers in RAG context")
    rag_context_narrative = """
    Infosys has been investing heavily in cloud services and AI capabilities.
    The company continues to expand its presence in key markets globally.
    """
    result = cross_validate("revenue growth trend", rag_context_narrative)
    print_validation_report(result)