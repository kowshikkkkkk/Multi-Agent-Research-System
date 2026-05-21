# evaluate.py - Custom RAG Evaluation Suite
# No RAGAS dependency — built from scratch

import os
import time
import json
import mlflow
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── Config ─────────────────────────────────────────────────
CHROMA_PATH = "chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Test Cases ──────────────────────────────────────────────
TEST_CASES = [
    {
        "question": "What was Infosys total revenue in fiscal year 2024?",
        "ground_truth": "Infosys total revenue from operations was 1,53,670 crore rupees for fiscal year 2024.",
        "keywords": ["revenue", "crore", "operations", "2024"]
    },
    {
        "question": "What is Infosys operating margin?",
        "ground_truth": "Infosys operating margin was around 20 to 21 percent in fiscal year 2024.",
        "keywords": ["margin", "operating", "percent"]
    },
    {
        "question": "What are the key business segments of Infosys?",
        "ground_truth": "Infosys key segments include Financial Services, Manufacturing, Retail, Energy, Communication and Hi-Tech.",
        "keywords": ["financial", "manufacturing", "retail", "segment", "energy"]
    },
    {
        "question": "How many employees does Infosys have?",
        "ground_truth": "Infosys had approximately 317,000 to 320,000 employees in fiscal year 2024.",
        "keywords": ["employees", "317", "320", "headcount", "workforce"]
    },
    {
        "question": "What is Infosys cloud strategy?",
        "ground_truth": "Infosys cloud strategy is driven by Infosys Cobalt offering cloud services to enterprise clients.",
        "keywords": ["cloud", "cobalt", "strategy", "enterprise", "services"]
    }
]

# ── Setup ───────────────────────────────────────────────────
def setup():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=800
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return llm, vector_store

# ── Metric 1: Retrieval Relevance ───────────────────────────
def score_retrieval_relevance(question: str, chunks: list, keywords: list) -> float:
    """
    Checks how many keywords from the question appear in retrieved chunks.
    Score = keywords found in chunks / total keywords
    """
    combined_chunks = " ".join(chunks).lower()
    found = sum(1 for kw in keywords if kw.lower() in combined_chunks)
    score = found / len(keywords)
    return round(score, 3)

# ── Metric 2: Faithfulness ──────────────────────────────────
def score_faithfulness(answer: str, chunks: list, llm) -> float:
    """
    Asks LLM: is every claim in the answer supported by the chunks?
    Returns score 0.0 to 1.0
    """
    context = "\n\n".join(chunks)
    
    prompt = f"""You are an evaluation judge. 

Given the following retrieved context and an answer, score how faithful the answer is to the context.

Faithful means: every claim in the answer can be traced back to the context. No made-up facts.

Context:
{context}

Answer:
{answer}

Score the faithfulness from 0.0 to 1.0 where:
1.0 = every claim is supported by context
0.5 = some claims supported, some not
0.0 = answer contradicts or ignores context

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    messages = [
        SystemMessage(content="You are a strict evaluation judge. Respond only with a decimal number."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5
    
    return round(score, 3)

# ── Metric 3: Answer Completeness ──────────────────────────
def score_completeness(question: str, answer: str, ground_truth: str, llm) -> float:
    """
    Asks LLM: does the answer cover what ground truth says?
    Returns score 0.0 to 1.0
    """
    prompt = f"""You are an evaluation judge.

Compare the generated answer to the ground truth answer.
Score how complete the generated answer is — does it cover the key information?

Question: {question}

Ground Truth: {ground_truth}

Generated Answer: {answer}

Score completeness from 0.0 to 1.0 where:
1.0 = answer covers all key information from ground truth
0.5 = answer covers some key information
0.0 = answer misses most key information

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    messages = [
        SystemMessage(content="You are a strict evaluation judge. Respond only with a decimal number."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5
    
    return round(score, 3)

# ── Metric 4: Latency ───────────────────────────────────────
def measure_rag_latency(question: str, vector_store, llm) -> dict:
    """
    Measures retrieval time and generation time separately.
    """
    # Retrieval latency
    t1 = time.time()
    results = vector_store.similarity_search(question, k=5)
    retrieval_time = round(time.time() - t1, 3)
    
    chunks = [doc.page_content for doc in results]
    context = "\n\n".join(chunks)
    
    # Generation latency
    t2 = time.time()
    prompt = f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = llm.invoke(prompt)
    generation_time = round(time.time() - t2, 3)
    
    return {
        "answer": response.content,
        "chunks": chunks,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": round(retrieval_time + generation_time, 3)
    }

# ── Main Evaluation ─────────────────────────────────────────
def run_evaluation():
    print("\n" + "="*60)
    print("CUSTOM RAG EVALUATION SUITE")
    print("Multi-Agent Research System")
    print("="*60)

    llm, vector_store = setup()
    
    all_results = []
    
    print(f"\nRunning {len(TEST_CASES)} test cases...\n")
    
    for i, test in enumerate(TEST_CASES):
        print(f"Test {i+1}/{len(TEST_CASES)}: {test['question'][:55]}...")
        
        # Run RAG + measure latency
        rag_output = measure_rag_latency(test["question"], vector_store, llm)
        
        # Score retrieval relevance
        retrieval_score = score_retrieval_relevance(
            test["question"],
            rag_output["chunks"],
            test["keywords"]
        )
        
        # Score faithfulness
        faith_score = score_faithfulness(
            rag_output["answer"],
            rag_output["chunks"],
            llm
        )
        
        # Score completeness
        complete_score = score_completeness(
            test["question"],
            rag_output["answer"],
            test["ground_truth"],
            llm
        )
        
        overall = round((retrieval_score + faith_score + complete_score) / 3, 3)
        
        result = {
            "question": test["question"],
            "retrieval_relevance": retrieval_score,
            "faithfulness": faith_score,
            "completeness": complete_score,
            "overall": overall,
            "retrieval_time": rag_output["retrieval_time"],
            "generation_time": rag_output["generation_time"],
            "total_time": rag_output["total_time"]
        }
        
        all_results.append(result)
        print(f"  Retrieval: {retrieval_score:.2f} | Faithfulness: {faith_score:.2f} | Completeness: {complete_score:.2f} | Overall: {overall:.2f}")
    
    # ── Aggregate Scores ────────────────────────────────────
    avg_retrieval   = round(sum(r["retrieval_relevance"] for r in all_results) / len(all_results), 3)
    avg_faith       = round(sum(r["faithfulness"] for r in all_results) / len(all_results), 3)
    avg_complete    = round(sum(r["completeness"] for r in all_results) / len(all_results), 3)
    avg_overall     = round(sum(r["overall"] for r in all_results) / len(all_results), 3)
    avg_latency     = round(sum(r["total_time"] for r in all_results) / len(all_results), 3)
    
    # ── MLflow Logging ──────────────────────────────────────
    print("\nLogging to MLflow...")
    mlflow.set_experiment("RAG_Evaluation")
    
    with mlflow.start_run(run_name="custom_rag_eval"):
        # Params
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("llm_model", "llama-3.3-70b-versatile")
        mlflow.log_param("chunk_size", 1000)
        mlflow.log_param("chunk_overlap", 200)
        mlflow.log_param("top_k", 3)
        mlflow.log_param("num_test_cases", len(TEST_CASES))
        
        # Aggregate metrics
        mlflow.log_metric("avg_retrieval_relevance", avg_retrieval)
        mlflow.log_metric("avg_faithfulness", avg_faith)
        mlflow.log_metric("avg_completeness", avg_complete)
        mlflow.log_metric("avg_overall_score", avg_overall)
        mlflow.log_metric("avg_latency_seconds", avg_latency)
        
        # Per question metrics
        for i, r in enumerate(all_results):
            mlflow.log_metric(f"q{i+1}_retrieval", r["retrieval_relevance"])
            mlflow.log_metric(f"q{i+1}_faithfulness", r["faithfulness"])
            mlflow.log_metric(f"q{i+1}_completeness", r["completeness"])
            mlflow.log_metric(f"q{i+1}_latency", r["total_time"])
        
        # Save detailed results
        with open("evaluation_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        mlflow.log_artifact("evaluation_results.json")
    
    # ── Print Summary ───────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"{'Metric':<28} {'Score':<10} {'Meaning'}")
    print("-"*60)
    print(f"{'Retrieval Relevance':<28} {avg_retrieval:<10} Keywords found in chunks")
    print(f"{'Faithfulness':<28} {avg_faith:<10} Answer grounded in context")
    print(f"{'Completeness':<28} {avg_complete:<10} Answer covers ground truth")
    print("-"*60)
    print(f"{'OVERALL SCORE':<28} {avg_overall:<10}")
    print(f"{'AVG LATENCY':<28} {avg_latency}s")
    print("="*60)
    
    if avg_overall >= 0.8:
        print("✅ EXCELLENT — RAG pipeline performing well")
    elif avg_overall >= 0.6:
        print("⚠️  GOOD — Some room for improvement")
    elif avg_overall >= 0.4:
        print("🔶 MODERATE — Consider tuning chunk size or retrieval")
    else:
        print("❌ POOR — RAG pipeline needs improvement")
    
    print("\nDetailed results saved to: evaluation_results.json")
    print("View MLflow dashboard: mlflow ui")
    
    return all_results

if __name__ == "__main__":
    run_evaluation()