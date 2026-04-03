import mlflow
import time
from datetime import datetime

def start_run(query: str):
    mlflow.set_experiment("multi-agent-research-system")
    run = mlflow.start_run(run_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    mlflow.log_param("query", query)
    return run

def log_agent(agent_name: str, input_text: str, output_text: str, start_time: float):
    elapsed = time.time() - start_time
    mlflow.log_metric(f"{agent_name}_latency_seconds", round(elapsed, 2))
    mlflow.log_text(input_text, f"{agent_name}_input.txt")
    mlflow.log_text(output_text, f"{agent_name}_output.txt")
    print(f"📊 {agent_name} logged — latency: {round(elapsed, 2)}s")

def end_run(final_score: str, approved: bool):
    mlflow.log_param("final_score", final_score)
    mlflow.log_param("approved", approved)
    mlflow.end_run()
    print("📊 MLflow run complete — check UI with: mlflow ui")