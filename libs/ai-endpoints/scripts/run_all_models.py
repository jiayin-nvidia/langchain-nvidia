import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

SCRIPT_PATH = Path(__file__).resolve()
AI_ENDPOINTS_DIR = SCRIPT_PATH.parents[1]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = AI_ENDPOINTS_DIR / "test_results" / f"test_run_{timestamp}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHAT_RESULTS_CSV = RESULTS_DIR / "chat_models_results.csv"
EMBEDDING_RESULTS_CSV = RESULTS_DIR / "embedding_models_results.csv"
VLM_RESULTS_CSV = RESULTS_DIR / "vlm_models_results.csv"
COMPLETION_RESULTS_CSV = RESULTS_DIR / "completion_models_results.csv"

MODEL_TEST_CONFIG = {
    "chat": {
        "test_buckets": {
            "chat": (["test_chat_models.py", "test_streaming.py"], "--chat-model-id"),
            "tools": (["test_bind_tools.py"], "--tool-model-id"),
            "structured": (["test_structured_output.py"], "--structured-model-id"),
            "thinking": (["test_thinking_mode.py"], "--thinking-model-id"),
        },
        "csv_columns": ["Model", "Chat Error", "Tool Calling Error", "Structured Output Error", "Thinking Test"],
        "column_map": {
            "chat": "Chat Error",
            "tools": "Tool Calling Error", 
            "structured": "Structured Output Error",
            "thinking": "Thinking Test"
        }
    },
    "embedding": {
        "test_buckets": {
            "main": (["test_embeddings.py"], "--embedding-model-id")
        },
        "csv_columns": ["Model", "Error"],
        "column_map": {"main": "Error"}
    },
    "vlm": {
        "test_buckets": {
            "main": (["test_vlm_models.py"], "--vlm-model-id")
        },
        "csv_columns": ["Model", "Error"],
        "column_map": {"main": "Error"}
    },
    "completion": {
        "test_buckets": {
            "main": (["test_completions_models.py"], "--completions-model-id")
        },
        "csv_columns": ["Model", "Error"],
        "column_map": {"main": "Error"}
    }
}

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def get_models(model_type: str) -> List[str]:
    sys.path.insert(0, str(AI_ENDPOINTS_DIR))
    from langchain_nvidia_ai_endpoints._statics import (
        CHAT_MODEL_TABLE, EMBEDDING_MODEL_TABLE, VLM_MODEL_TABLE, COMPLETION_MODEL_TABLE
    )
    
    tables = {
        "chat": CHAT_MODEL_TABLE,
        "embedding": EMBEDDING_MODEL_TABLE, 
        "vlm": VLM_MODEL_TABLE,
        "completion": COMPLETION_MODEL_TABLE
    }
    
    if model_type not in tables:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return list(tables[model_type].keys())


def run_test_bucket(model: str, model_type: str, bucket: str) -> Tuple[bool, str]:
    test_files, flag = MODEL_TEST_CONFIG[model_type]["test_buckets"][bucket]
    cmd = [
        "poetry", "run", "pytest",
        *[f"tests/integration_tests/{test_file}" for test_file in test_files],
        flag, model,
        "--tb=short", "-rA", "--color=yes"
    ]
    
    result = subprocess.run(
        cmd, cwd=AI_ENDPOINTS_DIR,
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        return True, "PASSED"
    
    # Extract failure messages from pytest summary
    output = ANSI_RE.sub("", result.stdout)
    lines = output.splitlines()
    
    for i, line in enumerate(lines):
        if "short test summary info" in line:
            failures = []
            for j in range(i + 1, len(lines)):
                line = lines[j].strip()
                if not line or line.startswith("="):
                    break
                if line.startswith("FAILED"):
                    failures.append(line)
            if not failures:
                raise RuntimeError(f"No failure details found in pytest output for {model}/{bucket}")
            return False, "\n".join(failures)
    
    raise RuntimeError(f"No test summary found in pytest output for {model}/{bucket}")



def run_model_tests(model_type: str, results_csv: Path) -> None:
    """Unified function to run tests for any model type"""
    models = get_models(model_type)
    if not models:
        raise RuntimeError(f"No {model_type} models found")
    
    config = MODEL_TEST_CONFIG[model_type]
    test_buckets = config["test_buckets"]
    headers = config["csv_columns"]
    column_map = config["column_map"]
    
    bucket_count = len(test_buckets)
    bucket_text = f"across {bucket_count} test buckets" if bucket_count > 1 else ""
    
    print(f"Testing {len(models)} {model_type} models {bucket_text}")
    print(f"Results: {results_csv}\n")
    
    try:
        with results_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for i, model in enumerate(models, 1):
                print(f"[{i}/{len(models)}] {model}")
                row = {"Model": model}
                
                for bucket in test_buckets:
                    if bucket_count > 1:
                        print(f"  - {bucket}…", end=" ", flush=True)
                    passed, msg = run_test_bucket(model, model_type, bucket)
                    if bucket_count > 1:
                        print("ok" if passed else "FAIL")
                    else:
                        print(" ok" if passed else " FAIL")
                    
                    row[column_map[bucket]] = msg
                
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                if bucket_count > 1:
                    print("Results saved")
                
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial results in: {results_csv}")
        raise
    
    print(f"Completed. Results saved to: {results_csv}\n")


def main() -> None:
    result_files = {
        "chat": CHAT_RESULTS_CSV,
        "embedding": EMBEDDING_RESULTS_CSV,
        "vlm": VLM_RESULTS_CSV,
        "completion": COMPLETION_RESULTS_CSV
    }
    
    for model_type in MODEL_TEST_CONFIG.keys():
        run_model_tests(model_type, result_files[model_type])
    
    print(f"All results saved to: {RESULTS_DIR}")
    for model_type, csv_file in result_files.items():
        print(f"  - {model_type.capitalize()} models: {csv_file.name}")


if __name__ == "__main__":
    main() 