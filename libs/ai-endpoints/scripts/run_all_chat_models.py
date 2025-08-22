import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

SCRIPT_PATH = Path(__file__).resolve()
AI_ENDPOINTS_DIR = SCRIPT_PATH.parents[1]
RESULTS_CSV = AI_ENDPOINTS_DIR / "test_results" / "chat_models_results.csv"
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

TEST_CONFIG = {
    "chat": ("test_chat_models.py", "--chat-model-id"),
    "tools": ("test_bind_tools.py", "--tool-model-id"),
    "structured": ("test_structured_output.py", "--structured-model-id"),
    "thinking": ("test_thinking_mode.py", "--thinking-model-id"),
}

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def get_chat_models() -> List[str]:
    sys.path.insert(0, str(AI_ENDPOINTS_DIR))
    from langchain_nvidia_ai_endpoints._statics import CHAT_MODEL_TABLE
    return list(CHAT_MODEL_TABLE.keys())


def run_test_bucket(model: str, bucket: str) -> Tuple[bool, str]:
    test_file, flag = TEST_CONFIG[bucket]
    cmd = [
        "poetry", "run", "pytest",
        f"tests/integration_tests/{test_file}",
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


def main() -> None:
    models = get_chat_models()
    if not models:
        raise RuntimeError("Could not find any chat models")
    
    print(f"Testing {len(models)} chat models across 4 test buckets")
    print(f"Results: {RESULTS_CSV}\n")
    
    headers = ["Model", "Chat Error", "Tool Calling Error", "Structured Output Error", "Thinking Test"]
    
    try:
        with RESULTS_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for i, model in enumerate(models, 1):
                print(f"[{i}/{len(models)}] {model}")
                row = {"Model": model}
                
                for bucket, (_, _) in TEST_CONFIG.items():
                    print(f"  - {bucket}…", end=" ", flush=True)
                    passed, msg = run_test_bucket(model, bucket)
                    print("ok" if passed else "FAIL")
                    
                    # Map bucket to CSV column
                    column_map = {
                        "chat": "Chat Error",
                        "tools": "Tool Calling Error", 
                        "structured": "Structured Output Error",
                        "thinking": "Thinking Test"
                    }
                    row[column_map[bucket]] = msg
                
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                print("Results saved")
                
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial results in: {RESULTS_CSV}")
        raise
    
    print(f"\nCompleted. Results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main() 