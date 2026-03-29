from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_PATH = PROJECT_ROOT / "Verification" / "case1_astar_fixed.py"
OUTPUT_PATH = Path(__file__).with_name("case1_out.txt")

print("Executing case 1...")
try:
    result = subprocess.run(
        [sys.executable, str(CASE_PATH)],
        capture_output=True,
        text=True,
        check=True,
        cwd=PROJECT_ROOT,
    )
    output = result.stdout
    print("Execution complete.")
except subprocess.CalledProcessError as e:
    output = f"ERROR: {e.returncode}\n{e.stdout}\n{e.stderr}"
    print("Execution failed.")
    
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    f.write(output)
