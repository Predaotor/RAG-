"""Run the RAG agent Streamlit app."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Run from project root so paths resolve correctly
    project_root = Path(__file__).parent
    app_path = project_root / "src" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"],
        cwd=str(project_root),
    )
