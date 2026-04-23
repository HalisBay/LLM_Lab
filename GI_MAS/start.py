import subprocess
import sys
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    backend_dir = base_dir / "backend"
    venv_python = base_dir / ".venv" / "Scripts" / "python.exe"
    python_exec = str(venv_python) if venv_python.exists() else sys.executable

    command = [
        python_exec,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]

    subprocess.run(command, cwd=str(backend_dir), check=True)


if __name__ == "__main__":
    main()
