## 🛠 Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### 1. Install uv
If you haven't installed `uv` yet, run the appropriate command for your system:

**macOS/Linux:**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
Windows:

PowerShell
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

2. Recreate the Environment
Navigate to the project root and synchronize the environment. This will automatically create a .venv and install all required dependencies:

Bash
uv sync
