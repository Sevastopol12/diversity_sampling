## 🛠 Environment Setup

### Option 1: Using uv (Recommended)
[uv](https://github.com/astral-sh/uv) is a fast, reliable Python package manager that handles virtual environments and dependencies automatically[cite: 1].

1.  **Install uv**:
    *   **macOS/Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`[cite: 1, 2]
    *   **Windows**: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`[cite: 1, 2]
2.  **Sync Environment**:
    ```bash
    uv sync
    ```
    *This creates a `.venv` and installs all dependencies from the lockfile[cite: 1, 2].*
3.  **Run**: `uv run python main.py`[cite: 1, 2]

---

### Option 2: Using pip (Legacy)
If you prefer the standard Python workflow, use the following steps:

1.  **Create a virtual environment**:
    ```bash
    # macOS/Linux
    python3 -m venv .venv
    # Windows
    python -m venv .venv
    ```
2.  **Activate the environment**:
    ```bash
    # macOS/Linux
    source .venv/bin/activate
    # Windows
    .venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

### Summary of Quick Commands

| Task | uv Command | pip Command |
| :--- | :--- | :--- |
| **Install Project** | `uv sync`[cite: 1, 2] | `pip install -r requirements.txt` |
| **Add Package** | `uv add <pkg>`[cite: 1, 2] | `pip install <pkg>` |
| **Run Script** | `uv run <script>`[cite: 1, 2] | `python <script>` |

> **Note:** For a visual walkthrough, you can refer to the [UV_Astral_Setup_Guide.pdf](UV_Astral_Setup_Guide.pdf) or the [setup_guide.html](setup_guide.html) files generated in this repositoryIf you need to support users who aren't using **uv** yet, or if you want to provide a fallback for traditional **pip** workflows, you can add this section to your `README.md`. It keeps things clean by offering the modern `uv` approach first, followed by the classic `pip` method.

---

## 🛠 Environment Setup

### Option 1: Using uv (Recommended)
[uv](https://github.com/astral-sh/uv) is a fast, reliable Python package manager that handles virtual environments and dependencies automatically[cite: 1].

1.  **Install uv**:
    *   **macOS/Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`[cite: 1, 2]
    *   **Windows**: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`[cite: 1, 2]
2.  **Sync Environment**:
    ```bash
    uv sync
    ```
    *This creates a `.venv` and installs all dependencies from the lockfile[cite: 1, 2].*
3.  **Run**: `uv run python main.py`[cite: 1, 2]

---

### Option 2: Using pip (Legacy)
If you prefer the standard Python workflow, use the following steps:

1.  **Create a virtual environment**:
    ```bash
    # macOS/Linux
    python3 -m venv .venv
    # Windows
    python -m venv .venv
    ```
2.  **Activate the environment**:
    ```bash
    # macOS/Linux
    source .venv/bin/activate
    # Windows
    .venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

### Summary of Quick Commands

| Task | uv Command | pip Command |
| :--- | :--- | :--- |
| **Install Project** | `uv sync`[cite: 1, 2] | `pip install -r requirements.txt` |
| **Add Package** | `uv add <pkg>`[cite: 1, 2] | `pip install <pkg>` |
| **Run Script** | `uv run <script>`[cite: 1, 2] | `python <script>` |

> **Note:** For a visual walkthrough, you can refer to the [UV_Astral_Setup_Guide.pdf](UV_Astral_Setup_Guide.pdf) or the [setup_guide.html](setup_guide.html) files generated in this repository[cite: 1, 2].
