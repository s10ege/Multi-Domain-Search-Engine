# CLAUDE.md — Semantic Search Engine (txtai)

DO not make any changes until you have 95% confidence in what you need to build. Ask me follow-up questions until you reach that confidence.

## Project Overview

Python-based Semantic Search Engine and RAG system using txtai. Includes a CLI client and a Streamlit web app with authentication, search history, and domain filtering.

## Stack

- **Python 3.8+** with virtual environment at `.venv/` (project root)
- **txtai** — semantic search and RAG framework
- **sentence-transformers** — embedding models
- **Streamlit** — web UI with GitHub OAuth login
- **FastAPI + uvicorn** — API server
- **SQLite** — auth (`auth.db`) and history (`search_history.db`) stores
- **pytest** — test framework (71 tests, 89% coverage)

## Project Structure

```
123/
├── backend/
│   ├── src/
│   │   ├── search_engine.py   # Core txtai engine
│   │   ├── client.py          # CLI client
│   │   └── build_embeddings.py
│   ├── streamlit_app/
│   │   ├── main.py            # Entry point
│   │   ├── search.py          # Search UI
│   │   ├── similar.py         # "More like this" UI
│   │   ├── auth_store.py      # SQLite auth
│   │   ├── history_store.py   # Search history
│   │   ├── login.py / signup.py / admin.py
│   │   ├── app_core.py        # Shared state/session
│   │   └── paths.py           # Path helpers
│   ├── embeddings/            # Pre-built txtai index
│   ├── final_data.csv         # Source dataset
│   └── requirements.txt
└── .venv/                     # Virtual environment
```

## Commands

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run CLI client
cd backend && python -m src.client

# Run Streamlit app (from project root)
python -m streamlit run backend/streamlit_app/main.py

# Run tests
pytest backend/test/

# Run tests with coverage
pytest backend/test/ --cov=backend/src --cov-report=html

# Format code
ruff format .

# Lint
ruff check .

# Lint + auto-fix
ruff check . --fix

# Type check
pyright
```

## Planned Features

- **External data fetcher via APIs** — fetch and index data from external sources into the search engine

## Core Development Rules

### Package Management
- Use `pip` and `venv` — do not introduce other package managers
- Install: `pip install package`
- Add to requirements: `pip freeze > requirements.txt` (or edit manually)
- Never commit `.venv/`

### Code Quality
- Type hints required on all functions and public APIs
- Public APIs must have docstrings
- Functions must be focused and small
- Follow existing patterns exactly
- Line length: **88 chars maximum** (enforced by Ruff)

### Testing Requirements
- Framework: `pytest`
- Test edge cases and error paths, not just the happy path
- New features require tests
- Bug fixes require a regression test

### Code Style
- PEP 8 naming: `snake_case` for functions and variables
- Class names in `PascalCase`
- Constants in `UPPER_SNAKE_CASE`
- Use f-strings for string formatting
- Docstrings on all public functions and classes

## Development Philosophy

- **Simplicity** — write simple, straightforward code
- **Readability** — make code easy to understand at a glance
- **Performance** — consider performance without sacrificing readability
- **Maintainability** — write code that is easy to update later
- **Testability** — structure code so it can be tested in isolation
- **Less Code = Less Debt** — minimise the code footprint

## Coding Best Practices

- **Early Returns** — use to avoid deeply nested conditions
- **Descriptive Names** — use clear variable and function names; prefix event handlers with `handle_`
- **DRY** — don't repeat yourself; extract shared logic
- **Functional Style** — prefer functional, immutable approaches where they don't add verbosity
- **Minimal Changes** — only modify code directly related to the task at hand
- **TODO Comments** — mark issues in existing code with `# TODO:` prefix
- **Build Iteratively** — start with minimal functionality, verify it works, then add complexity
- **Run Tests Frequently** — test with realistic inputs and validate outputs as you go
- **Clean Logic** — keep core logic clean; push implementation details to the edges
- **File Organisation** — balance file organisation with simplicity; don't over-split small projects

## Python Tooling

### Ruff (formatting + linting)
- Format: `ruff format .`
- Check: `ruff check .`
- Auto-fix: `ruff check . --fix`
- Critical rules enforced:
  - Line length (88 chars)
  - Import sorting (`I001`)
  - Unused imports
- Line wrapping style:
  - Long strings: wrap with parentheses
  - Long function calls: multi-line with consistent indent
  - Long imports: split across multiple lines

### Pyright (type checking)
- Run: `pyright`
- Install: `pip install pyright`
- Requirements:
  - Explicit `None` checks for `Optional` types
  - Type narrowing for union types
  - Version warnings can be ignored if all checks pass

## Error Resolution

When CI or local checks fail, fix in this order:
1. **Formatting** — run `ruff format .` first
2. **Type errors** — run `pyright`, fix each error with full line context in mind
3. **Linting** — run `ruff check . --fix` last

Common fixes:
- Line too long → break with parentheses or split into multiple lines
- `Optional` type error → add explicit `if x is not None` check
- Import order → let `ruff check . --fix` handle it automatically

## Git Workflow

- Commit directly to `main` for this project
- Write a short, plain-English message with every push explaining **what changed and why**
  - Good: `"Fix threshold filter — papers below 0.70 were slipping through due to wrong comparator"`
  - Bad: `"fix bug"` or `"update search.py"`
- Never commit secrets, `.env` files, or SQLite database files (`auth.db`, `search_history.db`)

## Development Notes

- Embeddings index is pre-built — only re-run `build_embeddings.py` when the dataset changes
- `faiss-cpu` is excluded on Windows (see `requirements.txt`) — txtai falls back to its built-in index
- Auth and history are SQLite files inside `streamlit_app/` — do not commit these
- Session state lives in `app_core.py` — check there first for login/logout bugs
- Use the **context7 MCP** to look up current library docs (txtai, Streamlit, FastAPI) before assuming behaviour from memory
