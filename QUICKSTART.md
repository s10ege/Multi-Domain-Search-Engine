# Quick Reference - Development Commands

All commands assume you're in the project root directory.

## Virtual Environment

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

## Running Application

```bash
# Run the CLI client
cd backend
.venv/Scripts/python.exe -m src.client

# Run the Streamlit app
cd ..
.venv/Scripts/python.exe -m streamlit run backend/streamlit_app/main.py
```

## Building Embeddings

```bash
# Build or rebuild the search index
cd backend
python src/build_embeddings.py
```

## Testing

```bash
# Run all tests
pytest backend/test/

# Run specific module tests
pytest backend/test/test_search_engine.py -v

# Run with coverage
pytest backend/test/ --cov=backend/src --cov-report=html

# View coverage report
start htmlcov/index.html  # Windows
# open htmlcov/index.html  # Mac
```

## Development Workflow

```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Make code changes in backend/src/
#    Streamlit UI changes live in backend/streamlit_app/

# 3. Run tests
pytest backend/test/ -v

# 4. Check coverage
pytest backend/test/ --cov=backend/src --cov-report=term

# 5. Run application
cd backend
.venv/Scripts/python.exe -m src.client
```

## Project Structure

```
project/
├── .venv/                      # Virtual environment (DO NOT COMMIT)
├── .claude/                    # Claude templates
├── backend/                    # Application backend
│   ├── src/                    # Python source code
│   │   ├── __init__.py
│   │   ├── build_embeddings.py
│   │   ├── client.py
│   │   ├── search_engine.py
│   │   └── api.py
│   ├── streamlit_app/          # Streamlit application
│   ├── test/                   # Test suite
│   │   ├── test_build_embeddings.py
│   │   ├── test_client.py
│   │   ├── test_search_engine.py
│   │   ├── pytest.ini
│   │   ├── requirements-test.txt
│   │   ├── README.md
│   │   ├── TESTING.md
│   │   └── TEST_PLAN.md
│   ├── embeddings/             # Pre-built search index
│   ├── final_data.csv          # Source data
│   ├── requirements.txt        # Python dependencies
│   ├── .gitignore
│   └── README.md
├── .gitignore                  # Root gitignore
└── README.md                   # Project overview
```

## Common Issues

### "Module not found" error
- Make sure you're in the `backend/` directory when running Python scripts
- Ensure virtual environment is activated

### Tests failing
- Run `pytest backend/test/ -v` from project root
- Check that all dependencies are installed: `pip install -r backend/requirements.txt`

### Coverage report not generating
- Install pytest-cov: `pip install pytest-cov`
- Run from project root: `pytest backend/test/ --cov=backend/src`

## File Locations

- **Application code**: `backend/src/`
- **Streamlit app**: `backend/streamlit_app/`
- **Tests**: `backend/test/`
- **Data**: `backend/final_data.csv`
- **Index**: `backend/embeddings/`
- **Dependencies**: `backend/requirements.txt`
- **Test dependencies**: `backend/test/requirements-test.txt`
