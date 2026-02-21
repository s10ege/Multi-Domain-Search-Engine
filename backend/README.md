# Backend - Semantic Search Engine

Python backend for the semantic search application using txtai.

## Structure

```
backend/
├── src/                  - Application code
│   ├── build_embeddings.py
│   ├── client.py
│   └── search_engine.py
├── test/                 - Test suite (71 tests, 89% coverage)
├── embeddings/           - Pre-built txtai index
├── final_data.csv        - Source dataset
└── requirements.txt      - Python dependencies
```

## Installation

```bash
cd backend

# Create virtual environment (from project root)
cd ..
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r backend/requirements.txt
```

## Usage

### Run the API server (FastAPI)

```bash
cd backend
python -m uvicorn src.api:app --reload --port 8000
```

### Run the CLI Client

```bash
cd backend
python src/client.py
```

### Build Embeddings (if needed)

```bash
cd backend
python src/build_embeddings.py
```

## Testing

```bash
# Run all tests
pytest backend/test/

# Run with coverage
pytest backend/test/ --cov=backend/src --cov-report=html

# View coverage report
start htmlcov/index.html
```

## Features

- **Semantic Search** - Vector similarity search using sentence transformers
- **RAG (Retrieval-Augmented Generation)** - LLM-powered question answering
- **Domain Filtering** - Search within specific domains
- **More Like This** - Find similar documents
- **BM25 Scoring** - Hybrid search combining vectors and keywords

## API

See [test/](test/) directory for comprehensive usage examples and API documentation through tests.

## Requirements

- Python 3.8+
- txtai >= 7.0.0
- pandas >= 2.0.0
- sentence-transformers >= 2.6.0
- transformers >= 4.40.0

See [requirements.txt](requirements.txt) for complete list.

## Development

Run tests before committing:

```bash
pytest backend/test/ -v
```

Maintain >80% code coverage (currently 89%).
