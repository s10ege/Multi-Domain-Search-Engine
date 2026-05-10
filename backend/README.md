# Backend - Semantic Search Engine

Python backend for the semantic search application using txtai.

## Structure

```
backend/
├── src/                  - Shared backend code
│   ├── build_embeddings.py
│   ├── client.py
│   └── search_engine.py
├── streamlit_app/        - Streamlit application
├── test/                 - Test suite (293 tests)
├── final_data.csv        - Source dataset
└── requirements.txt      - Python dependencies
```

## Setup

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Build the embeddings index (required before first run)

The embeddings index is not included. Generate it from the source dataset:

```bash
python src/build_embeddings.py
```

This creates the `embeddings/` folder required by the search engine. Only needs to be run once.

## Usage

### Run the CLI Client

```bash
cd backend
python src/client.py
```

### Run the Streamlit App

```bash
python -m streamlit run backend/streamlit_app/main.py
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
