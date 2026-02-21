# Semantic Search Engine (txtai)

This project provides a Semantic Search Engine and RAG (Retrieval-Augmented Generation) system using [txtai](https://github.com/neuMl/txtai). It allows you to index a dataset, perform semantic queries, ask questions (QA), and summarize results through a command-line interface.

## Project Structure

```
project/
├── backend/              - Python backend application
│   ├── src/              - Application code
│   ├── test/             - Test suite (71 tests, 89% coverage)
│   ├── embeddings/       - Pre-built txtai index
│   ├── final_data.csv    - Source dataset
│   └── requirements.txt  - Python dependencies
├── .venv/                - Virtual environment (project root)
└── README.md             - This file
```

## Quick Start

### 1. Setup Virtual Environment

```bash
# Create virtual environment at project root
python -m venv .venv

# Activate it
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Run the Application

```bash
cd backend
python src/client.py
```
C:/Users/soner/.vscode/projects/123/.venv/Scripts/python.exe -m src.client

## Features

- **Semantic Search** - Find documents by meaning, not just keywords
- **RAG (Answer Mode)** - Ask questions and get AI-generated answers from your data
- **Domain Filtering** - Search within specific domains
- **More Like This** - Find similar documents
- **Hybrid Search** - Combines vector similarity with BM25 keyword scoring

## Documentation

- **Backend Documentation**: See [backend/README.md](backend/README.md)
- **Testing Guide**: See [backend/test/TESTING.md](backend/test/TESTING.md)
- **Test Plan**: See [backend/test/TEST_PLAN.md](backend/test/TEST_PLAN.md)

## Development

### Running Tests

```bash
pytest backend/test/
```

### Code Coverage

```bash
pytest backend/test/ --cov=backend/src --cov-report=html
```

Current coverage: **89%**

## Technology Stack

- **Python 3.8+**
- **txtai** - Semantic search and RAG framework
- **sentence-transformers** - Embedding models
- **pandas** - Data processing
- **pytest** - Testing framework

## License

See LICENSE file for details.
