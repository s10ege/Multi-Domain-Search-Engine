# Test Suite for Semantic Search Application

This directory contains a comprehensive test suite for the semantic search application, covering all three main modules with 75+ test cases.

## Quick Start

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# From the project root directory
pytest test/

# Or from this directory
cd test
pytest
```

### Run with Coverage

```bash
pytest test/ --cov=.. --cov-report=html --cov-report=term
```

## Test Files

### Core Test Modules

- **`test_build_embeddings.py`** (11 tests)
  - CSV loading and parsing
  - Data cleaning and validation
  - Embedding creation workflow
  - Domain normalization
  - Error handling

- **`test_search_engine.py`** (40+ tests)
  - Resource loading and management
  - Search functionality
  - Domain filtering and SQL injection protection
  - RAG (Retrieval-Augmented Generation)
  - More-like-this search
  - Result normalization

- **`test_client.py`** (25+ tests)
  - CLI interface and menu navigation
  - User input validation
  - Search workflows
  - RAG workflows
  - Domain selection and filtering
  - Error handling and edge cases

### Configuration Files

- **`pytest.ini`** - Pytest configuration
- **`requirements-test.txt`** - Testing dependencies

### Documentation

- **`TESTING.md`** - Detailed testing guide
- **`TEST_PLAN.md`** - Complete test plan summary

## Running Specific Tests

```bash
# Run a specific test file
pytest test/test_search_engine.py -v

# Run a specific test class
pytest test/test_search_engine.py::TestDomainClause -v

# Run a specific test
pytest test/test_search_engine.py::TestDomainClause::test_domain_clause_with_string -v

# Run tests matching a pattern
pytest test/ -k "domain" -v
```

## Test Categories

Tests are organized by priority:

- **CRITICAL** (~45 tests) - Core functionality that must pass
- **IMPORTANT** (~20 tests) - Edge cases and error handling
- **NICE-TO-HAVE** (~10 tests) - Additional scenarios

## Coverage Report

Generate HTML coverage report:

```bash
pytest test/ --cov=.. --cov-report=html
```

Then open `htmlcov/index.html` in your browser.

## Test Architecture

### Mocking Strategy

All tests use comprehensive mocking to:
- Avoid loading actual ML models (fast execution)
- Eliminate network calls
- Prevent file system dependencies
- Enable isolated unit testing

### Fixtures

Common test fixtures:
- `temp_dir` - Temporary directory with auto-cleanup
- `sample_csv` - Sample CSV data for testing
- `mock_resources` - Mocked SearchResources objects

### Best Practices

- ✅ All tests are independent and isolated
- ✅ Descriptive test names and docstrings
- ✅ Arrange-Act-Assert pattern
- ✅ Automatic cleanup of temporary files
- ✅ No external dependencies during testing

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install -r test/requirements-test.txt
      - run: pytest test/ --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Test Statistics

| Module | Test Cases | Lines of Code | Coverage |
|--------|-----------|---------------|----------|
| build_embeddings | 11 | ~400 | Data processing, validation |
| search_engine | 40+ | ~700 | Search, RAG, filtering |
| client | 25+ | ~650 | CLI, workflows, UX |
| **TOTAL** | **75+** | **~1,750** | **Complete coverage** |

## Documentation

For detailed information, see:

- **[TESTING.md](TESTING.md)** - Complete testing guide
- **[TEST_PLAN.md](TEST_PLAN.md)** - Comprehensive test plan

## Requirements

- Python 3.8+
- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- pytest-mock >= 3.11.1

See `requirements-test.txt` for the complete list.

## Contributing

When adding new features to the main application:

1. Write tests first (TDD approach)
2. Include critical, important, and nice-to-have test cases
3. Maintain >80% code coverage
4. Ensure all tests pass before committing

## Support

For questions or issues with the test suite, please refer to:
- [TESTING.md](TESTING.md) for detailed documentation
- [TEST_PLAN.md](TEST_PLAN.md) for test coverage details
