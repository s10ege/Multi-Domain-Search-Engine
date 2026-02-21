# TESTING.md
# Testing Guide for Semantic Search Application

This guide provides comprehensive information about the test suite for the semantic search application.

## Overview

The test suite includes comprehensive unit and integration tests for all three main Python modules:
- **test_build_embeddings.py** - Tests for embedding generation and indexing
- **test_search_engine.py** - Tests for search functionality and RAG
- **test_client.py** - Tests for CLI interface and user interaction

All test files are located in the `test/` directory to keep the application code clean and separate from testing components.

## Quick Start

### Install Testing Dependencies

```bash
# Install test requirements from the project root
pip install -r test/requirements-test.txt
```

### Run All Tests

```bash
# From project root directory
pytest test/

# From within the test directory
cd test
pytest

# Run with coverage report (from project root)
pytest test/ --cov=. --cov-report=html --cov-report=term

# Run specific test file (from project root)
pytest test/test_search_engine.py

# Run specific test class
pytest test/test_search_engine.py::TestDomainClause

# Run specific test
pytest test/test_search_engine.py::TestDomainClause::test_domain_clause_with_string
```

## Test Organization

### Test Categories

Tests are organized by priority:

1. **CRITICAL** - Core functionality tests that must pass
   - Basic operations (load, search, save)
   - Primary user workflows
   - Data integrity and correctness

2. **IMPORTANT** - Edge cases and error handling
   - Invalid input handling
   - Boundary conditions
   - Error recovery

3. **NICE-TO-HAVE** - Additional coverage
   - Performance considerations
   - Optional features
   - Extended scenarios

### Test Structure

Each test file follows this structure:
```python
class TestModuleName:
    """Test suite for specific module/function"""
    
    @pytest.fixture
    def fixture_name(self):
        """Reusable test data or setup"""
        pass
    
    def test_specific_behavior(self):
        """
        Test description
        Priority: CRITICAL/IMPORTANT/NICE-TO-HAVE
        """
        # Arrange
        # Act
        # Assert
```

## Test Coverage

### test_build_embeddings.py

**Coverage:**
- ✅ CSV loading and parsing
- ✅ Unnamed column removal
- ✅ Missing value handling
- ✅ Domain normalization
- ✅ Graph feature detection
- ✅ Record structure validation
- ✅ Encoding handling
- ✅ Error scenarios

**Key Tests:**
- `test_build_creates_embeddings_successfully` - Verifies embedding creation workflow
- `test_build_removes_unnamed_columns` - Ensures data cleaning
- `test_build_drops_rows_with_missing_values` - Data validation
- `test_build_normalizes_domain_field` - Domain standardization

### test_search_engine.py

**Coverage:**
- ✅ Resource loading and initialization
- ✅ Domain filtering logic
- ✅ Search query execution
- ✅ RAG answer generation
- ✅ Result normalization
- ✅ More-like-this functionality
- ✅ Summarization
- ✅ Resource cleanup

**Key Tests:**
- `test_load_resources_success` - Complete resource initialization
- `test_domain_clause_*` - SQL injection protection and filtering
- `test_search_returns_top_k_results` - Search correctness
- `test_rag_answer_calls_rag_with_context` - RAG pipeline
- `test_more_like_this_filters_seed_document` - Similar document search

### test_client.py

**Coverage:**
- ✅ Domain selection UI
- ✅ Menu navigation
- ✅ Search workflow
- ✅ RAG workflow
- ✅ More-like-this workflow
- ✅ Configuration changes (domain, top-k)
- ✅ Input validation
- ✅ Error handling
- ✅ Resource cleanup

**Key Tests:**
- `test_main_initializes_resources` - Startup workflow
- `test_main_menu_option_1_search` - Search execution
- `test_main_menu_option_2_rag` - RAG execution
- `test_main_closes_resources_on_exception` - Cleanup guarantee
- `test_full_search_workflow` - End-to-end integration

## Running Tests

### Basic Commands

```bash
# Run all tests (from project root)
pytest test/

# Run all tests (from test directory)
cd test && pytest

# Run with verbose output
pytest test/ -v

# Run specific test file
pytest test/test_search_engine.py

# Run tests matching pattern
pytest test/ -k "domain"

# Run tests in parallel (requires pytest-xdist)
pytest test/ -n auto
```

### Coverage Reports

```bash
# Generate HTML coverage report (from project root)
pytest test/ --cov=. --cov-report=html

# View coverage in terminal
pytest test/ --cov=. --cov-report=term

# Generate coverage with missing lines
pytest test/ --cov=. --cov-report=term-missing
```

### Advanced Options
test/ -x

# Show local variables in tracebacks
pytest test/ -l

# Run only failed tests from last run
pytest test/ --lf

# Run failed tests first, then others
pytest test/ --ff

# Disable warnings
pytest test/ --ff

# Disable warnings
pytest --disable-warnings
```

## Mocking Strategy

The tests use extensive mocking to isolate units and avoid dependencies:

### External Dependencies
- **txtai.Embeddings** - Mocked to avoid loading actual models
- **txtai.RAG** - Mocked to avoid LLM calls
- **pandas.read_csv** - Real data used in fixtures, mocked when needed
- **File I/O** - Uses temporary directories via fixtures

### User Input
- **builtins.input** - Mocked with predetermined responses
- **builtins.print** - Mocked to capture and verify output

### Example Mock Usage:
```python
@patch('search_engine.Embeddings')
def test_function(mock_embeddings_class):
    mock_instance = MagicMock()
    mock_embeddings_class.return_value = mock_instance
    
    # Test code that uses Embeddings
    # Verify mock interactions
    mock_instance.load.assert_called_once()
```

## Fixtures

Common fixtures used across tests:

- **temp_dir** - Temporary directory for test files
- **sample_csv** - Sample CSV data for testing
- **mock_resources** - Mocked SearchResources object

## Best Practices

1. **Isolation** - Each test is independent and doesn't rely on others
2. **Cleanup** - Temporary files are cleaned up automatically
3. **Descriptive Names** - Test names clearly describe what is being tested
4. **Docstrings** - Each test includes description and priority
5. **Arrange-Act-Assert** - Tests follow AAA pattern
6. **Mock External Calls** - Avoid network/file system when possible

## Continuous Integration

To integrate with CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pip install -r requirements-test.txt
    pytest --cov=. --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
pip install -r requirements-test.txt
```

**Module Not Found:**
```bash
# Run pytest from project root directory
cd /path/to/project
pytest
```

**Fixture Conflicts:**
```bash
# Clear pytest cache
pytest --cache-clear
```

## Contributing Tests

When adding new features:

1. Write tests first (TDD approach)
2. Include critical, important, and nice-to-have tests
3. Add docstrings with priority levels
4. Update this guide with new test coverage
5. Ensure tests pass before committing

## Test Metrics

Target metrics:
- **Coverage**: >80% line coverage
- **Pass Rate**: 100% on main branch
- **Execution Time**: <30 seconds for full suite

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-mock documentation](https://pytest-mock.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
