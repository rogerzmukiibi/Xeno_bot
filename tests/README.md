# XENO Bot - Unit Tests

This directory contains comprehensive unit tests for the XENO Bot application.

## Test Coverage

The test suite covers the following modules:

1. **test_utils.py** - Tests for PipelineTimer and logging utilities
2. **test_intent_classifier.py** - Tests for intent classification (greetings, thanks, goodbye, queries)
3. **test_knowledge_base.py** - Tests for knowledge base loading and document preparation
4. **test_memory.py** - Tests for LangGraph memory operations (SQLite-based)
5. **test_response_generator.py** - Tests for LLM response generation
6. **test_vector_store.py** - Tests for ChromaDB vector store operations
7. **test_logger.py** - Tests for Google Sheets logging functionality

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_utils.py
```

### Run with Coverage Report

```bash
pytest --cov=src --cov-report=html
```

This will generate an HTML coverage report in the `htmlcov/` directory.

### Run with Verbose Output

```bash
pytest -v
```

### Run Specific Test

```bash
pytest tests/test_intent_classifier.py::TestIntentClassifier::test_classify_greeting
```

## Test Structure

Each test file follows a consistent structure:

- **setUp()**: Initialize test fixtures and mock data
- **tearDown()**: Clean up after tests (if needed)
- **test_***: Individual test methods

## Mocking

Tests use Python's `unittest.mock` to:
- Mock external API calls (Google Generative AI)
- Mock database connections (ChromaDB, SQLite)
- Mock Google Sheets operations
- Isolate units under test

## Best Practices

- Tests are isolated and don't depend on external services
- Each test focuses on a single behavior
- Mock objects are used to simulate dependencies
- Tests include both positive and negative scenarios
- Edge cases are covered (empty inputs, errors, etc.)

## Continuous Integration

To integrate with CI/CD pipelines, add to your workflow:

```yaml
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=src --cov-report=xml
```

## Test Configuration

Test configuration is defined in:
- `pytest.ini` - Pytest configuration
- `setup.cfg` - Additional pytest and coverage settings

## Writing New Tests

When adding new functionality:

1. Create a test file in `tests/` following the naming convention `test_<module>.py`
2. Create test class inheriting from `unittest.TestCase`
3. Add test methods starting with `test_`
4. Use mocks for external dependencies
5. Run tests to ensure they pass

Example:

```python
import unittest
from unittest.mock import patch
from src.my_module import my_function

class TestMyModule(unittest.TestCase):
    def test_my_function(self):
        result = my_function("test")
        self.assertEqual(result, expected_value)
```
