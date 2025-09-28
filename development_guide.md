# Development Guide for GenAI Python Projects

This guide provides comprehensive development standards and best practices for our GenAI/Agentic AI team. Follow these guidelines to write secure, maintainable, and scalable Python code.

## 🚀 Quick Start

### Initial Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd full-stack-gen-ai-with-python

# 2. Run the automated setup
python scripts/dev.py setup

# 3. Verify setup works
python scripts/dev.py check
```

### Daily Workflow

```bash
# Before starting work
python scripts/dev.py lint    # Format and lint code
python scripts/dev.py test    # Run tests

# Before committing (automatic via pre-commit)
python scripts/dev.py check   # Run all quality checks

# Weekly maintenance
python scripts/dev.py update  # Update dependencies
```

## 📋 Development Standards

### Code Quality Tools

Our development environment uses multiple tools working together:

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Ruff** | Fast linting & formatting | `pyproject.toml` |
| **MyPy** | Type checking | `pyproject.toml` |
| **Bandit** | Security scanning | `pyproject.toml` |
| **Pytest** | Testing framework | `pyproject.toml` |
| **Pre-commit** | Git hook automation | `.pre-commit-config.yaml` |
| **Coverage** | Test coverage tracking | `pyproject.toml` |

### Code Style Guidelines

#### 1. **Python Version & Imports**
```python
# Always use future annotations at the top
from __future__ import annotations

# Standard library imports
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports (grouped by category)
import numpy as np
import pandas as pd
import openai
from langchain.llms import OpenAI

# Local imports
from your_project.core import SomeClass
from your_project.utils import helper_function
```

#### 2. **Type Annotations**
```python
# Function annotations (gradually adopt)
def process_data(
    data: pd.DataFrame,
    config: dict[str, Any],
    output_path: Path | None = None
) -> dict[str, float]:
    """Process data with proper type hints."""
    return {"accuracy": 0.95}

# Class annotations
class AIAgent:
    def __init__(self, model_name: str, temperature: float = 0.7) -> None:
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """Generate response from the AI model."""
        # Implementation here
        return "response"
```

#### 3. **Error Handling**
```python
# Good: Specific exception handling
try:
    response = openai.chat.completions.create(...)
except openai.RateLimitError:
    logger.warning("Rate limit hit, retrying...")
    time.sleep(60)
except openai.APIError as e:
    logger.error(f"OpenAI API error: {e}")
    raise

# Good: Custom exceptions for your domain
class ModelNotFoundError(Exception):
    """Raised when a requested AI model is not available."""
    pass

class APIKeyMissingError(Exception):
    """Raised when required API key is not configured."""
    pass
```

#### 4. **Logging Best Practices**
```python
import logging

logger = logging.getLogger(__name__)

# Good: Structured logging
def train_model(data_path: Path, model_config: dict) -> None:
    logger.info(
        "Starting model training",
        extra={
            "data_path": str(data_path),
            "model_type": model_config.get("type"),
            "num_samples": len(data)
        }
    )

    try:
        # Training logic
        pass
    except Exception as e:
        logger.error(
            "Model training failed",
            extra={"error": str(e), "data_path": str(data_path)},
            exc_info=True
        )
        raise
```

#### 5. **Configuration Management**
```python
# Good: Use dataclasses/pydantic for configuration
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in environment")

# Usage
config = ModelConfig(
    model_name="gpt-4",
    temperature=0.5
)
```

### Security Best Practices

#### 1. **API Key Management**
```python
# ❌ Never do this
openai_client = OpenAI(api_key="sk-...")

# ✅ Always use environment variables
import os
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Or use a secure configuration system
from your_project.config import get_api_key
openai_client = OpenAI(api_key=get_api_key("openai"))
```

#### 2. **Input Validation**
```python
def process_user_prompt(prompt: str, max_length: int = 1000) -> str:
    """Process user input with validation."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if len(prompt) > max_length:
        raise ValueError(f"Prompt too long: {len(prompt)} > {max_length}")

    # Sanitize input
    prompt = prompt.strip()

    # Remove potential injection attempts
    dangerous_patterns = ["<script>", "javascript:", "data:"]
    for pattern in dangerous_patterns:
        if pattern.lower() in prompt.lower():
            logger.warning(f"Potentially dangerous pattern detected: {pattern}")
            prompt = prompt.replace(pattern, "")

    return prompt
```

#### 3. **Data Privacy**
```python
import hashlib

def anonymize_user_data(user_id: str, data: dict) -> dict:
    """Anonymize user data for logging/storage."""
    # Hash user ID
    hashed_id = hashlib.sha256(user_id.encode()).hexdigest()[:8]

    # Remove PII
    safe_data = {k: v for k, v in data.items()
                 if k not in ["email", "phone", "ssn", "address"]}

    return {"user_hash": hashed_id, **safe_data}
```

### Testing Best Practices

#### 1. **Test Structure**
```python
# tests/test_ai_agent.py
import pytest
from unittest.mock import Mock, patch
from your_project.ai_agent import AIAgent

class TestAIAgent:
    """Test suite for AIAgent class."""

    @pytest.fixture
    def agent(self):
        """Create a test agent instance."""
        return AIAgent(model_name="gpt-3.5-turbo", temperature=0.5)

    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        return {
            "choices": [{"message": {"content": "Test response"}}]
        }

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.model_name == "gpt-3.5-turbo"
        assert agent.temperature == 0.5

    @patch('openai.chat.completions.create')
    def test_generate_success(self, mock_create, agent, mock_openai_response):
        """Test successful text generation."""
        mock_create.return_value = mock_openai_response

        result = agent.generate("Test prompt")

        assert result == "Test response"
        mock_create.assert_called_once()

    @patch('openai.chat.completions.create')
    def test_generate_api_error(self, mock_create, agent):
        """Test handling of API errors."""
        mock_create.side_effect = openai.APIError("API Error")

        with pytest.raises(openai.APIError):
            agent.generate("Test prompt")

    @pytest.mark.llm  # Mark tests that require actual API calls
    def test_integration_with_real_api(self):
        """Integration test with real API (run sparingly)."""
        # Only run if API key is available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available")

        agent = AIAgent("gpt-3.5-turbo")
        response = agent.generate("Say 'test successful'")
        assert "test successful" in response.lower()
```

#### 2. **Test Data Management**
```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        "training_data": [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "How are you?", "output": "I'm doing well!"}
        ]
    }

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir

# pytest markers in pyproject.toml help organize tests
# Run fast tests: pytest -m "not slow"
# Run only unit tests: pytest -m unit
# Run expensive tests manually: pytest -m expensive --expensive
```

### Documentation Standards

#### 1. **Docstring Format (Google Style)**
```python
def train_model(
    data: pd.DataFrame,
    model_config: dict[str, Any],
    output_dir: Path,
    validation_split: float = 0.2
) -> dict[str, float]:
    """Train an AI model with the provided data.

    This function handles the complete training pipeline including
    data preprocessing, model training, and evaluation.

    Args:
        data: Training data with features and labels
        model_config: Configuration dictionary containing model parameters
            Expected keys: 'model_type', 'learning_rate', 'epochs'
        output_dir: Directory to save trained model and artifacts
        validation_split: Fraction of data to use for validation (0.0-1.0)

    Returns:
        Dictionary containing training metrics:
            - 'train_accuracy': Training accuracy score
            - 'val_accuracy': Validation accuracy score  
            - 'train_loss': Final training loss
            - 'val_loss': Final validation loss

    Raises:
        ValueError: If validation_split is not between 0.0 and 1.0
        ModelNotFoundError: If specified model_type is not supported

    Example:
        >>> config = {'model_type': 'transformer', 'learning_rate': 0.001}
        >>> metrics = train_model(df, config, Path('./models'))
        >>> print(f"Accuracy: {metrics['val_accuracy']:.2f}")
    """
    if not 0.0 <= validation_split <= 1.0:
        raise ValueError("validation_split must be between 0.0 and 1.0")

    # Implementation here
    return {"train_accuracy": 0.95, "val_accuracy": 0.92}
```

#### 2. **README Documentation**
Each module should have clear documentation:

```markdown
# AI Agent Module

## Overview
This module provides the core AI agent functionality for our GenAI application.

## Quick Start
```python
from your_project.ai_agent import AIAgent

agent = AIAgent("gpt-4", temperature=0.7)
response = agent.generate("Explain machine learning")
```

## API Reference
[Link to detailed API documentation]

## Configuration
- Set `OPENAI_API_KEY` environment variable
- Configure model parameters in `config/models.yaml`
```

### GenAI-Specific Best Practices

#### 1. **Prompt Engineering**
```python
class PromptTemplate:
    """Reusable prompt templates for consistency."""

    CLASSIFICATION = """
    Classify the following text into one of these categories: {categories}

    Text: {text}

    Category:"""

    SUMMARIZATION = """
    Summarize the following text in {max_words} words or less:

    {text}

    Summary:"""

def classify_text(text: str, categories: list[str]) -> str:
    """Classify text using a structured prompt."""
    prompt = PromptTemplate.CLASSIFICATION.format(
        categories=", ".join(categories),
        text=text
    )
    return llm.generate(prompt)
```

#### 2. **Token Management**
```python
def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for cost planning."""
    # Rough estimation: 1 token ≈ 4 characters for English
    return len(text) // 4

def manage_context_window(
    messages: list[dict],
    max_tokens: int = 8000
) -> list[dict]:
    """Manage context window to stay within limits."""
    total_tokens = sum(estimate_tokens(msg["content"]) for msg in messages)

    if total_tokens <= max_tokens:
        return messages

    # Keep system message and recent messages
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    other_messages = [msg for msg in messages if msg["role"] != "system"]

    # Truncate older messages
    while total_tokens > max_tokens and len(other_messages) > 1:
        removed = other_messages.pop(0)
        total_tokens -= estimate_tokens(removed["content"])

    return system_messages + other_messages
```

#### 3. **Error Handling for AI Services**
```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying API calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except openai.RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit, retrying in {delay:.1f}s")
                    time.sleep(delay)
                except openai.APIError as e:
                    logger.error(f"API error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def call_llm(prompt: str) -> str:
    """Call LLM with automatic retry logic."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

## 🔧 Development Workflow

### Daily Development Cycle

1. **Start of Day**
   ```bash
   git pull origin main
   python scripts/dev.py update  # Weekly
   ```

2. **Development**
   ```bash
   # Work on features
   python scripts/dev.py lint     # Format code
   python scripts/dev.py test     # Run tests
   ```

3. **Before Committing**
   ```bash
   python scripts/dev.py check    # Full quality check
   git add .
   git commit -m "feat: add new feature"  # Pre-commit runs automatically
   ```

4. **Before Pull Request**
   ```bash
   python scripts/dev.py full     # Complete check including docs
   ```

### Git Workflow & Commit Standards

#### Commit Message Format
Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, missing semi colons, etc)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvements
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies
- `ci`: Changes to CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files

**Examples:**
```bash
git commit -m "feat(agents): add retry logic for API calls"
git commit -m "fix: handle empty responses from OpenAI API"
git commit -m "docs: update installation instructions"
git commit -m "test(agents): add integration tests for GPT-4"
```

#### Branch Naming Convention
```
<type>/<short-description>
```

Examples:
- `feat/add-langchain-integration`
- `fix/memory-leak-in-embeddings`
- `docs/api-reference-update`
- `refactor/simplify-prompt-templates`

### Code Review Guidelines

#### For Authors
1. **Self-Review First**
   - Run `python scripts/dev.py check` before requesting review
   - Ensure all tests pass and coverage is maintained
   - Update documentation for new features

2. **PR Description Template**
   ```markdown
   ## What
   Brief description of changes

   ## Why
   Context and motivation

   ## How
   Technical implementation details

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No hardcoded secrets or API keys
   ```

#### For Reviewers
1. **Focus Areas**
   - Security: No hardcoded secrets, proper input validation
   - Performance: Efficient algorithms, proper resource management
   - Maintainability: Clear code, good abstractions
   - Testing: Adequate test coverage, edge cases handled

2. **Review Checklist**
   - [ ] Code is readable and well-documented
   - [ ] No security vulnerabilities introduced
   - [ ] Tests cover new functionality
   - [ ] Error handling is appropriate
   - [ ] No unnecessary complexity

## 🧪 Testing Strategy

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions/classes in isolation
   - Fast execution (< 1ms per test)
   - No external dependencies

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - May use test databases or mock services
   - Moderate execution time (< 100ms per test)

3. **End-to-End Tests** (`tests/e2e/`)
   - Test complete user workflows
   - Use real or staging services
   - Slower execution (< 10s per test)

4. **LLM Tests** (`tests/llm/`)
   - Test actual LLM integrations
   - Require API keys and credits
   - Run manually or in special CI jobs

### Test Organization Example

```
tests/
├── unit/
│   ├── test_prompt_templates.py
│   ├── test_token_counter.py
│   └── test_data_utils.py
├── integration/
│   ├── test_agent_workflow.py
│   ├── test_database_operations.py
│   └── test_api_endpoints.py
├── e2e/
│   ├── test_complete_pipeline.py
│   └── test_user_scenarios.py
├── llm/
│   ├── test_openai_integration.py
│   ├── test_anthropic_integration.py
│   └── test_model_performance.py
└── fixtures/
    ├── sample_data.json
    ├── mock_responses.py
    └── test_configs.py
```

### Running Different Test Suites

```bash
# Fast tests for development
pytest -m "not slow and not llm" --maxfail=1

# All tests except LLM
pytest -m "not llm"

# Integration tests only
pytest -m integration

# LLM tests (requires API keys)
pytest -m llm --expensive

# Performance tests
pytest -m perf --benchmark-only
```

## 🚀 Performance & Optimization

### GenAI-Specific Performance Tips

#### 1. **Caching Strategies**
```python
import functools
from typing import Dict, Any

# Cache expensive embeddings
@functools.lru_cache(maxsize=1000)
def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    """Get cached embeddings to avoid repeated API calls."""
    response = openai.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Cache prompt templates
class CachedPromptTemplate:
    def __init__(self):
        self._cache: Dict[str, str] = {}

    def format_prompt(self, template: str, **kwargs) -> str:
        cache_key = f"{template}:{hash(frozenset(kwargs.items()))}"
        if cache_key not in self._cache:
            self._cache[cache_key] = template.format(**kwargs)
        return self._cache[cache_key]
```

#### 2. **Batch Processing**
```python
async def process_batch(
    texts: list[str],
    batch_size: int = 20,
    delay: float = 1.0
) -> list[str]:
    """Process texts in batches to avoid rate limits."""
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Process batch concurrently
        tasks = [process_single_text(text) for text in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing text {i+j}: {result}")
                results.append("")  # or handle appropriately
            else:
                results.append(result)

        # Rate limiting delay
        if i + batch_size < len(texts):
            await asyncio.sleep(delay)

    return results
```

#### 3. **Memory Management**
```python
import gc
import psutil
from contextlib import contextmanager

@contextmanager
def monitor_memory():
    """Context manager to monitor memory usage."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    try:
        yield
    finally:
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB")

# Use for memory-intensive operations
with monitor_memory():
    large_embeddings = compute_embeddings(large_dataset)
```

## 🔒 Security Guidelines

### API Key Management

#### 1. **Environment Variables**
```python
# .env.example (commit this)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
DATABASE_URL=postgresql://user:pass@localhost/db

# .env (never commit this)
OPENAI_API_KEY=sk-real-key-here
ANTHROPIC_API_KEY=ant-real-key-here
```

#### 2. **Configuration Management**
```python
from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with validation."""

    openai_api_key: str
    anthropic_api_key: str | None = None
    database_url: str
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Usage
settings = get_settings()
openai_client = OpenAI(api_key=settings.openai_api_key)
```

### Input Validation & Sanitization

```python
from pydantic import BaseModel, validator
import re

class UserPrompt(BaseModel):
    """Validated user prompt model."""

    text: str
    max_tokens: int = 1000
    temperature: float = 0.7

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')

        if len(v) > 10000:
            raise ValueError('Text too long')

        # Remove potential script injections
        dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html'
        ]

        for pattern in dangerous_patterns:
            v = re.sub(pattern, '', v, flags=re.IGNORECASE | re.DOTALL)

        return v.strip()

    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v

# Usage
try:
    prompt = UserPrompt(text=user_input, temperature=0.8)
    response = generate_response(prompt)
except ValueError as e:
    logger.warning(f"Invalid input: {e}")
    return {"error": "Invalid input provided"}
```

## 📊 Monitoring & Observability

### Logging Configuration

```python
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: Path | None = None):
    """Setup structured logging for the application."""

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler (optional)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
```

### Metrics Collection

```python
import time
from functools import wraps
from collections import defaultdict, Counter
from typing import Dict, Any

class MetricsCollector:
    """Simple metrics collector for monitoring."""

    def __init__(self):
        self.counters: Counter = Counter()
        self.timers: Dict[str, list] = defaultdict(list)
        self.gauges: Dict[str, float] = {}

    def increment(self, metric: str, value: int = 1, tags: Dict[str, str] | None = None):
        """Increment a counter metric."""
        key = self._make_key(metric, tags)
        self.counters[key] += value

    def timing(self, metric: str, value: float, tags: Dict[str, str] | None = None):
        """Record a timing metric."""
        key = self._make_key(metric, tags)
        self.timers[key].append(value)

    def gauge(self, metric: str, value: float, tags: Dict[str, str] | None = None):
        """Set a gauge metric."""
        key = self._make_key(metric, tags)
        self.gauges[key] = value

    def _make_key(self, metric: str, tags: Dict[str, str] | None) -> str:
        if not tags:
            return metric
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric}|{tag_str}"

    def report(self) -> Dict[str, Any]:
        """Get current metrics report."""
        return {
            "counters": dict(self.counters),
            "timers": {k: {
                "count": len(v),
                "avg": sum(v) / len(v) if v else 0,
                "min": min(v) if v else 0,
                "max": max(v) if v else 0
            } for k, v in self.timers.items()},
            "gauges": dict(self.gauges)
        }

# Global metrics instance
metrics = MetricsCollector()

def timed(metric_name: str):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                metrics.increment(f"{metric_name}.success")
                return result
            except Exception as e:
                metrics.increment(f"{metric_name}.error", tags={"error_type": type(e).__name__})
                raise
            finally:
                execution_time = time.time() - start_time
                metrics.timing(metric_name, execution_time)
        return wrapper
    return decorator

# Usage
@timed("llm.generate")
def generate_text(prompt: str) -> str:
    """Generate text with automatic metrics collection."""
    return llm.generate(prompt)
```

## 📈 Continuous Improvement

### Weekly Team Practices

1. **Code Quality Review**
   - Run `python scripts/dev.py full` on main branch
   - Review metrics and coverage reports
   - Identify areas for improvement

2. **Dependency Updates**
   - Run `python scripts/dev.py update`
   - Test thoroughly after updates
   - Update lockfiles

3. **Security Audit**
   - Run `python scripts/dev.py security`
   - Review any new security alerts
   - Update dependencies with security fixes

4. **Performance Review**
   - Monitor API usage and costs
   - Review slow tests and optimize
   - Check memory usage patterns

### Team Learning Resources

1. **Python Best Practices**
   - [Real Python](https://realpython.com/)
   - [Effective Python by Brett Slatkin](https://effectivepython.com/)
   - [Clean Code in Python](https://www.packtpub.com/product/clean-code-in-python/9781788835831)

2. **GenAI Development**
   - [OpenAI Cookbook](https://cookbook.openai.com/)
   - [LangChain Documentation](https://python.langchain.com/)
   - [Prompt Engineering Guide](https://www.promptingguide.ai/)

3. **Testing & DevOps**
   - [Test-Driven Development with Python](https://www.obeythetestinggoat.com/)
   - [Python Testing with pytest](https://pragprog.com/titles/bopytest/python-testing-with-pytest/)

## 🎯 Team Guidelines Summary

### DO
✅ Use type hints gradually (start with function signatures)  
✅ Write docstrings for public functions and classes  
✅ Use environment variables for all secrets  
✅ Add tests for new functionality  
✅ Run `python scripts/dev.py check` before commits  
✅ Use structured logging with context  
✅ Handle API errors gracefully with retries  
✅ Cache expensive operations (embeddings, API calls)  
✅ Validate all user inputs  
✅ Use conventional commit messages  

### DON'T
❌ Commit API keys or secrets  
❌ Skip error handling for external APIs  
❌ Write functions longer than 50 lines  
❌ Ignore linting warnings  
❌ Skip writing tests for bug fixes  
❌ Use bare `except:` clauses  
❌ Hardcode file paths or URLs  
❌ Ignore security warnings from tools  
❌ Push code that fails CI checks  
❌ Use `print()` instead of proper logging  

### Getting Help

1. **Code Issues**: Create a GitHub issue with minimal reproduction example
2. **Tool Problems**: Check the development guide first, then ask team
3. **Architecture Decisions**: Discuss in team meetings or RFC documents
4. **Security Concerns**: Report immediately to team lead

Remember: **Code quality is everyone's responsibility**. These tools and practices are here to help us write better, more secure, and more maintainable code for our GenAI applications.
