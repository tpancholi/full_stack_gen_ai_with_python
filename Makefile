# Makefile for GenAI Python Development
# Provides simple commands for common development tasks

.PHONY: help setup install test lint check security docs clean full update
.DEFAULT_GOAL := help

# Python command detection
PYTHON_CMD := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "python -m")
PYTHON_EXEC := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python")

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
BLUE := \033[0;34m
BOLD := \033[1m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BOLD)GenAI Python Development Commands$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(BOLD)Setup Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(setup|install)" | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Development Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(test|lint|check|format)" | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Maintenance Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(clean|update|security|docs)" | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Composite Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(full|quick)" | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(RED)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Examples:$(NC)"
	@echo "  make setup          # Initial project setup"
	@echo "  make quick          # Quick development check"
	@echo "  make test           # Run tests with coverage"
	@echo "  make full           # Complete quality check"

setup: ## Initial project setup with dependencies and hooks
	@echo "$(BLUE)🚀 Setting up development environment...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py setup

install: ## Install/sync all dependencies
	@echo "$(GREEN)📦 Installing dependencies...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py install

update: ## Update dependencies and tools
	@echo "$(YELLOW)🔄 Updating dependencies and tools...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py update

test: ## Run tests with coverage report
	@echo "$(GREEN)🧪 Running tests with coverage...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py test

test-fast: ## Run fast tests only (no slow/LLM tests)
	@echo "$(GREEN)⚡ Running fast tests only...$(NC)"
	@$(PYTHON_CMD) pytest -m "not slow and not llm" --maxfail=3

test-unit: ## Run unit tests only
	@echo "$(GREEN)🔬 Running unit tests...$(NC)"
	@$(PYTHON_CMD) pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)🔗 Running integration tests...$(NC)"
	@$(PYTHON_CMD) pytest tests/integration/ -v

test-llm: ## Run LLM tests (requires API keys)
	@echo "$(GREEN)🤖 Running LLM tests...$(NC)"
	@$(PYTHON_CMD) pytest -m llm --expensive

lint: ## Run linting and formatting
	@echo "$(GREEN)✨ Running linting and formatting...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py lint

format: ## Run code formatting only
	@echo "$(GREEN)🎨 Formatting code...$(NC)"
	@$(PYTHON_CMD) ruff format .
	@$(PYTHON_CMD) isort .

check: ## Run all code quality checks
	@echo "$(GREEN)🔍 Running all quality checks...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py check

security: ## Run security scans
	@echo "$(YELLOW)🔒 Running security scans...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py security

docs: ## Check documentation coverage
	@echo "$(YELLOW)📚 Checking documentation coverage...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py docs

dead-code: ## Check for dead code
	@echo "$(YELLOW)🧹 Scanning for dead code...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py dead-code

clean: ## Clean temporary files and caches
	@echo "$(YELLOW)🧽 Cleaning temporary files...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py clean

full: ## Run complete pre-commit simulation
	@echo "$(RED)🔥 Running complete quality check...$(NC)"
	@$(PYTHON_EXEC) scripts/dev.py full

quick: ## Quick development check (lint + fast tests)
	@echo "$(RED)⚡ Running quick development check...$(NC)"
	@$(MAKE) lint
	@$(MAKE) test-fast

# Git and pre-commit commands
pre-commit: ## Run pre-commit on all files
	@echo "$(BLUE)🔗 Running pre-commit on all files...$(NC)"
	@$(PYTHON_CMD) pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	@echo "$(BLUE)🔄 Updating pre-commit hooks...$(NC)"
	@$(PYTHON_CMD) pre-commit autoupdate

# CI simulation commands
ci-lint: ## Simulate CI linting checks
	@echo "$(BLUE)🤖 Simulating CI linting checks...$(NC)"
	@$(PYTHON_CMD) ruff check . --output-format=github
	@$(PYTHON_CMD) ruff format . --check

ci-test: ## Simulate CI test run
	@echo "$(BLUE)🤖 Simulating CI test run...$(NC)"
	@$(PYTHON_CMD) pytest --cov --cov-report=xml --cov-report=term-missing --junitxml=pytest.xml

ci-security: ## Simulate CI security checks
	@echo "$(BLUE)🤖 Simulating CI security checks...$(NC)"
	@$(PYTHON_CMD) bandit -r . -f json -o bandit-report.json -c pyproject.toml
	@$(PYTHON_CMD) safety check --json --output safety-report.json

# Development server commands (if applicable)
dev-server: ## Start development server (if applicable)
	@echo "$(GREEN)🚀 Starting development server...$(NC)"
	@$(PYTHON_CMD) uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Docker commands (if using Docker)
docker-build: ## Build Docker image
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	@docker build -t genai-python:latest .

docker-run: ## Run Docker container
	@echo "$(BLUE)🐳 Running Docker container...$(NC)"
	@docker run --rm -it -p 8000:8000 genai-python:latest

# Database commands (if applicable)
db-migrate: ## Run database migrations
	@echo "$(GREEN)🗄️ Running database migrations...$(NC)"
	@$(PYTHON_CMD) alembic upgrade head

db-reset: ## Reset database
	@echo "$(YELLOW)🗄️ Resetting database...$(NC)"
	@$(PYTHON_CMD) alembic downgrade base
	@$(PYTHON_CMD) alembic upgrade head

# Jupyter notebook commands
nb-clean: ## Clean Jupyter notebook outputs
	@echo "$(GREEN)📓 Cleaning notebook outputs...$(NC)"
	@$(PYTHON_CMD) nbstripout notebooks/*.ipynb

nb-convert: ## Convert notebooks to Python scripts
	@echo "$(GREEN)📓 Converting notebooks to Python...$(NC)"
	@$(PYTHON_CMD) nbconvert --to script notebooks/*.ipynb --output-dir scripts/

# Dependency management
deps-list: ## List all dependencies
	@echo "$(BLUE)📋 Listing dependencies...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv pip list; \
	else \
		pip list; \
	fi

deps-outdated: ## Check for outdated dependencies
	@echo "$(YELLOW)📋 Checking for outdated dependencies...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		echo "Use 'uv sync --upgrade' to update dependencies"; \
	else \
		pip list --outdated; \
	fi

deps-tree: ## Show dependency tree
	@echo "$(BLUE)🌳 Showing dependency tree...$(NC)"
	@$(PYTHON_CMD) pipdeptree

# Performance and profiling
profile: ## Profile the application (example)
	@echo "$(YELLOW)⚡ Running performance profiling...$(NC)"
	@$(PYTHON_CMD) cProfile -o profile.stats main.py
	@$(PYTHON_CMD) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

benchmark: ## Run benchmarks
	@echo "$(YELLOW)⚡ Running benchmarks...$(NC)"
	@$(PYTHON_CMD) pytest -m perf --benchmark-only

# Environment and system info
env-info: ## Show environment information
	@echo "$(BLUE)ℹ️ Environment Information$(NC)"
	@echo "=============================="
	@echo "Python version: $$(python --version)"
	@echo "Python path: $$(which python)"
	@if command -v uv >/dev/null 2>&1; then \
		echo "UV version: $$(uv --version)"; \
		echo "UV path: $$(which uv)"; \
	fi
	@echo "Git version: $$(git --version)"
	@echo "Current branch: $$(git rev-parse --abbrev-ref HEAD)"
	@echo "Working directory: $$(pwd)"
	@echo "Virtual environment: $${VIRTUAL_ENV:-Not activated}"

# Quality gate for CI/CD
quality-gate: ## Quality gate check (for CI/CD)
	@echo "$(RED)🚪 Running quality gate checks...$(NC)"
	@echo "$(BLUE)Step 1/5: Linting...$(NC)"
	@$(MAKE) ci-lint
	@echo "$(BLUE)Step 2/5: Type checking...$(NC)"
	@$(PYTHON_CMD) mypy . --install-types --non-interactive
	@echo "$(BLUE)Step 3/5: Security scanning...$(NC)"
	@$(MAKE) ci-security
	@echo "$(BLUE)Step 4/5: Testing...$(NC)"
	@$(MAKE) ci-test
	@echo "$(BLUE)Step 5/5: Coverage check...$(NC)"
	@$(PYTHON_CMD) coverage report --fail-under=80
	@echo "$(GREEN)✅ All quality gate checks passed!$(NC)"

# Team workflow shortcuts
morning: ## Morning routine - update and check
	@echo "$(GREEN)🌅 Starting morning development routine...$(NC)"
	@git pull origin main
	@$(MAKE) update
	@$(MAKE) quick

commit-ready: ## Check if code is ready for commit
	@echo "$(RED)🎯 Checking if code is ready for commit...$(NC)"
	@$(MAKE) check
	@echo "$(GREEN)✅ Code is ready for commit!$(NC)"

pr-ready: ## Check if code is ready for PR
	@echo "$(RED)🎯 Checking if code is ready for pull request...$(NC)"
	@$(MAKE) full
	@echo "$(GREEN)✅ Code is ready for pull request!$(NC)"

# Release preparation
release-check: ## Check if code is ready for release
	@echo "$(RED)🚀 Checking release readiness...$(NC)"
	@$(MAKE) quality-gate
	@$(PYTHON_CMD) interrogate . --fail-under=70
	@echo "$(GREEN)✅ Code is ready for release!$(NC)"

# Help for specific workflows
help-dev: ## Show development workflow help
	@echo "$(BOLD)Development Workflow$(NC)"
	@echo "===================="
	@echo ""
	@echo "$(BOLD)Daily Development:$(NC)"
	@echo "1. make morning          # Start of day routine"
	@echo "2. [write code]"
	@echo "3. make quick           # Quick check during development"
	@echo "4. make commit-ready    # Before committing"
	@echo ""
	@echo "$(BOLD)Before Pull Request:$(NC)"
	@echo "1. make pr-ready        # Complete check"
	@echo "2. git push origin branch-name"
	@echo ""
	@echo "$(BOLD)Common Issues:$(NC)"
	@echo "• Tests failing: make test-fast"
	@echo "• Linting errors: make lint"
	@echo "• Security issues: make security"
	@echo "• Dependency issues: make update"

help-ci: ## Show CI/CD workflow help
	@echo "$(BOLD)CI/CD Commands$(NC)"
	@echo "==============="
	@echo ""
	@echo "$(BOLD)Local CI Simulation:$(NC)"
	@echo "• make ci-lint          # Simulate CI linting"
	@echo "• make ci-test          # Simulate CI testing"
	@echo "• make ci-security      # Simulate CI security"
	@echo "• make quality-gate     # Full CI simulation"
	@echo ""
	@echo "$(BOLD)Release Preparation:$(NC)"
	@echo "• make release-check    # Comprehensive release check"
	@echo "• make deps-outdated    # Check for updates"
	@echo "• make docs             # Verify documentation"

# Advanced debugging commands
debug-deps: ## Debug dependency issues
	@echo "$(YELLOW)🔍 Debugging dependency issues...$(NC)"
	@echo "$(BOLD)Python Environment:$(NC)"
	@python -c "import sys; print(f'Python: {sys.version}'); print(f'Path: {sys.path}')"
	@echo ""
	@echo "$(BOLD)Installed Packages:$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv pip list | head -20; \
	else \
		pip list | head -20; \
	fi
	@echo ""
	@echo "$(BOLD)Virtual Environment:$(NC)"
	@echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-Not set}"
	@echo "Which Python: $(which python)"

debug-git: ## Debug git and pre-commit issues
	@echo "$(YELLOW)🔍 Debugging git and pre-commit...$(NC)"
	@echo "$(BOLD)Git Status:$(NC)"
	@git status --porcelain
	@echo ""
	@echo "$(BOLD)Git Config:$(NC)"
	@git config --list | grep -E "(user\.|core\.)" | head -10
	@echo ""
	@echo "$(BOLD)Pre-commit Status:$(NC)"
	@$(PYTHON_CMD) pre-commit --version
	@if [ -f .git/hooks/pre-commit ]; then \
		echo "✅ Pre-commit hook installed"; \
	else \
		echo "❌ Pre-commit hook not installed - run 'make setup'"; \
	fi

debug-tools: ## Debug development tools
	@echo "$(YELLOW)🔍 Debugging development tools...$(NC)"
	@echo "$(BOLD)Tool Versions:$(NC)"
	@echo "Ruff: $($(PYTHON_CMD) ruff --version || echo 'Not installed')"
	@echo "MyPy: $($(PYTHON_CMD) mypy --version || echo 'Not installed')"
	@echo "Pytest: $($(PYTHON_CMD) pytest --version || echo 'Not installed')"
	@echo "Bandit: $($(PYTHON_CMD) bandit --version || echo 'Not installed')"
	@echo "Safety: $($(PYTHON_CMD) safety --version || echo 'Not installed')"

# Performance monitoring
perf-test: ## Run performance tests
	@echo "$(YELLOW)⚡ Running performance tests...$(NC)"
	@$(PYTHON_CMD) pytest tests/ -m perf --benchmark-sort=mean

memory-test: ## Check for memory leaks
	@echo "$(YELLOW)🧠 Running memory leak tests...$(NC)"
	@$(PYTHON_CMD) pytest tests/ -m memory --memray

# Documentation generation
docs-build: ## Build documentation
	@echo "$(GREEN)📖 Building documentation...$(NC)"
	@if [ -d docs ]; then \
		cd docs && make html; \
	else \
		echo "No docs directory found. Creating basic structure..."; \
		mkdir -p docs; \
		echo "# Project Documentation\n\nGenerated documentation goes here." > docs/README.md; \
	fi

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)📖 Serving documentation...$(NC)"
	@if [ -d docs/_build/html ]; then \
		cd docs/_build/html && python -m http.server 8080; \
	else \
		echo "Documentation not built. Run 'make docs-build' first."; \
	fi

# Container and deployment
container-test: ## Test in container environment
	@echo "$(BLUE)🐳 Testing in container...$(NC)"
	@docker build -t genai-test . && docker run --rm genai-test make test

deploy-check: ## Pre-deployment checks
	@echo "$(RED)🚀 Running pre-deployment checks...$(NC)"
	@$(MAKE) quality-gate
	@echo "$(BLUE)Checking for secrets...$(NC)"
	@$(PYTHON_CMD) detect-secrets scan --baseline .secrets.baseline
	@echo "$(BLUE)Checking dependencies for vulnerabilities...$(NC)"
	@$(PYTHON_CMD) safety check --json
	@echo "$(GREEN)✅ Deployment checks passed!$(NC)"

# Team collaboration
sync-hooks: ## Sync pre-commit hooks across team
	@echo "$(BLUE)🔄 Syncing pre-commit hooks...$(NC)"
	@$(PYTHON_CMD) pre-commit install
	@$(PYTHON_CMD) pre-commit autoupdate
	@$(PYTHON_CMD) pre-commit run --all-files || echo "Some files were updated by hooks"

team-setup: ## Setup for new team members
	@echo "$(GREEN)👥 Setting up for new team member...$(NC)"
	@$(MAKE) setup
	@$(MAKE) help-dev
	@echo ""
	@echo "$(BOLD)$(GREEN)🎉 Setup complete! Next steps:$(NC)"
	@echo "1. Review DEVELOPMENT_GUIDE.md"
	@echo "2. Run 'make morning' to start developing"
	@echo "3. Ask team for access to shared resources (API keys, etc.)"

# Maintenance and cleanup
deep-clean: ## Deep clean all generated files
	@echo "$(YELLOW)🧹 Deep cleaning project...$(NC)"
	@$(MAKE) clean
	@echo "Removing additional cache directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	@echo "$(GREEN)✅ Deep clean completed!$(NC)"

reset-env: ## Reset virtual environment
	@echo "$(RED)🔄 Resetting virtual environment...$(NC)"
	@if [ -d .venv ]; then \
		rm -rf .venv; \
		echo "Removed .venv directory"; \
	fi
	@if command -v uv >/dev/null 2>&1; then \
		uv venv; \
		uv sync --dev; \
	else \
		python -m venv .venv; \
		. .venv/bin/activate && pip install -e .[dev]; \
	fi
	@$(MAKE) setup
	@echo "$(GREEN)✅ Environment reset completed!$(NC)"

# Troubleshooting
troubleshoot: ## Run troubleshooting diagnostics
	@echo "$(YELLOW)🔧 Running troubleshooting diagnostics...$(NC)"
	@echo ""
	@$(MAKE) env-info
	@echo ""
	@$(MAKE) debug-deps
	@echo ""
	@$(MAKE) debug-git
	@echo ""
	@$(MAKE) debug-tools
	@echo ""
	@echo "$(BOLD)Common Solutions:$(NC)"
	@echo "• Environment issues: make reset-env"
	@echo "• Tool issues: make update"
	@echo "• Git issues: make sync-hooks"
	@echo "• Cache issues: make deep-clean"

# Special targets for different environments
dev-install: ## Install development dependencies only
	@echo "$(GREEN)📦 Installing development dependencies...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --dev; \
	else \
		pip install -e ".[dev]"; \
	fi

prod-install: ## Install production dependencies only
	@echo "$(GREEN)📦 Installing production dependencies...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --no-dev; \
	else \
		pip install -e .; \
	fi

# Version and release management
version: ## Show current version
	@echo "$(BLUE)📦 Current version information:$(NC)"
	@echo "Project version: $(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || echo "Could not read version")"
	@echo "Git tag: $(git describe --tags --abbrev=0 2>/dev/null || echo "No tags found")"
	@echo "Git commit: $(git rev-parse --short HEAD)"

# IDE and editor setup
vscode-setup: ## Setup VS Code configuration
	@echo "$(BLUE)💻 Setting up VS Code configuration...$(NC)"
	@mkdir -p .vscode
	@cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true,
            "source.fixAll": true
        }
    },
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/.ruff_cache": true,
        "**/htmlcov": true
    }
}
EOF
	@echo "$(GREEN)✅ VS Code configuration created!$(NC)"

# Final catch-all for unknown targets
%:
	@echo "$(RED)❌ Unknown target: $@$(NC)"
	@echo "Run '$(BOLD)make help$(NC)' to see available commands."
	@exit 1
