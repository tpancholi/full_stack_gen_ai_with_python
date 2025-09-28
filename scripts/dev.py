#!/usr/bin/env python3
"""
Development workflow automation script for GenAI Python projects.

This script provides common development tasks in a single command.
Usage: python scripts/dev.py [command]

Commands:
  setup     - Initial project setup
  check     - Run all code quality checks
  test      - Run tests with coverage
  lint      - Run linting and formatting
  security  - Run security checks
  docs      - Generate documentation coverage report
  clean     - Clean up temporary files
  full      - Run all checks (pre-commit simulation)
  install   - Install all dependencies
  update    - Update dependencies
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    END = "\033[0m"


class DevWorkflow:
    """Main development workflow automation class."""

    def __init__(self) -> None:
        self.root_dir = Path(__file__).parent.parent
        self.python_cmd = "uv run" if self._has_uv() else "python -m"

    def _has_uv(self) -> bool:
        """Check if uv is available."""
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)  # noqa: S607
            return True  # noqa: TRY300
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _run_command(
        self,
        *,
        command: str,
        description: str,
        check: bool = True,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a shell command with nice output."""
        print(f"{Colors.BLUE}🔧 {description}...{Colors.END}")
        print(f"{Colors.CYAN}   Running: {command}{Colors.END}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                check=check,
                cwd=cwd or self.root_dir,
                text=True,
                capture_output=False,
            )
            if result.returncode == 0:
                print(f"{Colors.GREEN}✅ {description} completed successfully{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠️  {description} completed with warnings{Colors.END}")
            return result  # noqa: TRY300
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}❌ {description} failed with exit code {e.returncode}{Colors.END}")
            if check:
                raise
            return e

    def _print_header(self, title: str) -> None:
        """Print a formatted header."""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'=' * 60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.MAGENTA} {title.upper()}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'=' * 60}{Colors.END}\n")

    def setup(self) -> None:
        """Initial project setup."""
        self._print_header("Project Setup")

        # Install dependencies
        if self._has_uv():
            self._run_command(command="uv sync --dev", description="Installing dependencies with uv")
        else:
            self._run_command(command="pip install -e .", description="Installing project in development mode")

        # Setup pre-commit
        self._run_command(command=f"{self.python_cmd} pre-commit install", description="Installing pre-commit hooks")

        # Initialize secrets baseline
        secrets_baseline = self.root_dir / ".secrets.baseline"
        if not secrets_baseline.exists():
            self._run_command(
                command=f"{self.python_cmd} detect-secrets scan --baseline .secrets.baseline",
                description="Creating secrets detection baseline",
            )

        # Create necessary directories
        for directory in ["tests", "docs", "scripts", "src"]:
            Path(self.root_dir / directory).mkdir(exist_ok=True)

        print(f"\n{Colors.GREEN}🎉 Project setup completed!{Colors.END}")
        print(f"{Colors.CYAN}Next steps:{Colors.END}")
        print("  1. Review and update pyproject.toml configuration")
        print("  2. Run 'python scripts/dev.py test' to verify setup")
        print("  3. Start developing!")

    def lint(self) -> None:
        """Run linting and formatting."""
        self._print_header("Code Linting & Formatting")

        # Ruff linting and formatting
        self._run_command(
            f"{self.python_cmd} ruff check . --fix --show-fixes",
            "Running Ruff linting with auto-fix",
        )
        self._run_command(f"{self.python_cmd} ruff format .", "Running Ruff formatting")

        # Import sorting
        self._run_command(f"{self.python_cmd} isort .", "Sorting imports with isort")

    def type_check(self) -> None:
        """Run type checking."""
        self._print_header("Type Checking")

        # MyPy type checking
        self._run_command(
            f"{self.python_cmd} mypy . --install-types --non-interactive",
            "Running MyPy type checking",
            check=False,
        )

        # Pyright type checking (if available)
        try:
            self._run_command("pyright", "Running Pyright type checking", check=False)
        except subprocess.CalledProcessError:
            print(f"{Colors.YELLOW}⚠️  Pyright not available, skipping{Colors.END}")

    def test(self) -> None:
        """Run tests with coverage."""
        self._print_header("Running Tests")

        # Create tests directory if it doesn't exist
        tests_dir = self.root_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        # Create a basic test file if none exist
        if not any(tests_dir.glob("test_*.py")):
            basic_test = tests_dir / "test_basic.py"
            basic_test.write_text('''"""Basic test to ensure testing setup works."""

def test_basic():
	"""Basic test that always passes."""
	assert True


def test_imports():
	"""Test that we can import common libraries."""
	import sys
	assert sys.version_info >= (3, 13)
''')

        # Run tests
        self._run_command(
            f"{self.python_cmd} pytest -v --cov --cov-report=term-missing --cov-report=html",
            "Running tests with coverage",
        )

    def security(self) -> None:
        """Run security checks."""
        self._print_header("Security Scanning")

        # Bandit security scanning
        self._run_command(
            f"{self.python_cmd} bandit -r . -f json -c pyproject.toml",
            "Running Bandit security scan",
            check=False,
        )

        # Secret detection
        self._run_command(
            f"{self.python_cmd} detect-secrets scan --baseline .secrets.baseline",
            "Scanning for secrets",
            check=False,
        )

        # Safety check for known vulnerabilities
        self._run_command(
            f"{self.python_cmd} safety check --json",
            "Checking for known vulnerabilities",
            check=False,
        )

    def docs(self) -> None:
        """Generate documentation coverage report."""
        self._print_header("Documentation Coverage")

        self._run_command(
            f"{self.python_cmd} interrogate . --config pyproject.toml",
            "Checking docstring coverage",
        )

    def dead_code(self) -> None:
        """Check for dead code."""
        self._print_header("Dead Code Detection")

        self._run_command(
            f"{self.python_cmd} vulture . --min-confidence 80",
            "Scanning for dead code",
            check=False,
        )

    def clean(self) -> None:
        """Clean up temporary files."""
        self._print_header("Cleaning Temporary Files")

        cleanup_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/*.egg-info",
            ".coverage",
            "htmlcov",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "dist",
            "build",
            "*.egg-info",
        ]

        for pattern in cleanup_patterns:
            for path in self.root_dir.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"   Removed directory: {path}")
                elif path.is_file():
                    path.unlink()
                    print(f"   Removed file: {path}")

        print(f"{Colors.GREEN}✅ Cleanup completed{Colors.END}")

    def check(self) -> None:
        """Run all code quality checks."""
        self._print_header("Running All Quality Checks")

        checks = [
            ("Linting & Formatting", self.lint),
            ("Type Checking", self.type_check),
            ("Security Scanning", self.security),
            ("Running Tests", self.test),
            ("Documentation Coverage", self.docs),
        ]

        failed_checks = []

        for check_name, check_func in checks:
            try:
                check_func()
            except subprocess.CalledProcessError:
                failed_checks.append(check_name)
                print(f"{Colors.RED}❌ {check_name} failed{Colors.END}")

        if failed_checks:
            print(f"\n{Colors.RED}❌ Some checks failed:{Colors.END}")
            for check in failed_checks:
                print(f"   - {check}")
            sys.exit(1)
        else:
            print(f"\n{Colors.GREEN}🎉 All quality checks passed!{Colors.END}")

    def full(self) -> None:
        """Run full pre-commit simulation."""
        self._print_header("Full Pre-commit Simulation")

        self._run_command("pre-commit run --all-files", "Running all pre-commit hooks")

        # Run additional manual checks
        self.dead_code()
        self.docs()

    def install(self) -> None:
        """Install dependencies."""
        self._print_header("Installing Dependencies")

        if self._has_uv():
            self._run_command("uv sync --dev", "Syncing dependencies with uv")
        else:
            self._run_command("pip install -e .[dev]", "Installing with pip")

    def update(self) -> None:
        """Update dependencies."""
        self._print_header("Updating Dependencies")

        if self._has_uv():
            self._run_command("uv sync --dev --upgrade", "Updating dependencies with uv")
        else:
            self._run_command("pip install --upgrade -e .[dev]", "Updating with pip")

        # Update pre-commit hooks
        self._run_command("pre-commit autoupdate", "Updating pre-commit hooks")


def main() -> NoReturn:
    """Main entry point."""
    workflow = DevWorkflow()

    parser = argparse.ArgumentParser(description="Development workflow automation for GenAI Python projects")
    parser.add_argument(
        "command",
        choices=[
            "setup",
            "check",
            "test",
            "lint",
            "security",
            "docs",
            "clean",
            "full",
            "install",
            "update",
            "dead-code",
        ],
        help="Command to execute",
    )

    args = parser.parse_args()

    # Map commands to methods
    command_map = {
        "setup": workflow.setup,
        "check": workflow.check,
        "test": workflow.test,
        "lint": workflow.lint,
        "security": workflow.security,
        "docs": workflow.docs,
        "clean": workflow.clean,
        "full": workflow.full,
        "install": workflow.install,
        "update": workflow.update,
        "dead-code": workflow.dead_code,
    }

    try:
        command_map[args.command]()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Operation cancelled by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()
