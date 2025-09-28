"""Basic test to ensure testing setup works."""

from __future__ import annotations

import sys


def test_basic():
    """Basic test that always passes."""
    assert True


def test_imports():
    """Test that we can import common libraries."""

    assert sys.version_info >= (3, 13)
