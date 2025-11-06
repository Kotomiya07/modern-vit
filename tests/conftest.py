"""Pytest configuration and fixtures."""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from project_name.core.example import ExampleClass, ExampleConfig


@pytest.fixture
def example_config() -> ExampleConfig:
    """Create a test configuration."""
    return ExampleConfig(
        name="test",
        max_items=10,
        enable_validation=True,
    )


@pytest.fixture
def example_instance(example_config: ExampleConfig) -> ExampleClass:
    """Create a test ExampleClass instance."""
    return ExampleClass(example_config)


@pytest.fixture
def sample_data() -> list[dict[str, Any]]:
    """Create sample data for testing."""
    return [
        {"id": 1, "name": "Item 1", "value": 100},
        {"id": 2, "name": "Item 2", "value": 200},
        {"id": 3, "name": "Item 3", "value": 300},
    ]


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment variables for each test."""
    # Remove any test-specific environment variables
    test_env_vars = [var for var in os.environ if var.startswith("TEST_")]
    for var in test_env_vars:
        monkeypatch.delenv(var, raising=False)
