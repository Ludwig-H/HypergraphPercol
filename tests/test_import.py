import importlib

import pytest


def test_import_package():
    try:
        module = importlib.import_module("hypergraphpercol")
    except ModuleNotFoundError as exc:  # pragma: no cover
        pytest.skip(f"Missing dependency: {exc.name}")
    assert hasattr(module, "HypergraphPercol")
