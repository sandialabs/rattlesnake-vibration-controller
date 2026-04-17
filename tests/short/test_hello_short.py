"""This is a minimum working example of a test framework."""

import pytest

from rattlesnake import hello


def test_greet():
    """Test the greet function."""
    print("tests/short/test_hello_short.py")
    assert hello.greet("World") == "Hello, World!"
