"""This is a minimum working example of a test framework."""

import pytest

from rattlesnake import hello


def test_greet():
    """Test the greet function from the long path."""
    print("tests/long/test_hello_long.py")
    assert hello.greet("World") == "Hello, World!"
