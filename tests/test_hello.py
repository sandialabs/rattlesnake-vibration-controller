"""This is a minimum working example of a test framework."""

# import pytest

from rattlesnake import hello


def test_greet():
    """Test the gret function."""
    assert hello.greet("World") == "Hello, World!"
