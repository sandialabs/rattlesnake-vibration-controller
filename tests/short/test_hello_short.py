"""
Minimum Working Example of a Test Framework

This module provides a basic test to verify that the project's testing
infrastructure is working correctly.
"""

from rattlesnake import hello

# import pytest  # unused import


def test_greet():
    """
    Test the greet function from the short path.
    """
    print("tests/short/test_hello_short.py")
    assert hello.greet("World") == "Hello, World!"
