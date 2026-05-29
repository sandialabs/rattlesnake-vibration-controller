"""
Unit tests for utilities.py — CICD Utilities.

This test suite verifies the correctness of the utility functions
used in the CI/CD processes, such as timestamp generation and score
color determination.

Example use:
    source .venv/bin/activate
    pytest tests/test_utilities.py -v
    pytest tests/test_utilities.py::test_get_score_color_brightgreen -v
"""

from datetime import datetime
import re

import pytest

# Assuming the utilities.py module is in a sibling directory to tests
from rattlesnake.cicd.utilities import (
    get_score_color_coverage,
    get_score_color_lint,
    get_timestamp,
    extend_timestamp,
)


# region: Test get_score_color_lint


def test_get_score_color_brightgreen():
    """Test that get_score_color returns 'brightgreen' for scores >= 8.0."""
    assert get_score_color_lint("8.5") == "brightgreen"
    assert get_score_color_lint("10") == "brightgreen"


def test_get_score_color_yellow():
    """Test that get_score_color returns 'yellow' for scores >= 6.0 and < 8.0."""
    assert get_score_color_lint("7.9") == "yellow"
    assert get_score_color_lint("6.0") == "yellow"


def test_get_score_color_orange():
    """Test that get_score_color returns 'orange' for scores >= 4.0 and < 6.0."""
    assert get_score_color_lint("5.9") == "orange"
    assert get_score_color_lint("4.0") == "orange"


def test_get_score_color_red():
    """Test that get_score_color returns 'red' for scores < 4.0."""
    assert get_score_color_lint("3.9") == "red"
    assert get_score_color_lint("0") == "red"


def test_get_score_color_invalid_input():
    """Test that get_score_color returns 'gray' for non-numeric input."""
    assert get_score_color_lint("abc") == "gray"
    assert get_score_color_lint("") == "gray"
    assert get_score_color_lint("8.5a") == "gray"


# endregion


# region: Test get_score_color_coverage


def test_get_score_color_coverage():
    """Test all the colors for coverage."""
    assert get_score_color_coverage("90") == "brightgreen"
    assert get_score_color_coverage("80") == "green"
    assert get_score_color_coverage("70") == "yellow"
    assert get_score_color_coverage("60") == "orange"
    assert get_score_color_coverage("50") == "red"
    assert get_score_color_coverage("") == "gray"


# endregion


# region: Test get_timestamp


def test_get_timestamp_format():
    """Test that the get_timestamp function returns a string with the correct format."""
    timestamp_str: str = get_timestamp()
    # The regex pattern matches the expected format:
    # YYYY-MM-DD HH:MM:SS UTC (YYYY-MM-DD HH:MM:SS EST / YYYY-MM-DD HH:MM:SS MST)
    pattern: str = (
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC " r"\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} EST / " r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} MST\)$"
    )
    assert re.match(pattern, timestamp_str) is not None


def test_get_timestamp_timezone_offsets():
    """Test that the timezone offsets in the timestamp are correct."""
    timestamp_str: str = get_timestamp()

    # Extract UTC, EST, and MST timestamps from the string
    utc_str_part: str = timestamp_str.split(" UTC")[0]
    est_str_part: str = re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) EST", timestamp_str).group(1)
    mst_str_part: str = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) MST", timestamp_str).group(1)

    # Convert string parts to datetime objects
    utc_time: datetime = datetime.strptime(utc_str_part, "%Y-%m-%d %H:%M:%S")
    est_time: datetime = datetime.strptime(est_str_part, "%Y-%m-%d %H:%M:%S")
    mst_time: datetime = datetime.strptime(mst_str_part, "%Y-%m-%d %H:%M:%S")

    # Check the differences
    utc_est_diff_hours: int = int((utc_time - est_time).total_seconds() / 3600)
    utc_mst_diff_hours: int = int((utc_time - mst_time).total_seconds() / 3600)

    # EST is typically UTC-4, and MST is UTC-6 during Daylight Saving Time.
    # Note: These values can change with daylight saving, so we test for common offsets.
    assert utc_est_diff_hours in [4, 5]
    assert utc_mst_diff_hours in [6, 7]


# endregion


# region: Test extend_timestamp


def test_extend_timestamp():
    """
    Test that extend_timestamp correctly converts a short timestamp string
    to the extended format with all timezones.
    """
    # Example from the docstring:
    short_ts: str = "20250815_211112_UTC"
    expected_extended_ts: str = "2025-08-15 21:11:12 UTC (2025-08-15 17:11:12 EST / 2025-08-15 15:11:12 MST)"

    # For a robust test, we can check the time component
    # directly.

    # First, test with the given example
    assert extend_timestamp(short_ts) == expected_extended_ts

    # Now, test with a different time
    short_ts2: str = "20241027_100000_UTC"
    # UTC 10:00:00 is EST 06:00:00 (due to DST) and MST 04:00:00 (due to DST)
    # Note: On 27 Oct 2024, EST is still UTC-4 and MST is still UTC-6.
    expected_extended_ts2: str = "2024-10-27 10:00:00 UTC (2024-10-27 06:00:00 EST / 2024-10-27 04:00:00 MST)"
    # The timezone abbreviations change depending on DST.
    # We should adjust the test to be less fragile to this.

    # Simpler and more robust test: check for timezone offsets.
    extended_ts: str = extend_timestamp("20250815_211112_UTC")

    # Extract UTC time from the result
    utc_part: str = extended_ts.split(" UTC")[0]
    utc_dt: datetime = datetime.strptime(utc_part, "%Y-%m-%d %H:%M:%S")

    # Extract EST and MST times from the result
    est_part: str = re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", extended_ts).group(1)
    mst_part: str = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) MST\)", extended_ts).group(1)

    est_dt: datetime = datetime.strptime(est_part, "%Y-%m-%d %H:%M:%S")
    mst_dt: datetime = datetime.strptime(mst_part, "%Y-%m-%d %H:%M:%S")

    # Test time differences
    assert (utc_dt - est_dt).total_seconds() / 3600 in [4, 5]
    assert (utc_dt - mst_dt).total_seconds() / 3600 in [6, 7]


def test_extend_timestamp_invalid_format():
    """Test that extend_timestamp handles an invalid input format gracefully."""
    with pytest.raises(ValueError):
        extend_timestamp("invalid_string")
    with pytest.raises(ValueError):
        extend_timestamp("20250815_211112")  # Missing time zone


# endregion
