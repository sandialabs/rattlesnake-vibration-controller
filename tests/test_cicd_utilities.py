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
    get_multiline_timestamp,
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
    # YYYY-MM-DD HH:MM:SS UTC (YYYY-MM-DD HH:MM:SS EST / YYYY-MM-DD HH:MM:SS MST / YYYY-MM-DD HH:MM:SS PST)
    pattern: str = (
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC "
        r"\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} EST / "
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} MST / "
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} PST\)$"
    )
    assert re.match(pattern, timestamp_str) is not None


def test_get_timestamp_timezone_offsets():
    """Test that the timezone offsets in the timestamp are correct."""
    timestamp_str: str = get_timestamp()

    # Extract UTC, EST, MST, and PST timestamps from the string
    utc_str_part: str = timestamp_str.split(" UTC")[0]
    est_str_part: str = re.search(
        r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) EST", timestamp_str
    ).group(1)
    mst_str_part: str = re.search(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) MST", timestamp_str
    ).group(1)
    pst_str_part: str = re.search(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) PST", timestamp_str
    ).group(1)

    # Convert string parts to datetime objects
    utc_time: datetime = datetime.strptime(utc_str_part, "%Y-%m-%d %H:%M:%S")
    est_time: datetime = datetime.strptime(est_str_part, "%Y-%m-%d %H:%M:%S")
    mst_time: datetime = datetime.strptime(mst_str_part, "%Y-%m-%d %H:%M:%S")
    pst_time: datetime = datetime.strptime(pst_str_part, "%Y-%m-%d %H:%M:%S")

    # Check the differences
    utc_est_diff_hours: int = int((utc_time - est_time).total_seconds() / 3600)
    utc_mst_diff_hours: int = int((utc_time - mst_time).total_seconds() / 3600)
    utc_pst_diff_hours: int = int((utc_time - pst_time).total_seconds() / 3600)

    # EST is typically UTC-4, MST is UTC-6, and PST is UTC-7 during Daylight Saving Time.
    # Note: These values can change with daylight saving, so we test for common offsets.
    assert utc_est_diff_hours in [4, 5]
    assert utc_mst_diff_hours in [6, 7]
    assert utc_pst_diff_hours in [7, 8]


# endregion


# region: Test extend_timestamp


def test_extend_timestamp():
    """
    Test that extend_timestamp correctly converts a short timestamp string
    to the extended format with all timezones.
    """
    # Example from the docstring:
    short_ts: str = "20250815_211112_UTC"
    expected_extended_ts: str = (
        "2025-08-15 21:11:12 UTC (2025-08-15 17:11:12 EST / 2025-08-15 15:11:12 MST / 2025-08-15 14:11:12 PST)"
    )

    # For a robust test, we can check the time component
    # directly.

    # First, test with the given example
    assert extend_timestamp(short_ts) == expected_extended_ts

    # Now, test with a different time
    short_ts2: str = "20241027_100000_UTC"
    # UTC 10:00:00 is EST 06:00:00 (due to DST), MST 04:00:00 (due to DST), and PST 03:00:00 (due to DST)
    # Note: On 27 Oct 2024, EST is still UTC-4, MST is still UTC-6, and PST is still UTC-7.
    expected_extended_ts2: str = (
        "2024-10-27 10:00:00 UTC (2024-10-27 06:00:00 EST / 2024-10-27 04:00:00 MST / 2024-10-27 03:00:00 PST)"
    )
    assert extend_timestamp(short_ts2) == expected_extended_ts2

    # Simpler and more robust test: check for timezone offsets.
    extended_ts: str = extend_timestamp("20250815_211112_UTC")

    # Extract UTC time from the result
    utc_part: str = extended_ts.split(" UTC")[0]
    utc_dt: datetime = datetime.strptime(utc_part, "%Y-%m-%d %H:%M:%S")

    # Extract EST, MST, and PST times from the result
    est_part: str = re.search(
        r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", extended_ts
    ).group(1)
    mst_part: str = re.search(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) MST", extended_ts
    ).group(1)
    pst_part: str = re.search(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) PST", extended_ts
    ).group(1)

    est_dt: datetime = datetime.strptime(est_part, "%Y-%m-%d %H:%M:%S")
    mst_dt: datetime = datetime.strptime(mst_part, "%Y-%m-%d %H:%M:%S")
    pst_dt: datetime = datetime.strptime(pst_part, "%Y-%m-%d %H:%M:%S")

    # Test time differences
    assert (utc_dt - est_dt).total_seconds() / 3600 in [4, 5]
    assert (utc_dt - mst_dt).total_seconds() / 3600 in [6, 7]
    assert (utc_dt - pst_dt).total_seconds() / 3600 in [7, 8]


def test_extend_timestamp_invalid_format():
    """Test that extend_timestamp handles an invalid input format gracefully."""
    with pytest.raises(ValueError):
        extend_timestamp("invalid_string")
    with pytest.raises(ValueError):
        extend_timestamp("20250815_211112")  # Missing time zone


# endregion


# region: Test get_multiline_timestamp


def test_get_multiline_timestamp():
    """Test the multiline timestamp generation."""
    short_ts = "20240324_204412_UTC"
    lines = get_multiline_timestamp(short_ts)

    assert len(lines) == 5
    assert lines[0] == "Generated:"
    assert lines[1] == "2024-03-24 20:44:12 UTC"
    # On March 24, it is Daylight Saving Time (EDT, MDT, and PDT).
    # EST/MST/PST are used as fixed strings in the utility function for now as requested.
    assert lines[2] == "2024-03-24 16:44:12 EST"
    assert lines[3] == "2024-03-24 14:44:12 MST"
    assert lines[4] == "2024-03-24 13:44:12 PST"


def test_get_multiline_timestamp_rollover():
    """Test date rollover across timezones."""
    # 2:44 AM UTC on March 25 is 10:44 PM EST, 8:44 PM MST, and 7:44 PM PST on March 24
    short_ts = "20240325_024412_UTC"
    lines = get_multiline_timestamp(short_ts)

    assert len(lines) == 5
    assert lines[0] == "Generated:"
    assert lines[1] == "2024-03-25 02:44:12 UTC"
    assert lines[2] == "2024-03-24 22:44:12 EST"
    assert lines[3] == "2024-03-24 20:44:12 MST"
    assert lines[4] == "2024-03-24 19:44:12 PST"


def test_get_multiline_timestamp_invalid():
    """Test invalid format for multiline timestamp."""
    with pytest.raises(ValueError):
        get_multiline_timestamp("invalid")


# endregion
