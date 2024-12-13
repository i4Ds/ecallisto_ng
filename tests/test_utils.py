from datetime import datetime, timezone, timedelta
from ecallisto_ng.data_download.utils import to_naive_utc


def test_to_naive_utc_with_timezone():
    # Create a timezone-aware datetime (e.g., UTC+2)
    aware_dt = datetime(2024, 12, 11, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    # Convert to naive UTC
    result = to_naive_utc(aware_dt)
    # Expected naive UTC datetime
    expected = datetime(2024, 12, 11, 10, 0)  # UTC+2 converted to UTC
    assert result == expected
    assert result.tzinfo is None


def test_to_naive_utc_already_naive():
    # Create a naive datetime
    naive_dt = datetime(2024, 12, 11, 12, 0)
    # Convert (should remain unchanged)
    result = to_naive_utc(naive_dt)
    assert result == naive_dt
    assert result.tzinfo is None


def test_to_naive_utc_utc_timezone():
    # Create a timezone-aware datetime in UTC
    aware_utc_dt = datetime(2024, 12, 11, 12, 0, tzinfo=timezone.utc)
    # Convert to naive UTC
    result = to_naive_utc(aware_utc_dt)
    # Expected naive UTC datetime
    expected = datetime(2024, 12, 11, 12, 0)  # Should stay the same
    assert result == expected
    assert result.tzinfo is None
