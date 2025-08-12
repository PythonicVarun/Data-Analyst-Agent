import pytest
import numpy as np
from app.tools.utils import (
    encode_image_base64,
    compute_regression_line,
    compute_correlation,
    summary_statistics,
    parse_dates,
    generate_fake_response,
)


def test_encode_image_base64():
    """
    Test that image bytes are correctly encoded to a base64 data URI.
    """
    img_bytes = b"test_image_bytes"
    result = encode_image_base64(img_bytes)
    assert result.startswith("data:image/png;base64,")
    import base64

    assert base64.b64decode(result.split(",")[1]) == img_bytes


def test_compute_regression_line():
    """
    Test the computation of a linear regression line.
    """
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    result = compute_regression_line(x, y)
    assert isinstance(result, dict)
    assert "slope" in result
    assert "intercept" in result
    np.testing.assert_almost_equal(result["slope"], 0.6)
    np.testing.assert_almost_equal(result["intercept"], 2.2)


def test_compute_regression_line_value_error():
    """
    Test that compute_regression_line raises ValueError for invalid input.
    """
    with pytest.raises(ValueError):
        compute_regression_line([1], [2, 3])
    with pytest.raises(ValueError):
        compute_regression_line([1], [2])


def test_compute_correlation():
    """
    Test the computation of the correlation coefficient.
    """
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    result = compute_correlation(x, y)
    assert isinstance(result, float)
    np.testing.assert_almost_equal(result, 0.7745966692414834)


def test_summary_statistics():
    """
    Test the computation of summary statistics.
    """
    data = [1, 2, 3, 4, 5]
    result = summary_statistics(data)
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
    assert result["std"] == pytest.approx(1.41421356)
    assert result["min"] == 1.0
    assert result["max"] == 5.0


def test_parse_dates():
    """
    Test parsing of date strings.
    """
    dates = ["2023-01-01", "2023-01-02T12:00:00"]
    result = parse_dates(dates)
    assert result == ["2023-01-01T00:00:00", "2023-01-02T12:00:00"]


def test_parse_dates_with_format():
    """
    Test parsing of date strings with a specific format.
    """
    dates = ["01/01/2023", "02/01/2023"]
    result = parse_dates(dates, date_format="%d/%m/%Y")
    assert result == ["2023-01-01T00:00:00", "2023-01-02T00:00:00"]


def test_generate_fake_response():
    """
    Test generation of fake responses based on a format.
    """
    format_dict = {"key1": "string", "key2": 123, "key3": [1.0]}
    response = generate_fake_response(format_dict)
    assert response["key1"] == "..."
    assert response["key2"] == 0
    assert response["key3"] == [0.0]

    format_list = [{"name": "string", "value": 0}]
    response = generate_fake_response(format_list)
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0]["name"] == "..."
    assert response[0]["value"] == 0

    format_image = "data:image/png;base64,..."
    response = generate_fake_response(format_image)
    assert response.startswith("data:image/png;base64,")
