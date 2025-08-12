import base64
import numpy as np
from typing import List, Dict, Any, Union
from datetime import datetime


def encode_image_base64(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def compute_regression_line(x: List[float], y: List[float], **kwargs) -> dict:
    if len(x) != len(y):
        raise ValueError("Input lists must have the same length.")
    if len(x) < 2:
        raise ValueError("Input lists must have at least two points.")

    try:
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)

        m, c = np.polyfit(x_arr, y_arr, 1)

        return {"slope": m, "intercept": c}
    except Exception as e:
        raise Exception(f"Failed to compute regression: {e}")


def compute_correlation(x: List[float], y: List[float], **kwargs) -> float:
    if len(x) != len(y):
        raise ValueError("Input lists must have the same length.")
    arr_x = np.array(x, dtype=float)
    arr_y = np.array(y, dtype=float)
    corr_matrix = np.corrcoef(arr_x, arr_y)
    return float(corr_matrix[0, 1])


def summary_statistics(data: List[float], **kwargs) -> dict:
    arr = np.array(data, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def parse_dates(dates: List[str], date_format: str = None, **kwargs) -> List[str]:
    """Parse list of date strings into ISO format strings."""
    parsed = []
    for d in dates:
        try:
            dt = (
                datetime.fromisoformat(d)
                if date_format is None
                else datetime.strptime(d, date_format)
            )
            parsed.append(dt.isoformat())
        except Exception as e:
            raise ValueError(f"Failed to parse date '{d}': {e}")
    return parsed


def generate_fake_response(expected_format: Union[Dict, List, Any]) -> Any:
    if isinstance(expected_format, dict):
        fake_data = {}
        for key, value in expected_format.items():
            fake_data[key] = generate_fake_response(value)
        return fake_data
    elif isinstance(expected_format, list):
        if not expected_format:
            return []
        return [generate_fake_response(expected_format[0])]
    elif isinstance(expected_format, str):
        if "data:image" in expected_format:
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        return "..."
    elif isinstance(expected_format, int):
        return 0
    elif isinstance(expected_format, float):
        return 0.0
    else:
        return None
