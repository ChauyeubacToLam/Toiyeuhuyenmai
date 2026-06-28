from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class DataLoadResult:
    path: str
    frame: pd.DataFrame
    warnings: list[str]


def normalize_column_name(value: object) -> str:
    return str(value).replace("\ufeff", "").strip() or "Unnamed"


def make_unique_columns(columns: Iterable[object]) -> tuple[list[str], list[str]]:
    seen: dict[str, int] = {}
    result: list[str] = []
    warnings: list[str] = []

    for raw in columns:
        name = normalize_column_name(raw)
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            new_name = f"{name}_{count + 1}"
            warnings.append(f"Tên biến bị trùng: '{name}' đã đổi thành '{new_name}'.")
            result.append(new_name)
        else:
            result.append(name)

    return result, warnings


def read_dataset(path: str) -> DataLoadResult:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    warnings: list[str] = []

    if suffix in {".csv", ".txt"}:
        try:
            frame = pd.read_csv(file_path, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(file_path, sep=None, engine="python")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(file_path)
    elif suffix == ".sav":
        frame = pd.read_spss(file_path)
    else:
        raise ValueError("Định dạng tệp chưa hỗ trợ. Hãy dùng CSV, TXT, XLS, XLSX hoặc SAV.")

    new_columns, column_warnings = make_unique_columns(frame.columns)
    frame.columns = new_columns
    warnings.extend(column_warnings)

    if frame.empty:
        warnings.append("Tệp dữ liệu đang trống.")
    if len(frame.columns) == 0:
        warnings.append("Không tìm thấy biến trong dữ liệu.")

    return DataLoadResult(path=str(file_path), frame=frame, warnings=warnings)


def coerce_numeric_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    converted_columns: dict[str, pd.Series] = {}
    warnings: list[str] = []

    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            converted_columns[column] = pd.to_numeric(series, errors="coerce")
            continue

        as_text = (
            series.astype("string")
            .str.strip()
            .str.replace(",", ".", regex=False)
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )
        converted = pd.to_numeric(as_text, errors="coerce")
        non_missing_before = int(series.notna().sum())
        non_missing_after = int(converted.notna().sum())
        if non_missing_after < non_missing_before:
            warnings.append(
                f"Phát hiện giá trị không phải số trong '{column}': "
                f"{non_missing_before - non_missing_after} giá trị đã được chuyển thành thiếu."
            )
        converted_columns[column] = converted

    numeric = pd.DataFrame(converted_columns, index=frame.index)
    return numeric, warnings


def infer_scale_type(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "không rõ"
    unique_count = values.nunique()
    if unique_count <= 2:
        return "nhị phân"
    if unique_count <= 10 and np.allclose(values, np.round(values)):
        return "thứ bậc"
    return "định lượng"


def profile_dataset(frame: pd.DataFrame, used_columns: Iterable[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    used = set(used_columns or [])
    numeric, warnings = coerce_numeric_frame(frame)
    rows: list[dict[str, object]] = []
    n_rows = max(len(frame), 1)

    for column in frame.columns:
        series = numeric[column]
        missing = int(series.isna().sum())
        non_missing = series.dropna()
        zero_variance = bool(non_missing.size > 1 and non_missing.std(ddof=1) == 0)
        if missing / n_rows >= 0.2:
            warnings.append(f"Tỷ lệ thiếu cao trong '{column}': {missing / n_rows:.1%}.")
        if zero_variance:
            warnings.append(f"Biến quan sát có phương sai bằng 0: '{column}'.")

        rows.append(
            {
                "Tên biến": column,
                "Kiểu thang đo": infer_scale_type(series),
                "Thiếu": missing,
                "Tỷ lệ thiếu": missing / n_rows,
                "Trung bình": _safe_stat(non_missing.mean()),
                "Trung vị": _safe_stat(non_missing.median()),
                "Min": _safe_stat(non_missing.min()),
                "Max": _safe_stat(non_missing.max()),
                "Độ lệch chuẩn": _safe_stat(non_missing.std(ddof=1)),
                "Độ lệch": _safe_stat(non_missing.skew()),
                "Kurtosis": _safe_stat(non_missing.kurtosis()),
                "Khoảng thang đo": _scale_hint(series),
                "Dùng trong mô hình": "Có" if column in used else "",
            }
        )

    return pd.DataFrame(rows), warnings


def prepare_analysis_frame(
    frame: pd.DataFrame,
    columns: Iterable[str],
    missing_strategy: str = "casewise",
) -> tuple[pd.DataFrame, list[str]]:
    selected = [normalize_column_name(column) for column in dict.fromkeys(columns)]
    column_lookup: dict[str, str] = {}
    for column in frame.columns:
        column_lookup.setdefault(normalize_column_name(column), column)

    missing_columns = [column for column in selected if column not in column_lookup]
    if missing_columns:
        raise ValueError("Biến quan sát không có trong dữ liệu: " + ", ".join(missing_columns))

    source_columns = [column_lookup[column] for column in selected]
    selected_frame = frame[source_columns].copy()
    selected_frame.columns = selected
    numeric, warnings = coerce_numeric_frame(selected_frame)
    strategy = missing_strategy.lower()

    if strategy == "mean":
        numeric = numeric.fillna(numeric.mean(numeric_only=True))
        numeric = numeric.dropna(axis=0, how="any")
    elif strategy == "pairwise":
        warnings.append("Pairwise deletion không dùng cho điểm PLS lặp; đã áp dụng casewise deletion.")
        numeric = numeric.dropna(axis=0, how="any")
    else:
        numeric = numeric.dropna(axis=0, how="any")

    if numeric.empty:
        raise ValueError("Không còn dòng số liệu hoàn chỉnh sau khi xử lý giá trị thiếu.")

    zero_variance = [column for column in numeric.columns if numeric[column].std(ddof=1) == 0]
    if zero_variance:
        raise ValueError("Biến quan sát có phương sai bằng 0: " + ", ".join(zero_variance))

    return numeric, warnings


def standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    std = frame.std(ddof=1).replace(0, np.nan)
    standardized = (frame - frame.mean()) / std
    return standardized.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")


def export_cleaned_data(frame: pd.DataFrame, path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".xlsx":
        frame.to_excel(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _safe_stat(value: object) -> float | str:
    try:
        if pd.isna(value):
            return ""
        return float(value)
    except (TypeError, ValueError):
        return ""


def _scale_hint(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return ""
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if minimum >= 1 and maximum <= 5:
        return "1-5"
    if minimum >= 1 and maximum <= 7:
        return "1-7"
    if minimum >= 0 and maximum <= 10:
        return "0-10"
    return f"{minimum:g}-{maximum:g}"
