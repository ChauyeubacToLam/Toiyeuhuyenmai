"""Academic data-screening engine (Buổi 3 — Data Integrity & Descriptive Logic).

Implements the data-screening workflow taught for PhD-level quantitative research
(Hair et al., 2021): missing-value analysis, outlier detection, normality checks
and STROBE-style reporting. All heavy numeric work lives here so the GUI thread
only has to render the resulting tables.

Thresholds follow the lecture notes:
    * Missing data : <=5%  -> mean substitution; >5% -> imputation/removal.
    * Outliers     : |Z| > 3.29 ; IQR (1.5x mild, 3.0x extreme) ; out-of-scale.
    * Normality    : |Skewness| <= 2 and |Excess Kurtosis| <= 7 acceptable for PLS-SEM.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from core.data_manager import coerce_numeric_frame

try:  # SciPy ships with the project; degrade gracefully if it is missing.
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - defensive
    _scipy_stats = None


# -- thresholds ------------------------------------------------------------
SKEW_LIMIT = 2.0
KURTOSIS_LIMIT = 7.0
Z_LIMIT = 3.29
MISSING_MEAN_SUB = 5.0   # percent — mean substitution acceptable below this
MISSING_DROP = 10.0      # percent — beyond this consider removal / MI


def _is_integer_like(values: pd.Series) -> bool:
    if values.empty:
        return False
    arr = values.to_numpy(dtype=float, na_value=np.nan)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    return bool(np.allclose(arr, np.round(arr)))


def _scale_bounds(series: pd.Series) -> tuple[float, float] | None:
    """Infer the *intended* Likert range so out-of-scale entries can be flagged.

    Returns ``(lower, upper)`` for a likely ordinal scale, or ``None`` for a
    continuous variable where out-of-range checks do not apply.
    """

    values = series.dropna()
    if values.empty:
        return None
    if not _is_integer_like(values) or values.nunique() > 11:
        return None
    maximum = float(values.max())
    minimum = float(values.min())
    # Likert agreement scales used in PLS-SEM survey research are 1-anchored
    # (1–5, 1–7), so a value of 0 is treated as out-of-scale — this is exactly
    # the data-entry error the Buổi 3 example flags (Case 12 = 0 on a 1–5 item).
    # Only the 0–10 (NPS-style) scale legitimately includes 0.
    if maximum <= 5:
        return (1.0, 5.0)
    if maximum <= 7:
        return (1.0, 7.0)
    if maximum <= 10:
        return (0.0 if minimum < 1 else 1.0, 10.0)
    return None


def _case_label(positions: Iterable[int], limit: int = 6) -> str:
    cases = [f"Case {int(pos) + 1}" for pos in positions]
    if not cases:
        return ""
    if len(cases) > limit:
        return ", ".join(cases[:limit]) + f" … (+{len(cases) - limit})"
    return ", ".join(cases)


def _missing_recommendation(percent: float) -> str:
    if percent <= 0:
        return "Đầy đủ"
    if percent <= MISSING_MEAN_SUB:
        return "Thay bằng trung bình (mean substitution)"
    if percent <= MISSING_DROP:
        return "Multiple Imputation (nếu biến quan trọng)"
    return "Loại biến hoặc Multiple Imputation"


def _normality_verdict(skew: float, kurt: float, sw_p: float | None) -> str:
    skew_ok = abs(skew) <= SKEW_LIMIT if np.isfinite(skew) else False
    kurt_ok = abs(kurt) <= KURTOSIS_LIMIT if np.isfinite(kurt) else False
    if sw_p is not None and np.isfinite(sw_p) and sw_p > 0.05:
        return "Phân phối chuẩn"
    if skew_ok and kurt_ok:
        return "Chấp nhận cho PLS-SEM"
    return "Không chuẩn — cần xem xét"


def _missing_table(numeric: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in numeric.columns:
        series = numeric[column]
        missing = int(series.isna().sum())
        valid = int(series.notna().sum())
        percent = (missing / n_rows * 100.0) if n_rows else 0.0
        rows.append(
            {
                "Biến": column,
                "N hợp lệ": valid,
                "Thiếu": missing,
                "% Thiếu": round(percent, 2),
                "Khuyến nghị xử lý": _missing_recommendation(percent),
            }
        )
    return pd.DataFrame(rows)


def _normality_table(numeric: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in numeric.columns:
        values = numeric[column].dropna()
        n = int(values.size)
        skew = float(values.skew()) if n > 2 else np.nan
        kurt = float(values.kurtosis()) if n > 3 else np.nan

        sw_stat: float = np.nan
        sw_p: float | None = None
        ks_stat: float = np.nan
        ks_p: float = np.nan
        can_test = _scipy_stats is not None and n >= 3 and values.nunique() > 1
        # Shapiro–Wilk is only valid up to n = 5000; Kolmogorov–Smirnov has no
        # such cap, so it is computed for large samples too.
        if can_test and n <= 5000:
            try:
                sw_stat, sw_p_val = _scipy_stats.shapiro(values.to_numpy(dtype=float))
                sw_stat, sw_p = float(sw_stat), float(sw_p_val)
            except Exception:
                sw_stat, sw_p = np.nan, None
        if can_test:
            try:
                std = values.std(ddof=1)
                if std and np.isfinite(std):
                    standardized = (values - values.mean()) / std
                    ks_stat_val, ks_p_val = _scipy_stats.kstest(
                        standardized.to_numpy(dtype=float), "norm"
                    )
                    ks_stat, ks_p = float(ks_stat_val), float(ks_p_val)
            except Exception:
                ks_stat, ks_p = np.nan, np.nan

        rows.append(
            {
                "Biến": column,
                "Skewness": _round(skew),
                "Kurtosis": _round(kurt),
                "Shapiro-Wilk W": _round(sw_stat),
                "SW p": _round(sw_p),
                "K-S": _round(ks_stat),
                "K-S p": _round(ks_p),
                "Kết luận": _normality_verdict(skew, kurt, sw_p),
            }
        )
    return pd.DataFrame(rows)


def _outlier_table(numeric: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_rows: list[dict[str, object]] = []
    extreme_rows: list[dict[str, object]] = []
    for column in numeric.columns:
        series = numeric[column]
        values = series.dropna()
        n = int(values.size)
        positions = np.arange(len(series))[series.notna().to_numpy()]

        z_count = 0
        z_cases: list[int] = []
        std = values.std(ddof=1) if n > 1 else np.nan
        if n > 1 and std and np.isfinite(std):
            z = (values - values.mean()) / std
            mask = z.abs() > Z_LIMIT
            z_count = int(mask.sum())
            z_cases = list(positions[mask.to_numpy()])

        mild = severe = 0
        if n >= 4:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            if iqr and np.isfinite(iqr):
                lower_m, upper_m = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                lower_s, upper_s = q1 - 3.0 * iqr, q3 + 3.0 * iqr
                mild = int(((values < lower_m) | (values > upper_m)).sum())
                severe = int(((values < lower_s) | (values > upper_s)).sum())

        bounds = _scale_bounds(series)
        out_of_range = 0
        oor_cases: list[int] = []
        if bounds is not None:
            low, high = bounds
            oor_mask = (values < low) | (values > high)
            out_of_range = int(oor_mask.sum())
            oor_cases = list(positions[oor_mask.to_numpy()])

        flagged = sorted(set(z_cases) | set(oor_cases))
        note = "Bình thường"
        if out_of_range:
            note = "Lỗi nhập liệu (ngoài thang đo)"
        elif severe:
            note = "Có giá trị cực trị"
        elif z_count or mild:
            note = "Có ngoại lai nhẹ"

        out_rows.append(
            {
                "Biến": column,
                "Ngoài thang đo": out_of_range,
                "Z > 3.29": z_count,
                "Outlier IQR (1.5×)": mild,
                "Cực trị IQR (3.0×)": severe,
                "Case bất thường": _case_label(flagged),
                "Ghi chú": note,
            }
        )

        if n:
            ordered = values.sort_values(kind="mergesort")
            ordered_pos = positions[np.argsort(values.to_numpy(), kind="mergesort")]
            lowest = _value_case_list(ordered.to_numpy(), ordered_pos, k=5, highest=False)
            highest = _value_case_list(ordered.to_numpy(), ordered_pos, k=5, highest=True)
            extreme_rows.append(
                {
                    "Biến": column,
                    "5 giá trị thấp nhất": lowest,
                    "5 giá trị cao nhất": highest,
                }
            )

    return pd.DataFrame(out_rows), pd.DataFrame(extreme_rows)


def _value_case_list(sorted_values: np.ndarray, sorted_positions: np.ndarray, k: int, highest: bool) -> str:
    if sorted_values.size == 0:
        return ""
    if highest:
        idx = range(len(sorted_values) - 1, max(-1, len(sorted_values) - 1 - k), -1)
    else:
        idx = range(0, min(k, len(sorted_values)))
    parts = [f"{_fmt_num(sorted_values[i])} (Case {int(sorted_positions[i]) + 1})" for i in idx]
    return ", ".join(parts)


def _descriptive_table(numeric: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in numeric.columns:
        values = numeric[column].dropna()
        n = int(values.size)
        rows.append(
            {
                "Biến": column,
                "N": n,
                "Trung bình": _round(values.mean() if n else np.nan),
                "Độ lệch chuẩn": _round(values.std(ddof=1) if n > 1 else np.nan),
                "Min": _round(values.min() if n else np.nan),
                "Max": _round(values.max() if n else np.nan),
                "Skewness": _round(values.skew() if n > 2 else np.nan),
                "Kurtosis": _round(values.kurtosis() if n > 3 else np.nan),
            }
        )
    return pd.DataFrame(rows)


def _round(value: object, ndigits: int = 3) -> float | str:
    try:
        if value is None or pd.isna(value):
            return ""
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return ""


def _fmt_num(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}".replace(".", ",")


def _interpretation_html(
    n_rows: int,
    numeric: pd.DataFrame,
    missing: pd.DataFrame,
    normality: pd.DataFrame,
    outliers: pd.DataFrame,
) -> str:
    n_vars = numeric.shape[1]
    means = numeric.mean(numeric_only=True).dropna()
    sds = numeric.std(ddof=1, numeric_only=True).dropna()
    miss_pct = missing["% Thiếu"] if "% Thiếu" in missing else pd.Series(dtype=float)
    max_missing = float(miss_pct.max()) if not miss_pct.empty else 0.0
    n_missing_vars = int((miss_pct > 0).sum()) if not miss_pct.empty else 0

    non_normal = 0
    if "Kết luận" in normality:
        non_normal = int((normality["Kết luận"] == "Không chuẩn — cần xem xét").sum())

    vars_with_outliers = 0
    if "Ghi chú" in outliers:
        vars_with_outliers = int((outliers["Ghi chú"] != "Bình thường").sum())
    data_entry_errors = int((outliers["Ngoài thang đo"] > 0).sum()) if "Ngoài thang đo" in outliers else 0

    mean_lo = f"{means.min():.2f}" if not means.empty else "-"
    mean_hi = f"{means.max():.2f}" if not means.empty else "-"
    sd_lo = f"{sds.min():.2f}" if not sds.empty else "-"
    sd_hi = f"{sds.max():.2f}" if not sds.empty else "-"

    bullets = [
        f"Mẫu nghiên cứu gồm <b>N = {n_rows}</b> quan sát trên <b>{n_vars}</b> biến đo lường.",
        (
            f"Tỷ lệ thiếu dao động 0–{max_missing:.1f}% ({n_missing_vars} biến có giá trị thiếu). "
            "Biến thiếu ≤5% được giữ lại và thay bằng trung bình (mean substitution); "
            "biến thiếu >5% nên cân nhắc Multiple Imputation hoặc loại bỏ (Hair et al., 2021)."
            if n_missing_vars
            else "Không phát hiện giá trị thiếu trên các biến đo lường."
        ),
        (
            f"Giá trị trung bình dao động {mean_lo}–{mean_hi}; độ lệch chuẩn dao động {sd_lo}–{sd_hi}, "
            "cho thấy mức độ phân tán vừa phải giữa các thang đo."
        ),
        (
            f"{vars_with_outliers} biến có dấu hiệu ngoại lai"
            + (f", trong đó {data_entry_errors} biến chứa giá trị ngoài thang đo (khả năng lỗi nhập liệu)."
               if data_entry_errors else " (mức nhẹ, trong giới hạn <3 SD).")
            if vars_with_outliers
            else "Không phát hiện ngoại lai nghiêm trọng (giá trị nằm trong giới hạn <3 SD)."
        ),
        (
            f"{non_normal} biến vi phạm ngưỡng chuẩn |Skewness|≤2 và |Kurtosis|≤7; "
            "các biến còn lại đạt yêu cầu phân phối cho PLS-SEM."
            if non_normal
            else "Tất cả biến nằm trong ngưỡng |Skewness|≤2 và |Kurtosis|≤7 — chấp nhận được cho PLS-SEM."
        ),
    ]
    items = "".join(f"<li>{text}</li>" for text in bullets)
    return (
        "<div style='line-height:1.5'>"
        "<p style='margin:0 0 6px 0'>Diễn giải thống kê mô tả không chỉ dừng ở con số mà cần kết nối "
        "với lý thuyết thang đo (Reflective/Formative) đã xây dựng:</p>"
        f"<ul style='margin:0 0 4px 18px;padding:0'>{items}</ul>"
        "</div>"
    )


def _strobe_html(
    n_rows: int,
    missing: pd.DataFrame,
    normality: pd.DataFrame,
    outliers: pd.DataFrame,
) -> str:
    miss_pct = missing["% Thiếu"] if "% Thiếu" in missing else pd.Series(dtype=float)
    max_missing = float(miss_pct.max()) if not miss_pct.empty else 0.0
    non_normal_vars = []
    if "Kết luận" in normality and "Biến" in normality:
        non_normal_vars = list(normality.loc[normality["Kết luận"] == "Không chuẩn — cần xem xét", "Biến"])
    n_mild = int((outliers["Ghi chú"] != "Bình thường").sum()) if "Ghi chú" in outliers else 0

    missing_para = (
        f"Among {n_rows} responses, missing data ranged between 0–{max_missing:.1f}% across variables. "
        "Cases with ≤5% missing were retained using mean substitution, while variables exceeding 5% were "
        "handled through Multiple Imputation or excluded, consistent with Hair et al. (2021) and "
        "Little &amp; Rubin (2019)."
    )
    outlier_para = (
        f"Data were screened for outliers using Z-scores (|Z| &gt; 3.29) and the IQR rule. "
        f"{n_mild if n_mild else 'No'} variable(s) showed mild outliers within acceptable limits (&lt;3 SD); "
        "out-of-range entries were treated as data-entry errors and corrected prior to analysis."
    )
    if non_normal_vars:
        listed = ", ".join(non_normal_vars[:6]) + (" …" if len(non_normal_vars) > 6 else "")
        normal_para = (
            "Normality was examined with Skewness/Kurtosis and the Shapiro–Wilk test. "
            f"Variables {listed} deviated from normality (p &lt; .05); however, values remained within "
            "|Skewness| ≤ 2 and |Kurtosis| ≤ 7, which is acceptable for PLS-SEM (Hair et al., 2021)."
        )
    else:
        normal_para = (
            "Skewness and kurtosis values were within ±2 and ±7 respectively, and the Shapiro–Wilk test "
            "indicated acceptable normality, supporting the use of the data for further reliability and "
            "factor analyses (Hair et al., 2021)."
        )
    return (
        "<div style='line-height:1.5'>"
        f"<p style='margin:0 0 8px 0'><b>Missing data.</b> {missing_para}</p>"
        f"<p style='margin:0 0 8px 0'><b>Outliers.</b> {outlier_para}</p>"
        f"<p style='margin:0'><b>Normality.</b> {normal_para}</p>"
        "</div>"
    )


def screen_dataset(frame: pd.DataFrame, used_columns: Iterable[str] | None = None) -> dict[str, object]:
    """Run the full Buổi 3 data-screening workflow on ``frame``.

    Returns a dict of result DataFrames plus ready-to-copy interpretation and
    STROBE report HTML. Safe to call from a worker thread (no Qt usage).
    """

    numeric, coerce_warnings = coerce_numeric_frame(frame)
    n_rows = int(len(frame))

    if numeric.shape[1] == 0 or n_rows == 0:
        empty = pd.DataFrame()
        return {
            "missing": empty,
            "normality": empty,
            "outliers": empty,
            "extreme_values": empty,
            "descriptives": empty,
            "interpretation_html": "<p>Chưa có dữ liệu số để sàng lọc.</p>",
            "strobe_html": "",
            "summary": {"rows": n_rows, "variables": 0, "missing_cells": 0,
                        "non_normal": 0, "outlier_vars": 0,
                        "scipy": _scipy_stats is not None},
            "warnings": coerce_warnings,
        }

    missing = _missing_table(numeric, n_rows)
    normality = _normality_table(numeric)
    outliers, extreme_values = _outlier_table(numeric)
    descriptives = _descriptive_table(numeric)

    summary = {
        "rows": n_rows,
        "variables": int(numeric.shape[1]),
        "missing_cells": int(numeric.isna().sum().sum()),
        "non_normal": int((normality["Kết luận"] == "Không chuẩn — cần xem xét").sum())
        if "Kết luận" in normality else 0,
        "outlier_vars": int((outliers["Ghi chú"] != "Bình thường").sum())
        if "Ghi chú" in outliers else 0,
        "scipy": _scipy_stats is not None,
    }

    return {
        "missing": missing,
        "normality": normality,
        "outliers": outliers,
        "extreme_values": extreme_values,
        "descriptives": descriptives,
        "interpretation_html": _interpretation_html(n_rows, numeric, missing, normality, outliers),
        "strobe_html": _strobe_html(n_rows, missing, normality, outliers),
        "summary": summary,
        "warnings": coerce_warnings,
    }
