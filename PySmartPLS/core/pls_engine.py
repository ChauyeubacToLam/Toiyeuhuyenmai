from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from core.data_manager import normalize_column_name, prepare_analysis_frame, read_dataset, standardize_frame


class ModelValidationError(ValueError):
    pass


@dataclass
class FitContext:
    lvs: list[str]
    indicators: dict[str, list[str]]
    modes: dict[str, str]
    z: pd.DataFrame
    warnings: list[str]
    original: pd.DataFrame


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


# Per-worker process state. The heavy, identical-across-tasks data (resampling base,
# model, metric keys) is shipped ONCE per worker via the pool initializer; each task
# then carries only a (count, seed) tuple, so we can use many small chunks for smooth
# progress without re-pickling the data set every time.
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(common: dict[str, Any]) -> None:
    """Pin the worker to a single BLAS thread and cache its bootstrap context.

    Without the thread pin, every worker spawns its own BLAS thread pool and they
    oversubscribe the CPU, collapsing the process-pool speed-up to ~2x. Limiting
    threads per process restores near-linear scaling.
    """
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:
        pass
    engine = PLSEngine(common["sample_base"])
    engine.set_model(
        common["measurement_model"],
        common["structural_model"],
        common["measurement_modes"],
        common["effects"],
    )
    _WORKER_STATE.clear()
    _WORKER_STATE["engine"] = engine
    _WORKER_STATE["sample_base"] = common["sample_base"]
    _WORKER_STATE["keys"] = common["keys"]
    _WORKER_STATE["settings"] = common["settings"]


def _bootstrap_chunk_worker(task: tuple[int, int]) -> tuple[dict[str, list[float]], int]:
    count, seed = task
    engine: PLSEngine = _WORKER_STATE["engine"]
    sample_base: pd.DataFrame = _WORKER_STATE["sample_base"]
    keys: list[str] = _WORKER_STATE["keys"]
    settings: dict[str, Any] = _WORKER_STATE["settings"]
    rows = len(sample_base)
    rng = np.random.default_rng(int(seed))
    values: dict[str, list[float]] = {key: [] for key in keys}
    completed = 0
    for _ in range(int(count)):
        sample_index = rng.integers(0, rows, size=rows)
        sample = sample_base.iloc[sample_index].reset_index(drop=True)
        try:
            flat = engine._fit_bootstrap_flat_numeric(sample, settings)
        except Exception:
            completed += 1
            continue
        for key in keys:
            value = flat.get(key, np.nan)
            if pd.notna(value):
                values[key].append(float(value))
        completed += 1
    return values, completed


class PLSEngine:
    def __init__(self, data: pd.DataFrame | str | None = None):
        self.raw_data: pd.DataFrame | None = None
        self.data_path: str = ""
        self.measurement_model: dict[str, list[str]] = {}
        self.structural_model: list[tuple[str, str]] = []
        self.measurement_modes: dict[str, str] = {}
        self.effects: list[dict[str, str]] = []

        if isinstance(data, pd.DataFrame):
            self.raw_data = data.copy()
        elif isinstance(data, str):
            self.load_data(data)

    def load_data(self, data_path: str) -> None:
        loaded = read_dataset(data_path)
        self.raw_data = loaded.frame
        self.data_path = loaded.path

    def set_model(
        self,
        measurement_model: dict[str, list[str]],
        structural_model: list[tuple[str, str]],
        measurement_modes: dict[str, str] | None = None,
        effects: list[dict[str, str]] | None = None,
    ) -> None:
        self.measurement_model = {
            str(construct).strip(): list(dict.fromkeys(normalize_column_name(indicator) for indicator in indicators))
            for construct, indicators in measurement_model.items()
            if str(construct).strip()
        }
        self.structural_model = [(str(source).strip(), str(target).strip()) for source, target in structural_model]
        self.measurement_modes = {
            construct: (measurement_modes or {}).get(construct, "reflective")
            for construct in self.measurement_model
        }
        self.effects = [dict(effect) for effect in (effects or [])]

    def calculate(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        settings = settings or {}
        if self.raw_data is None:
            raise ValueError("Chưa có dữ liệu.")

        base = self._fit(self.raw_data, settings)
        if settings.get("bootstrap_enabled"):
            subsamples = int(settings.get("bootstrap_subsamples", 500))
            if subsamples > 0:
                summary, samples = self._bootstrap(self.raw_data, settings, base, subsamples)
                base["bootstrap"] = summary
                base["bootstrap_samples"] = samples
                base["bootstrap_confidence"] = float(settings.get("confidence_level", 0.95))
                base["bootstrap_subsamples"] = subsamples
                base["bootstrap_test_type"] = settings.get("test_type", "two-tailed")
                base["bootstrap_meta"] = {
                    "confidence": float(settings.get("confidence_level", 0.95)),
                    "subsamples": subsamples,
                    "valid": int(max((values.size for values in samples.values()), default=0)),
                    "test_type": settings.get("test_type", "two-tailed"),
                    "ci_method": settings.get("ci_method", "percentile"),
                    "seed": int(settings.get("random_seed", 12345)),
                }
                base["interpretation"] = self._make_interpretation(base)
        return base

    def calculate_ipma(self, target: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        """Importance-Performance Map Analysis (construct level).

        Importance = unstandardized total effect on the target; Performance = mean of
        the latent index rescaled to 0-100 using each indicator's observed range.
        """
        settings = settings or {}
        if self.raw_data is None:
            raise ValueError("Chưa có dữ liệu.")
        base = self._fit(self.raw_data, settings)
        measurement: dict[str, list[str]] = base["measurement_model"]
        if target not in measurement:
            raise ModelValidationError(f"Biến mục tiêu IPMA '{target}' không hợp lệ.")

        weights: pd.DataFrame = base["outer_weights"]
        original: pd.DataFrame = base["indicator_data_original"]
        index = pd.DataFrame(index=original.index)
        for lv, indicators in measurement.items():
            indicators = [name for name in indicators if name in original.columns]
            if not indicators:
                continue
            w = np.array(
                [float(weights.loc[name, lv]) if name in weights.index else 0.0 for name in indicators],
                dtype=float,
            )
            if not np.any(w):
                w = np.ones(len(indicators), dtype=float)
            block = original[indicators].astype(float)
            minimum = block.min()
            maximum = block.max()
            span = (maximum - minimum).replace(0, np.nan)
            rescaled = ((block - minimum) / span * 100.0).fillna(50.0)
            index[lv] = (rescaled.to_numpy(dtype=float) @ w) / (w.sum() if w.sum() else 1.0)

        performance = index.mean()
        sd = index.std(ddof=1)
        total: pd.DataFrame = base["total_effects"]
        rows: list[dict[str, Any]] = []
        for predictor in measurement:
            if predictor == target or predictor not in index.columns:
                continue
            if predictor in total.index and target in total.columns:
                beta_total = float(total.loc[predictor, target])
            else:
                beta_total = 0.0
            if abs(beta_total) < 1e-9:
                continue
            sd_pred = float(sd.get(predictor, np.nan))
            sd_target = float(sd.get(target, np.nan))
            importance = beta_total * (sd_target / sd_pred) if sd_pred and np.isfinite(sd_pred) else beta_total
            rows.append({
                "Construct": predictor,
                "Importance (Total Effect)": importance,
                "Performance (Index 0-100)": float(performance.get(predictor, np.nan)),
            })
        ipma_table = pd.DataFrame(rows).set_index("Construct") if rows else pd.DataFrame(
            columns=["Importance (Total Effect)", "Performance (Index 0-100)"]
        )
        performance_table = pd.DataFrame({"Performance (Index 0-100)": performance})
        performance_table.index.name = "Construct"
        return {
            "algorithm": "IPMA",
            "target": target,
            "ipma": ipma_table,
            "ipma_performance": performance_table,
            "mean_importance": float(ipma_table["Importance (Total Effect)"].mean()) if not ipma_table.empty else 0.0,
            "mean_performance": float(performance.get(target, performance.mean())),
            "diagnostics": base.get("diagnostics", []),
        }

    @staticmethod
    def _kfold(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
        indices = rng.permutation(n)
        return [fold for fold in np.array_split(indices, k) if len(fold)]

    def _topological_order(self, lvs: list[str], structural: list[tuple[str, str]]) -> list[str]:
        incoming = {lv: 0 for lv in lvs}
        adjacency: dict[str, list[str]] = {lv: [] for lv in lvs}
        for source, target in structural:
            if source in incoming and target in incoming:
                adjacency[source].append(target)
                incoming[target] += 1
        queue = [lv for lv in lvs if incoming[lv] == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nxt in adjacency[node]:
                incoming[nxt] -= 1
                if incoming[nxt] == 0:
                    queue.append(nxt)
        # Fallback for any nodes left out by a cycle: keep model order.
        order.extend(lv for lv in lvs if lv not in order)
        return order

    def calculate_predict(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        """PLSpredict: k-fold out-of-sample prediction (Q²predict, RMSE, MAE) vs LM benchmark."""
        settings = settings or {}
        if self.raw_data is None:
            raise ValueError("Chưa có dữ liệu.")
        errors, _warnings = self.validate_model(self.raw_data)
        if errors:
            raise ModelValidationError("\n".join(errors))

        context = self._make_context(self.raw_data, settings)
        lvs, indicators, modes = context.lvs, context.indicators, context.modes
        structural = self.structural_model
        endogenous = [lv for lv in lvs if any(t == lv for _s, t in structural)]
        exogenous = [lv for lv in lvs if lv not in endogenous]
        if not endogenous:
            raise ModelValidationError("Mô hình không có biến nội sinh để dự báo.")
        all_inds = [i for lv in lvs for i in indicators[lv]]
        exo_inds = [i for lv in exogenous for i in indicators[lv]]
        endo_pairs = [(lv, ind) for lv in endogenous for ind in indicators[lv]]
        order = self._topological_order(lvs, structural)

        data = context.original.reset_index(drop=True)
        n = len(data)
        k = max(2, int(settings.get("folds", 10)))
        reps = max(1, int(settings.get("repetitions", 10)))
        rng = np.random.default_rng(int(settings.get("random_seed", 12345)))

        # Per-case accumulators (averaged across repetitions) so the holdout
        # predictions and errors can be described exactly like SmartPLS.
        pred_sum = {ind: np.zeros(n) for _lv, ind in endo_pairs}
        pred_cnt = {ind: np.zeros(n) for _lv, ind in endo_pairs}
        pred_sum_lm = {ind: np.zeros(n) for _lv, ind in endo_pairs}
        lv_pred_sum = {lv: np.zeros(n) for lv in endogenous}
        lv_actual_sum = {lv: np.zeros(n) for lv in endogenous}
        lv_cnt = {lv: np.zeros(n) for lv in endogenous}

        for _rep in range(reps):
            for test_idx in self._kfold(n, k, rng):
                train_idx = np.setdiff1d(np.arange(n), test_idx)
                if len(train_idx) < max(10, len(lvs) + 2) or len(test_idx) == 0:
                    continue
                train, test = data.iloc[train_idx], data.iloc[test_idx]
                mu = train[all_inds].mean()
                sd = train[all_inds].std(ddof=1)
                sd_safe = sd.replace(0, np.nan)
                ztr = ((train[all_inds] - mu) / sd_safe).fillna(0.0)
                zte = ((test[all_inds] - mu) / sd_safe).fillna(0.0)
                weights, scores_tr, _conv, _it = self._run_pls(ztr, indicators, modes, lvs, settings)
                paths, _r2, _adj = self._estimate_paths(scores_tr, lvs, structural)

                raw_mean, raw_std, fscore = {}, {}, {}
                for lv in lvs:
                    raw_tr = ztr[indicators[lv]].to_numpy(dtype=float) @ weights[lv]
                    raw_mean[lv] = float(np.mean(raw_tr))
                    std = float(np.std(raw_tr, ddof=1))
                    raw_std[lv] = std if std else 1.0
                    raw_te = zte[indicators[lv]].to_numpy(dtype=float) @ weights[lv]
                    fscore[lv] = (raw_te - raw_mean[lv]) / raw_std[lv]

                fhat: dict[str, np.ndarray] = {}
                for lv in order:
                    preds = [s for s, t in structural if t == lv]
                    if not preds:
                        fhat[lv] = fscore[lv]
                    else:
                        acc = np.zeros(len(test_idx))
                        for s in preds:
                            beta = float(paths.loc[s, lv]) if (s in paths.index and lv in paths.columns) else 0.0
                            acc = acc + beta * fhat.get(s, fscore[s])
                        fhat[lv] = acc

                # Construct (LV) level: structural prediction vs measurement score.
                for lv in endogenous:
                    lv_pred_sum[lv][test_idx] += fhat[lv]
                    lv_actual_sum[lv][test_idx] += fscore[lv]
                    lv_cnt[lv][test_idx] += 1.0

                # Indicator (MV) level: reconstruct each endogenous item.
                for lv in endogenous:
                    for ind in indicators[lv]:
                        loading = _corr(ztr[ind], scores_tr[lv])
                        sd_ind = float(sd.get(ind)) if pd.notna(sd.get(ind)) and sd.get(ind) else 1.0
                        x_pred = loading * fhat[lv] * sd_ind + float(mu[ind])
                        pred_sum[ind][test_idx] += x_pred
                        pred_cnt[ind][test_idx] += 1.0

                if exo_inds:
                    x_train = np.column_stack([np.ones(len(train)), train[exo_inds].to_numpy(dtype=float)])
                    x_test = np.column_stack([np.ones(len(test)), test[exo_inds].to_numpy(dtype=float)])
                    for _lv, ind in endo_pairs:
                        beta_lm, *_ = np.linalg.lstsq(x_train, train[ind].to_numpy(dtype=float), rcond=None)
                        pred_sum_lm[ind][test_idx] += x_test @ beta_lm

        def _avg(total: np.ndarray, cnt: np.ndarray) -> np.ndarray:
            out = np.full(n, np.nan)
            mask = cnt > 0
            out[mask] = total[mask] / cnt[mask]
            return out

        # ---- Indicator (MV) level ---------------------------------------
        mv_pls_rows, mv_lm_rows, mv_cmp_rows = [], [], []
        mv_err_cols: dict[str, np.ndarray] = {}
        mv_val_cols: dict[str, np.ndarray] = {}
        for lv, ind in endo_pairs:
            cnt = pred_cnt[ind]
            actual_full = data[ind].to_numpy(dtype=float)
            avg_pred = _avg(pred_sum[ind], cnt)
            mask = (cnt > 0) & np.isfinite(avg_pred) & np.isfinite(actual_full)
            actual, pred = actual_full[mask], avg_pred[mask]
            pls = _pred_metrics(actual, pred)
            mv_pls_rows.append({"Indicator": ind, "Construct": lv,
                                "RMSE": pls["rmse"], "MAE": pls["mae"],
                                "MAPE": pls["mape"], "Q²predict": pls["q2"]})
            mv_err_cols[ind] = actual - pred
            mv_val_cols[ind] = pred
            if exo_inds:
                avg_lm = _avg(pred_sum_lm[ind], cnt)
                m2 = mask & np.isfinite(avg_lm)
                lm = _pred_metrics(data[ind].to_numpy(dtype=float)[m2], avg_lm[m2])
                mv_lm_rows.append({"Indicator": ind, "Construct": lv,
                                   "RMSE": lm["rmse"], "MAE": lm["mae"],
                                   "MAPE": lm["mape"], "Q²predict": lm["q2"]})
                mv_cmp_rows.append({"Indicator": ind, "Construct": lv,
                                    "PLS-SEM_RMSE": pls["rmse"], "LM_RMSE": lm["rmse"],
                                    "RMSE (PLS-LM)": pls["rmse"] - lm["rmse"],
                                    "PLS-SEM_MAE": pls["mae"], "LM_MAE": lm["mae"],
                                    "MAE (PLS-LM)": pls["mae"] - lm["mae"]})

        mv_table = pd.DataFrame(mv_pls_rows).set_index("Indicator")
        mv_lm_table = pd.DataFrame(mv_lm_rows).set_index("Indicator") if mv_lm_rows else pd.DataFrame()
        mv_cmp_table = pd.DataFrame(mv_cmp_rows).set_index("Indicator") if mv_cmp_rows else pd.DataFrame()
        mv_err_desc = pd.DataFrame(
            {ind: _describe(arr) for ind, arr in mv_err_cols.items()}
        ).T if mv_err_cols else pd.DataFrame()
        mv_val_desc = pd.DataFrame(
            {ind: _describe(arr) for ind, arr in mv_val_cols.items()}
        ).T if mv_val_cols else pd.DataFrame()

        # ---- Construct (LV) level ---------------------------------------
        lv_rows = []
        lv_err_cols: dict[str, np.ndarray] = {}
        lv_val_cols: dict[str, np.ndarray] = {}
        for lv in endogenous:
            cnt = lv_cnt[lv]
            avg_pred = _avg(lv_pred_sum[lv], cnt)
            avg_actual = _avg(lv_actual_sum[lv], cnt)
            mask = (cnt > 0) & np.isfinite(avg_pred) & np.isfinite(avg_actual)
            actual, pred = avg_actual[mask], avg_pred[mask]
            m = _pred_metrics(actual, pred)
            lv_rows.append({"Construct": lv, "RMSE": m["rmse"], "MAE": m["mae"], "Q²predict": m["q2"]})
            lv_err_cols[lv] = actual - pred
            lv_val_cols[lv] = pred
        lv_table = pd.DataFrame(lv_rows).set_index("Construct")
        lv_err_desc = pd.DataFrame(
            {lv: _describe(arr) for lv, arr in lv_err_cols.items()}
        ).T if lv_err_cols else pd.DataFrame()
        lv_val_desc = pd.DataFrame(
            {lv: _describe(arr) for lv, arr in lv_val_cols.items()}
        ).T if lv_val_cols else pd.DataFrame()

        return {
            "algorithm": "PLSpredict",
            "mv_prediction": mv_table,
            "mv_lm": mv_lm_table,
            "mv_compare": mv_cmp_table,
            "mv_error_desc": mv_err_desc,
            "mv_pred_desc": mv_val_desc,
            "lv_prediction": lv_table,
            "lv_error_desc": lv_err_desc,
            "lv_pred_desc": lv_val_desc,
            "has_lm": bool(exo_inds),
            "folds": k,
            "repetitions": reps,
            "diagnostics": context.warnings,
        }

    def _split_groups(self, group_column: str, value_a: str, value_b: str):
        if self.raw_data is None:
            raise ValueError("Chưa có dữ liệu.")
        if group_column not in self.raw_data.columns:
            raise ModelValidationError(f"Không tìm thấy biến phân nhóm '{group_column}'.")
        as_text = self.raw_data[group_column].astype(str)
        data_a = self.raw_data[as_text == str(value_a)].reset_index(drop=True)
        data_b = self.raw_data[as_text == str(value_b)].reset_index(drop=True)
        if len(data_a) < 10 or len(data_b) < 10:
            raise ModelValidationError(
                f"Mỗi nhóm cần ít nhất 10 quan sát (A={len(data_a)}, B={len(data_b)})."
            )
        return data_a, data_b

    def _group_path_estimate(self, data: pd.DataFrame, settings: dict[str, Any], subsamples: int, seed: int):
        """Per-group path coefficients + their bootstrap distributions (for MGA tests)."""
        all_indicators = [ind for inds in self.measurement_model.values() for ind in inds]
        numeric, _warn = prepare_analysis_frame(data, all_indicators, settings.get("missing_strategy", "casewise"))
        paths_df = self._paths_from_numeric(numeric, settings)
        paths = {
            (s, t): float(paths_df.loc[s, t])
            for s, t in self.structural_model
            if s in paths_df.index and t in paths_df.columns
        }
        rng = np.random.default_rng(seed)
        boot: dict[tuple[str, str], list[float]] = {edge: [] for edge in self.structural_model}
        n = len(numeric)
        for _ in range(subsamples):
            sample = numeric.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
            try:
                pf = self._paths_from_numeric(sample, settings)
            except Exception:
                continue
            for s, t in self.structural_model:
                if s in pf.index and t in pf.columns:
                    boot[(s, t)].append(float(pf.loc[s, t]))
        boot_arrays = {edge: np.array(values, dtype=float) for edge, values in boot.items()}
        return {"paths": paths, "boot": boot_arrays, "n": n}

    def calculate_mga(self, group_column: str, value_a: str, value_b: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        """Multi-Group Analysis: parametric, Welch-Satterthwaite, and Henseler PLS-MGA tests."""
        settings = settings or {}
        errors, _warnings = self.validate_model(self.raw_data)
        if errors:
            raise ModelValidationError("\n".join(errors))
        if not self.structural_model:
            raise ModelValidationError("Cần mô hình cấu trúc để chạy MGA.")
        data_a, data_b = self._split_groups(group_column, value_a, value_b)
        subsamples = int(settings.get("bootstrap_subsamples", 300))
        seed = int(settings.get("random_seed", 12345))
        res_a = self._group_path_estimate(data_a, settings, subsamples, seed)
        res_b = self._group_path_estimate(data_b, settings, subsamples, seed + 1)
        n_a, n_b = res_a["n"], res_b["n"]

        rows: list[dict[str, Any]] = []
        for s, t in self.structural_model:
            beta_a = res_a["paths"].get((s, t), np.nan)
            beta_b = res_b["paths"].get((s, t), np.nan)
            arr_a = res_a["boot"].get((s, t), np.array([]))
            arr_b = res_b["boot"].get((s, t), np.array([]))
            se_a = float(np.std(arr_a, ddof=1)) if arr_a.size > 1 else np.nan
            se_b = float(np.std(arr_b, ddof=1)) if arr_b.size > 1 else np.nan
            diff = beta_a - beta_b
            p_par = p_welch = p_mga = np.nan
            if np.isfinite(se_a) and np.isfinite(se_b) and se_a > 0 and se_b > 0:
                pooled = np.sqrt(
                    ((n_a - 1) ** 2 / (n_a + n_b - 2)) * se_a ** 2
                    + ((n_b - 1) ** 2 / (n_a + n_b - 2)) * se_b ** 2
                ) * np.sqrt(1 / n_a + 1 / n_b)
                if pooled > 0:
                    t_par = diff / pooled
                    p_par = float(2 * stats.t.sf(abs(t_par), df=n_a + n_b - 2))
                denom_w = np.sqrt(se_a ** 2 + se_b ** 2)
                if denom_w > 0:
                    t_w = diff / denom_w
                    df_w = (se_a ** 2 + se_b ** 2) ** 2 / (
                        se_a ** 4 / (n_a - 1) + se_b ** 4 / (n_b - 1)
                    )
                    if df_w > 0:
                        p_welch = float(2 * stats.t.sf(abs(t_w), df=df_w))
            if arr_a.size and arr_b.size:
                p_mga = _henseler_pls_mga(arr_a, arr_b)
            rows.append({
                "Path": f"{s} -> {t}",
                f"β ({value_a})": beta_a,
                f"β ({value_b})": beta_b,
                "Diff (A-B)": diff,
                "p (Parametric)": p_par,
                "p (Welch)": p_welch,
                "p (PLS-MGA)": p_mga,
            })
        table = pd.DataFrame(rows).set_index("Path")
        return {
            "algorithm": "MGA",
            "mga": table,
            "group_column": group_column,
            "value_a": str(value_a),
            "value_b": str(value_b),
            "n_a": n_a,
            "n_b": n_b,
        }

    def calculate_permutation(self, group_column: str, value_a: str, value_b: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        """Permutation test of group path differences (reassigns group labels)."""
        settings = settings or {}
        errors, _warnings = self.validate_model(self.raw_data)
        if errors:
            raise ModelValidationError("\n".join(errors))
        data_a, data_b = self._split_groups(group_column, value_a, value_b)
        permutations = int(settings.get("permutations", settings.get("bootstrap_subsamples", 300)))
        seed = int(settings.get("random_seed", 12345))
        all_indicators = [ind for inds in self.measurement_model.values() for ind in inds]
        strategy = settings.get("missing_strategy", "casewise")
        numeric_a, _wa = prepare_analysis_frame(data_a, all_indicators, strategy)
        numeric_b, _wb = prepare_analysis_frame(data_b, all_indicators, strategy)

        fit_a = self._paths_from_numeric(numeric_a, settings)
        fit_b = self._paths_from_numeric(numeric_b, settings)
        observed = {}
        for s, t in self.structural_model:
            a = float(fit_a.loc[s, t]) if (s in fit_a.index and t in fit_a.columns) else np.nan
            b = float(fit_b.loc[s, t]) if (s in fit_b.index and t in fit_b.columns) else np.nan
            observed[(s, t)] = (a, b, a - b)

        pooled = pd.concat([numeric_a, numeric_b], ignore_index=True)
        n_a, total = len(numeric_a), len(pooled)
        rng = np.random.default_rng(seed)
        ge = {edge: 0 for edge in self.structural_model}
        valid = {edge: 0 for edge in self.structural_model}
        for _ in range(permutations):
            order = rng.permutation(total)
            perm_a = pooled.iloc[order[:n_a]].reset_index(drop=True)
            perm_b = pooled.iloc[order[n_a:]].reset_index(drop=True)
            try:
                pa = self._paths_from_numeric(perm_a, settings)
                pb = self._paths_from_numeric(perm_b, settings)
            except Exception:
                continue
            for s, t in self.structural_model:
                if s in pa.index and t in pa.columns and s in pb.index and t in pb.columns:
                    perm_diff = abs(float(pa.loc[s, t]) - float(pb.loc[s, t]))
                    valid[(s, t)] += 1
                    if perm_diff >= abs(observed[(s, t)][2]):
                        ge[(s, t)] += 1

        rows = []
        for s, t in self.structural_model:
            a, b, diff = observed[(s, t)]
            p = float(ge[(s, t)] / valid[(s, t)]) if valid[(s, t)] else np.nan
            rows.append({
                "Path": f"{s} -> {t}",
                f"β ({value_a})": a,
                f"β ({value_b})": b,
                "Diff (A-B)": diff,
                "p (Permutation)": p,
            })
        table = pd.DataFrame(rows).set_index("Path")
        return {
            "algorithm": "Permutation",
            "permutation": table,
            "group_column": group_column,
            "value_a": str(value_a),
            "value_b": str(value_b),
            "n_a": n_a,
            "n_b": total - n_a,
            "permutations": permutations,
        }

    def calculate_sum_scores(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        settings = settings or {}
        if self.raw_data is None:
            raise ValueError("Chưa có dữ liệu.")

        context = self._make_context(self.raw_data, settings)
        scores = pd.DataFrame(index=context.z.index)
        for lv, indicators in context.indicators.items():
            scores[lv] = context.z[indicators].mean(axis=1)
        scores = standardize_frame(scores)

        paths, r2, adjusted_r2 = self._estimate_paths(scores, context.lvs)
        return {
            "algorithm": "Sum scores / OLS",
            "scores": scores,
            "path_coefficients": paths,
            "r_square": pd.Series(r2, name="R2"),
            "adjusted_r_square": pd.Series(adjusted_r2, name="Adjusted R2"),
            "diagnostics": context.warnings,
        }

    def validate_model(self, frame: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
        frame = frame if frame is not None else self.raw_data
        errors: list[str] = []
        warnings: list[str] = []

        if not self.measurement_model:
            errors.append("Hãy tạo ít nhất một biến tiềm ẩn và gán biến quan sát trước khi chạy.")

        known_constructs = set(self.measurement_model)
        for construct, indicators in self.measurement_model.items():
            if not indicators:
                errors.append(f"Biến tiềm ẩn '{construct}' chưa có biến quan sát.")
            if self.measurement_modes.get(construct, "reflective") == "formative" and len(indicators) < 2:
                warnings.append(f"Biến hình thành '{construct}' có ít hơn hai biến quan sát.")

        assigned: dict[str, str] = {}
        for construct, indicators in self.measurement_model.items():
            for indicator in indicators:
                if indicator in assigned:
                    warnings.append(
                        f"Biến quan sát '{indicator}' đang được gán cho cả '{assigned[indicator]}' và '{construct}'."
                    )
                assigned[indicator] = construct

        for source, target in self.structural_model:
            if source not in known_constructs or target not in known_constructs:
                errors.append(f"Đường dẫn '{source} -> {target}' tham chiếu biến tiềm ẩn không tồn tại.")
            if source == target:
                errors.append(f"Không hợp lệ khi biến tự trỏ vào chính nó: '{source} -> {target}'.")

        if self._has_cycle():
            errors.append("Mô hình cấu trúc có vòng lặp có hướng. App hiện chưa hỗ trợ mô hình non-recursive.")

        if frame is not None:
            frame_columns = {normalize_column_name(column) for column in frame.columns}
            missing = sorted({indicator for indicator in assigned if normalize_column_name(indicator) not in frame_columns})
            if missing:
                errors.append("Biến quan sát không có trong dữ liệu: " + ", ".join(missing))

        return errors, warnings

    def _run_pls(
        self,
        z: pd.DataFrame,
        indicators: dict[str, list[str]],
        modes: dict[str, str],
        lvs: list[str],
        settings: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], pd.DataFrame, bool, int]:
        """Core PLS weight-estimation loop. Returns (weights, standardized scores, converged, iterations)."""
        max_iter = int(settings.get("max_iterations", 300))
        stop_criterion = float(settings.get("stop_criterion", 1e-7))
        scheme = settings.get("weighting_scheme", "path")
        weights = {lv: self._initial_weights(len(indicators[lv])) for lv in lvs}
        # Precompute numpy blocks once so the hot loop avoids per-indicator pandas access.
        block_arrays = {lv: z[indicators[lv]].to_numpy(dtype=float) for lv in lvs}
        converged = False
        scores = self._estimate_scores(z, indicators, weights)
        iteration = max_iter
        for iteration in range(1, max_iter + 1):
            previous = {lv: value.copy() for lv, value in weights.items()}
            inner = self._inner_weights(scores, lvs, scheme)
            for lv in lvs:
                proxy = self._inner_proxy(scores, inner, lv)
                if modes.get(lv, "reflective") == "formative":
                    new_weights = self._mode_b_weights(z[indicators[lv]], proxy)
                else:
                    new_weights = _vec_corr(block_arrays[lv], np.asarray(proxy, dtype=float))
                if not np.all(np.isfinite(new_weights)) or np.linalg.norm(new_weights) == 0:
                    new_weights = previous[lv]
                if np.dot(new_weights, previous[lv]) < 0:
                    new_weights = -new_weights
                weights[lv] = _normalize(new_weights)
            scores = self._estimate_scores(z, indicators, weights)
            max_diff = max(float(np.max(np.abs(weights[lv] - previous[lv]))) for lv in lvs)
            if max_diff < stop_criterion:
                converged = True
                break
        return weights, scores, converged, iteration

    def _run_pls_fast(
        self,
        z_values: np.ndarray,
        indicator_indices: dict[str, list[int]],
        modes: dict[str, str],
        lvs: list[str],
        settings: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, bool, int]:
        max_iter = int(settings.get("max_iterations", 300))
        stop_criterion = float(settings.get("stop_criterion", 1e-7))
        scheme = settings.get("weighting_scheme", "path")
        lv_index = {lv: index for index, lv in enumerate(lvs)}
        incoming: dict[int, list[int]] = {index: [] for index in range(len(lvs))}
        outgoing: dict[int, list[int]] = {index: [] for index in range(len(lvs))}
        for source, target in self.structural_model:
            if source in lv_index and target in lv_index:
                source_i, target_i = lv_index[source], lv_index[target]
                incoming[target_i].append(source_i)
                outgoing[source_i].append(target_i)

        weights = {lv: self._initial_weights(len(indicator_indices[lv])) for lv in lvs}
        block_arrays = {lv: z_values[:, indicator_indices[lv]] for lv in lvs}

        def estimate_scores() -> np.ndarray:
            columns = []
            for lv in lvs:
                columns.append(_standardize_array(block_arrays[lv] @ weights[lv]))
            return np.column_stack(columns)

        scores = estimate_scores()
        converged = False
        iteration = max_iter
        for iteration in range(1, max_iter + 1):
            previous = {lv: value.copy() for lv, value in weights.items()}
            inner = np.zeros((len(lvs), len(lvs)), dtype=float)
            for target_i in range(len(lvs)):
                predictors = incoming[target_i]
                if scheme == "path" and predictors:
                    beta = _ols_array(scores[:, predictors], scores[:, target_i])
                    for source_i, value in zip(predictors, beta):
                        inner[source_i, target_i] = value
                for other_i in set(incoming[target_i] + outgoing[target_i]):
                    if inner[other_i, target_i] != 0:
                        continue
                    corr = _corr_array(scores[:, other_i], scores[:, target_i])
                    inner[other_i, target_i] = np.sign(corr) if scheme == "centroid" and corr != 0 else corr

            for lv_i, lv in enumerate(lvs):
                inner_weights = inner[:, lv_i]
                if np.isclose(float(np.abs(inner_weights).sum()), 0.0):
                    proxy = scores[:, lv_i]
                else:
                    proxy = _standardize_array(scores @ inner_weights)
                    if np.nanstd(proxy, ddof=1) == 0:
                        proxy = scores[:, lv_i]
                if modes.get(lv, "reflective") == "formative":
                    new_weights = _ols_array(block_arrays[lv], proxy)
                else:
                    new_weights = _vec_corr(block_arrays[lv], proxy)
                if not np.all(np.isfinite(new_weights)) or np.linalg.norm(new_weights) == 0:
                    new_weights = previous[lv]
                if np.dot(new_weights, previous[lv]) < 0:
                    new_weights = -new_weights
                weights[lv] = _normalize(new_weights)

            scores = estimate_scores()
            max_diff = max(float(np.max(np.abs(weights[lv] - previous[lv]))) for lv in lvs)
            if max_diff < stop_criterion:
                converged = True
                break
        return weights, scores, converged, iteration

    def _fit_paths_only(self, frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
        """Lightweight fit returning only the (effects-augmented) path-coefficient matrix.

        Used by the resampling loops (MGA / permutation) which need paths, not the full
        measurement report — avoids recomputing loadings/HTMT/model-fit per subsample.
        """
        context = self._make_context(frame, settings)
        _weights, scores, _conv, _it = self._run_pls(
            context.z, context.indicators, context.modes, context.lvs, settings
        )
        analysis_scores, analysis_lvs, analysis_structural = self._build_effects_layer(scores, context)
        paths, _r2, _adj = self._estimate_paths(analysis_scores, analysis_lvs, analysis_structural)
        return paths

    def _paths_from_numeric(self, numeric_frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
        """Paths from an already-coerced, complete-case numeric frame (fast resampling path)."""
        z = standardize_frame(numeric_frame)
        lvs = list(self.measurement_model.keys())
        indicators = {lv: [i for i in self.measurement_model[lv] if i in z.columns] for lv in lvs}
        modes = {lv: self.measurement_modes.get(lv, "reflective") for lv in lvs}
        context = FitContext(lvs=lvs, indicators=indicators, modes=modes, z=z, warnings=[], original=numeric_frame)
        _weights, scores, _conv, _it = self._run_pls(z, indicators, modes, lvs, settings)
        analysis_scores, analysis_lvs, analysis_structural = self._build_effects_layer(scores, context)
        paths, _r2, _adj = self._estimate_paths(analysis_scores, analysis_lvs, analysis_structural)
        return paths

    def _fit_bootstrap_numeric(self, numeric_frame: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
        """Fast fit for bootstrap samples.

        It computes only metrics used by _flatten_result, skipping report-only tables
        such as HTMT, cross-loadings, VIF, f-square and model fit.
        """
        std = numeric_frame.std(ddof=1).replace(0, np.nan)
        z = ((numeric_frame - numeric_frame.mean()) / std).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if z.empty:
            raise ValueError("Bootstrap sample has no complete standardized rows.")
        lvs = list(self.measurement_model.keys())
        indicators = {lv: [i for i in self.measurement_model[lv] if i in z.columns] for lv in lvs}
        if any(not values for values in indicators.values()):
            raise ValueError("Bootstrap sample is missing indicators.")
        modes = {lv: self.measurement_modes.get(lv, "reflective") for lv in lvs}
        context = FitContext(lvs=lvs, indicators=indicators, modes=modes, z=z, warnings=[], original=numeric_frame)
        column_index = {column: index for index, column in enumerate(z.columns)}
        indicator_indices = {lv: [column_index[indicator] for indicator in indicators[lv]] for lv in lvs}
        weights, score_values, _converged, _iteration = self._run_pls_fast(
            z.to_numpy(dtype=float), indicator_indices, modes, lvs, settings
        )
        scores = pd.DataFrame(score_values, index=z.index, columns=lvs)
        analysis_scores, analysis_lvs, analysis_structural = self._build_effects_layer(scores, context)
        path_coefficients, r2, _adjusted_r2 = self._estimate_paths(analysis_scores, analysis_lvs, analysis_structural)
        outer_loadings = self._outer_loadings(context.z, scores, context)
        outer_weights = self._outer_weights_frame(weights, context)
        reliability = self._reliability(context.z, outer_loadings, context, weights)
        total_effects, indirect_effects = self._effects(path_coefficients, analysis_lvs)
        return {
            "measurement_model": context.indicators,
            "structural_model": analysis_structural,
            "path_coefficients": path_coefficients,
            "outer_loadings": outer_loadings,
            "outer_weights": outer_weights,
            "r_square": pd.Series(r2, name="R2"),
            "reliability": reliability,
            "total_effects": total_effects,
            "indirect_effects": indirect_effects,
        }

    def _fit_bootstrap_flat_numeric(self, numeric_frame: pd.DataFrame, settings: dict[str, Any]) -> dict[str, float]:
        std = numeric_frame.std(ddof=1).replace(0, np.nan)
        z = ((numeric_frame - numeric_frame.mean()) / std).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if z.empty:
            raise ValueError("Bootstrap sample has no complete standardized rows.")
        z_values = z.to_numpy(dtype=float)
        n = z_values.shape[0]
        column_index = {column: index for index, column in enumerate(z.columns)}
        lvs = list(self.measurement_model.keys())
        indicators = {lv: [i for i in self.measurement_model[lv] if i in column_index] for lv in lvs}
        if any(not values for values in indicators.values()):
            raise ValueError("Bootstrap sample is missing indicators.")
        indicator_indices = {lv: [column_index[indicator] for indicator in indicators[lv]] for lv in lvs}
        modes = {lv: self.measurement_modes.get(lv, "reflective") for lv in lvs}
        weights, scores, _converged, _iteration = self._run_pls_fast(z_values, indicator_indices, modes, lvs, settings)

        analysis_scores, analysis_lvs, analysis_structural = self._build_effects_layer_fast(scores, lvs)
        path_matrix, r2 = self._estimate_paths_fast(analysis_scores, analysis_lvs, analysis_structural)
        total_effects, indirect_effects = _effects_array(path_matrix)
        analysis_index = {lv: index for index, lv in enumerate(analysis_lvs)}
        lv_index = {lv: index for index, lv in enumerate(lvs)}
        flat: dict[str, float] = {}

        # Unit-length centred columns → correlation matrices via plain matrix products.
        zc = z_values - z_values.mean(axis=0)
        z_norm = np.sqrt((zc**2).sum(axis=0))
        zc_u = zc / np.where(z_norm > 0, z_norm, 1.0)
        sc = scores - scores.mean(axis=0)
        s_norm = np.sqrt((sc**2).sum(axis=0))
        sc_u = sc / np.where(s_norm > 0, s_norm, 1.0)
        indicator_corr = zc_u.T @ zc_u            # indicator correlation matrix
        abs_corr = np.abs(indicator_corr)
        loadings_matrix = zc_u.T @ sc_u           # indicator × construct loadings
        score_corr = sc_u.T @ sc_u                # latent variable correlations

        # Direct paths.
        for source, target in analysis_structural:
            source_i = analysis_index.get(source)
            target_i = analysis_index.get(target)
            if source_i is not None and target_i is not None:
                flat[f"path:{source} -> {target}"] = float(path_matrix[source_i, target_i])

        # Specific indirect effects (each mediation chain).
        for chain in _indirect_chains(analysis_structural, analysis_lvs):
            flat[f"specific:{' -> '.join(chain)}"] = _chain_specific(path_matrix, analysis_index, chain)

        # Outer loadings and weights.
        for construct in lvs:
            construct_i = lv_index[construct]
            weight_values = weights.get(construct)
            for local_i, indicator in enumerate(indicators[construct]):
                indicator_i = column_index[indicator]
                flat[f"loading:{indicator} <- {construct}"] = float(loadings_matrix[indicator_i, construct_i])
                if weight_values is not None:
                    flat[f"weight:{indicator} <- {construct}"] = float(weight_values[local_i])

        # R² and adjusted R².
        targets = {target for _source, target in analysis_structural}
        for construct, value in r2.items():
            flat[f"r2:{construct}"] = float(value)
        for target in targets:
            predictors = [s for s, t in analysis_structural if t == target]
            p = len(predictors)
            included = r2.get(target)
            if included is not None and n > p + 1:
                flat[f"radj:{target}"] = float(1 - (1 - included) * (n - 1) / (n - p - 1))

        # f² effect sizes.
        for target in targets:
            target_i = analysis_index.get(target)
            predictors = [s for s, t in analysis_structural if t == target]
            if target_i is None or not predictors:
                continue
            included = r2.get(target, 0.0)
            denom = 1 - included
            for source in predictors:
                others = [analysis_index[o] for o in predictors if o != source and o in analysis_index]
                if others:
                    _beta, excluded = _ols_r2_array(analysis_scores[:, others], analysis_scores[:, target_i])
                else:
                    excluded = 0.0
                flat[f"f2:{source} -> {target}"] = float((included - excluded) / denom) if denom > 0 else np.nan

        # Reliability (Cronbach α, composite reliability, AVE, rho_A).
        for construct in lvs:
            construct_i = lv_index[construct]
            block_indices = indicator_indices[construct]
            loadings = loadings_matrix[block_indices, construct_i]
            block = z_values[:, block_indices]
            k = len(block_indices)
            if k == 1:
                alpha = composite = ave = rho_a = 1.0
            else:
                total = block.sum(axis=1)
                total_var = float(np.var(total, ddof=1))
                item_var_sum = float(np.var(block, axis=0, ddof=1).sum())
                alpha = (k / (k - 1)) * (1 - item_var_sum / total_var) if total_var else np.nan
                error_var = np.sum(np.maximum(0.0, 1 - loadings**2))
                composite = (np.sum(loadings) ** 2) / ((np.sum(loadings) ** 2) + error_var) if error_var >= 0 else np.nan
                ave = float(np.mean(loadings**2))
                rho_a = _rho_a_block(weights[construct], indicator_corr[np.ix_(block_indices, block_indices)])
                if not np.isfinite(rho_a):
                    rho_a = alpha
            flat[f"Cronbach alpha:{construct}"] = float(alpha)
            flat[f"Composite reliability:{construct}"] = float(composite)
            flat[f"AVE:{construct}"] = float(ave)
            flat[f"rho_A:{construct}"] = float(rho_a)

        # HTMT (reflective constructs with at least two indicators).
        eligible = [lv for lv in lvs if modes.get(lv, "reflective") != "formative" and len(indicators[lv]) >= 2]
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                a, b = eligible[i], eligible[j]
                flat[f"htmt:{a} <-> {b}"] = _htmt_from_corr(abs_corr, indicator_indices[a], indicator_indices[b])

        # Latent variable correlations (base constructs).
        for i in range(len(lvs)):
            for j in range(i + 1, len(lvs)):
                a, b = lvs[i], lvs[j]
                flat[f"lvcorr:{a} <-> {b}"] = float(score_corr[lv_index[a], lv_index[b]])

        # Total effects (all reachable pairs) and total indirect effects (mediated pairs).
        for source, target in _total_pairs(analysis_structural, analysis_lvs):
            source_i = analysis_index.get(source)
            target_i = analysis_index.get(target)
            if source_i is not None and target_i is not None:
                flat[f"total:{source} -> {target}"] = float(total_effects[source_i, target_i])
        for source, target in _indirect_pairs(analysis_structural, analysis_lvs):
            source_i = analysis_index.get(source)
            target_i = analysis_index.get(target)
            if source_i is not None and target_i is not None:
                flat[f"total_indirect:{source} -> {target}"] = float(indirect_effects[source_i, target_i])

        # Model fit (SRMR, d_ULS, d_G).
        ordered = list(dict.fromkeys(ind for lv in lvs for ind in indicators[lv]))
        ord_idx = [column_index[ind] for ind in ordered]
        observed = indicator_corr[np.ix_(ord_idx, ord_idx)]
        lambda_matrix = np.zeros((len(ordered), len(lvs)))
        for row, indicator in enumerate(ordered):
            for col, lv in enumerate(lvs):
                if indicator in indicators[lv]:
                    lambda_matrix[row, col] = float(loadings_matrix[column_index[indicator], lv_index[lv]])
        phi = score_corr if score_corr.ndim == 2 else np.atleast_2d(score_corr)
        implied = lambda_matrix @ phi @ lambda_matrix.T
        np.fill_diagonal(implied, 1.0)
        mask = np.triu(np.ones_like(observed, dtype=bool), k=1)
        if mask.any():
            residual = observed - implied
            p = observed.shape[0]
            flat["model_fit:SRMR"] = float(np.sqrt(np.sum(residual[mask] ** 2) / (p * (p + 1) / 2)))
            flat["model_fit:d_ULS"] = float(np.sum(residual[mask] ** 2))
        flat["model_fit:d_G"] = self._geodesic_distance(observed, implied)

        return {key: value for key, value in flat.items() if np.isfinite(value)}

    def _build_effects_layer_fast(self, scores: np.ndarray, lvs: list[str]) -> tuple[np.ndarray, list[str], list[tuple[str, str]]]:
        columns = [scores[:, index] for index in range(scores.shape[1])]
        analysis_lvs = list(lvs)
        analysis_structural = list(self.structural_model)
        base = set(lvs)
        lv_index = {lv: index for index, lv in enumerate(lvs)}
        for effect in self.effects:
            etype = effect.get("type")
            name = str(effect.get("name", "")).strip()
            if not name or name in analysis_lvs:
                continue
            if etype == "interaction":
                predictor = str(effect.get("predictor", "")).strip()
                moderator = str(effect.get("moderator", "")).strip()
                outcome = str(effect.get("outcome", "")).strip()
                if not {predictor, moderator, outcome} <= base:
                    continue
                columns.append(_standardize_array(scores[:, lv_index[predictor]] * scores[:, lv_index[moderator]]))
                analysis_lvs.append(name)
                analysis_structural.extend([(name, outcome), (moderator, outcome), (predictor, outcome)])
            elif etype == "quadratic":
                source = str(effect.get("source", "")).strip()
                outcome = str(effect.get("outcome", "")).strip()
                if not {source, outcome} <= base:
                    continue
                columns.append(_standardize_array(scores[:, lv_index[source]] ** 2))
                analysis_lvs.append(name)
                analysis_structural.extend([(name, outcome), (source, outcome)])
        return np.column_stack(columns), analysis_lvs, list(dict.fromkeys(analysis_structural))

    def _estimate_paths_fast(
        self,
        scores: np.ndarray,
        lvs: list[str],
        structural: list[tuple[str, str]],
    ) -> tuple[np.ndarray, dict[str, float]]:
        lv_index = {lv: index for index, lv in enumerate(lvs)}
        coefficients = np.zeros((len(lvs), len(lvs)), dtype=float)
        r2: dict[str, float] = {}
        for target in {target for _source, target in structural}:
            sources = [source for source, current_target in structural if current_target == target]
            source_indices = [lv_index[source] for source in sources if source in lv_index]
            target_i = lv_index.get(target)
            if target_i is None or not source_indices:
                continue
            beta, target_r2 = _ols_r2_array(scores[:, source_indices], scores[:, target_i])
            for source_i, value in zip(source_indices, beta):
                coefficients[source_i, target_i] = value
            r2[target] = target_r2
        return coefficients, r2

    def _fit(self, frame: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
        context = self._make_context(frame, settings)
        weights, scores, converged, iteration = self._run_pls(
            context.z, context.indicators, context.modes, context.lvs, settings
        )

        analysis_scores, analysis_lvs, analysis_structural = self._build_effects_layer(scores, context)
        path_coefficients, r2, adjusted_r2 = self._estimate_paths(analysis_scores, analysis_lvs, analysis_structural)
        f_square = self._f_square(analysis_scores, r2, analysis_structural)
        inner_vif = self._inner_vif(analysis_scores, analysis_structural)
        outer_loadings = self._outer_loadings(context.z, scores, context)
        outer_vif = self._outer_vif(context.z, context)
        outer_weights = self._outer_weights_frame(weights, context)
        reliability = self._reliability(context.z, outer_loadings, context, weights)
        lv_correlations = scores.corr()
        cross_loadings = self._cross_loadings(context.z, scores, context)
        fornell_larcker = self._fornell_larcker(lv_correlations, reliability)
        htmt = self._htmt(context.z, context)
        total_effects, indirect_effects = self._effects(path_coefficients, analysis_lvs)
        model_fit = self._model_fit(context.z, lv_correlations, outer_loadings, context)
        ordered_indicators = list(
            dict.fromkeys(indicator for values in context.indicators.values() for indicator in values)
        )

        result = {
            "algorithm": "PLS-SEM",
            "converged": converged,
            "iterations": iteration,
            "n_observations": int(context.z.shape[0]),
            "n_indicators": int(context.z.shape[1]),
            "weighting_scheme": settings.get("weighting_scheme", "path"),
            "max_iterations": int(settings.get("max_iterations", 300)),
            "stop_criterion": float(settings.get("stop_criterion", 1e-7)),
            "indicator_data_original": context.original[ordered_indicators].reset_index(drop=True),
            "indicator_data_standardized": context.z[ordered_indicators].reset_index(drop=True),
            "indicator_correlations": context.z[ordered_indicators].corr(),
            "measurement_model": context.indicators,
            "measurement_modes": context.modes,
            "structural_model": analysis_structural,
            "effects": list(self.effects),
            "scores": analysis_scores,
            "path_coefficients": path_coefficients,
            "r_square": pd.Series(r2, name="R2"),
            "adjusted_r_square": pd.Series(adjusted_r2, name="Adjusted R2"),
            "f_square": f_square,
            "inner_vif": inner_vif,
            "outer_vif": outer_vif,
            "outer_loadings": outer_loadings,
            "outer_weights": outer_weights,
            "reliability": reliability,
            "lv_correlations": lv_correlations,
            "cross_loadings": cross_loadings,
            "fornell_larcker": fornell_larcker,
            "htmt": htmt,
            "total_effects": total_effects,
            "indirect_effects": indirect_effects,
            "model_fit": model_fit,
            "diagnostics": context.warnings,
        }
        result["interpretation"] = self._make_interpretation(result)
        return result

    def _make_context(self, frame: pd.DataFrame, settings: dict[str, Any]) -> FitContext:
        errors, warnings = self.validate_model(frame)
        if errors:
            raise ModelValidationError("\n".join(errors))

        all_indicators = [
            indicator
            for indicators in self.measurement_model.values()
            for indicator in indicators
        ]
        analysis_frame, data_warnings = prepare_analysis_frame(
            frame,
            all_indicators,
            settings.get("missing_strategy", "casewise"),
        )
        warnings.extend(data_warnings)
        z = standardize_frame(analysis_frame)

        lvs = list(self.measurement_model.keys())
        indicators = {
            lv: [indicator for indicator in self.measurement_model[lv] if indicator in z.columns]
            for lv in lvs
        }
        modes = {
            lv: self.measurement_modes.get(lv, "reflective")
            for lv in lvs
        }
        return FitContext(lvs=lvs, indicators=indicators, modes=modes, z=z, warnings=warnings, original=analysis_frame)

    def _build_effects_layer(
        self,
        scores: pd.DataFrame,
        context: FitContext,
    ) -> tuple[pd.DataFrame, list[str], list[tuple[str, str]]]:
        """Two-stage interaction/quadratic terms built from converged latent scores.

        Returns an augmented (scores, lvs, structural) used only for the structural
        (path/R2/f2/VIF) estimation; the measurement side stays on the base constructs.
        """
        analysis_scores = scores.copy()
        analysis_lvs = list(context.lvs)
        analysis_structural = list(self.structural_model)
        base = set(context.lvs)
        for effect in self.effects:
            etype = effect.get("type")
            name = str(effect.get("name", "")).strip()
            if not name or name in analysis_scores.columns:
                continue
            if etype == "interaction":
                predictor = str(effect.get("predictor", "")).strip()
                moderator = str(effect.get("moderator", "")).strip()
                outcome = str(effect.get("outcome", "")).strip()
                if not {predictor, moderator, outcome} <= base:
                    context.warnings.append(
                        f"Bỏ qua hiệu ứng điều tiết '{name}': thiếu biến tiềm ẩn liên quan."
                    )
                    continue
                product = _standardize_array(
                    scores[predictor].to_numpy(dtype=float) * scores[moderator].to_numpy(dtype=float)
                )
                analysis_scores[name] = product
                analysis_lvs.append(name)
                analysis_structural.extend([(name, outcome), (moderator, outcome), (predictor, outcome)])
            elif etype == "quadratic":
                source = str(effect.get("source", "")).strip()
                outcome = str(effect.get("outcome", "")).strip()
                if not {source, outcome} <= base:
                    context.warnings.append(
                        f"Bỏ qua hiệu ứng bậc hai '{name}': thiếu biến tiềm ẩn liên quan."
                    )
                    continue
                squared = _standardize_array(scores[source].to_numpy(dtype=float) ** 2)
                analysis_scores[name] = squared
                analysis_lvs.append(name)
                analysis_structural.extend([(name, outcome), (source, outcome)])
        analysis_structural = list(dict.fromkeys(analysis_structural))
        return analysis_scores, analysis_lvs, analysis_structural

    def _estimate_scores(
        self,
        z: pd.DataFrame,
        indicators: dict[str, list[str]],
        weights: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        scores = pd.DataFrame(index=z.index)
        for lv, indicator_names in indicators.items():
            raw = z[indicator_names].to_numpy(dtype=float) @ weights[lv]
            scores[lv] = _standardize_array(raw)
        return scores

    def _inner_weights(self, scores: pd.DataFrame, lvs: list[str], scheme: str) -> pd.DataFrame:
        inner = pd.DataFrame(0.0, index=lvs, columns=lvs)
        incoming: dict[str, list[str]] = {lv: [] for lv in lvs}
        outgoing: dict[str, list[str]] = {lv: [] for lv in lvs}
        for source, target in self.structural_model:
            incoming[target].append(source)
            outgoing[source].append(target)

        for lv in lvs:
            predictors = incoming[lv]
            if scheme == "path" and predictors:
                beta, _, _, _ = _ols(scores[predictors], scores[lv])
                for source, value in zip(predictors, beta):
                    inner.loc[source, lv] = value

            related = set(incoming[lv] + outgoing[lv])
            for other in related:
                if inner.loc[other, lv] != 0:
                    continue
                corr = _corr(scores[other], scores[lv])
                if scheme == "centroid":
                    inner.loc[other, lv] = np.sign(corr) if corr != 0 else 0.0
                else:
                    inner.loc[other, lv] = corr

        return inner

    def _inner_proxy(self, scores: pd.DataFrame, inner: pd.DataFrame, lv: str) -> pd.Series:
        weights = inner[lv]
        if np.isclose(float(np.abs(weights).sum()), 0.0):
            return scores[lv]
        proxy = scores.to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
        if np.nanstd(proxy, ddof=1) == 0:
            return scores[lv]
        return pd.Series(_standardize_array(proxy), index=scores.index)

    def _mode_b_weights(self, indicators: pd.DataFrame, proxy: pd.Series) -> np.ndarray:
        beta, _, _, _ = _ols(indicators, proxy)
        return np.array(beta, dtype=float)

    def _estimate_paths(
        self,
        scores: pd.DataFrame,
        lvs: list[str],
        structural: list[tuple[str, str]] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
        structural = self.structural_model if structural is None else structural
        coefficients = pd.DataFrame(0.0, index=lvs, columns=lvs)
        r2: dict[str, float] = {}
        adjusted_r2: dict[str, float] = {}

        for target in {target for _, target in structural}:
            sources = [source for source, current_target in structural if current_target == target]
            if not sources:
                continue
            beta, target_r2, target_adjusted, _ = _ols(scores[sources], scores[target])
            for source, value in zip(sources, beta):
                coefficients.loc[source, target] = value
            r2[target] = target_r2
            adjusted_r2[target] = target_adjusted

        return coefficients, r2, adjusted_r2

    def _outer_loadings(self, z: pd.DataFrame, scores: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        indicators = list(dict.fromkeys(indicator for values in context.indicators.values() for indicator in values))
        loadings = pd.DataFrame(0.0, index=indicators, columns=context.lvs)
        for indicator in indicators:
            for lv in context.lvs:
                loadings.loc[indicator, lv] = _corr(z[indicator], scores[lv])
        primary = pd.DataFrame(index=indicators)
        primary["Construct"] = ""
        primary["Primary loading"] = np.nan
        for lv, indicator_names in context.indicators.items():
            for indicator in indicator_names:
                primary.loc[indicator, "Construct"] = lv
                primary.loc[indicator, "Primary loading"] = float(loadings.loc[indicator, lv])
        return pd.concat([primary, loadings], axis=1)

    def _outer_weights_frame(self, weights: dict[str, np.ndarray], context: FitContext) -> pd.DataFrame:
        indicators = list(dict.fromkeys(indicator for values in context.indicators.values() for indicator in values))
        table = pd.DataFrame(0.0, index=indicators, columns=context.lvs)
        for lv, indicator_names in context.indicators.items():
            for indicator, weight in zip(indicator_names, weights[lv]):
                table.loc[indicator, lv] = weight
        return table

    def _reliability(
        self,
        z: pd.DataFrame,
        outer_loadings: pd.DataFrame,
        context: FitContext,
        weights: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for lv, indicator_names in context.indicators.items():
            loadings = pd.to_numeric(outer_loadings.loc[indicator_names, lv], errors="coerce").to_numpy(dtype=float)
            block = z[indicator_names]
            k = len(indicator_names)
            if k == 1:
                alpha = rho_a = composite = ave = 1.0
            else:
                total = block.sum(axis=1)
                total_var = float(total.var(ddof=1))
                item_var_sum = float(block.var(ddof=1).sum())
                alpha = (k / (k - 1)) * (1 - item_var_sum / total_var) if total_var else np.nan
                error_var = np.sum(np.maximum(0.0, 1 - loadings**2))
                composite = (np.sum(loadings) ** 2) / ((np.sum(loadings) ** 2) + error_var) if error_var >= 0 else np.nan
                ave = float(np.mean(loadings**2))
                if weights is not None and lv in weights:
                    rho_a = _rho_a_block(np.asarray(weights[lv], dtype=float), block.corr().to_numpy(dtype=float))
                    if not np.isfinite(rho_a):
                        rho_a = alpha
                else:
                    rho_a = alpha
            rows.append(
                {
                    "Construct": lv,
                    "Mode": context.modes.get(lv, "reflective"),
                    "Indicators": k,
                    "Cronbach alpha": alpha,
                    "rho_A approx.": rho_a,
                    "Composite reliability": composite,
                    "AVE": ave,
                }
            )
        return pd.DataFrame(rows).set_index("Construct")

    def _cross_loadings(self, z: pd.DataFrame, scores: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        indicators = list(dict.fromkeys(indicator for values in context.indicators.values() for indicator in values))
        table = pd.DataFrame(index=indicators, columns=context.lvs, dtype=float)
        for indicator in indicators:
            for lv in context.lvs:
                table.loc[indicator, lv] = _corr(z[indicator], scores[lv])
        return table

    def _fornell_larcker(self, lv_correlations: pd.DataFrame, reliability: pd.DataFrame) -> pd.DataFrame:
        table = lv_correlations.copy()
        for lv in table.index:
            ave = reliability.loc[lv, "AVE"] if lv in reliability.index else np.nan
            table.loc[lv, lv] = np.sqrt(ave) if pd.notna(ave) else np.nan
        return table

    def _htmt(self, z: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        table = pd.DataFrame(np.nan, index=context.lvs, columns=context.lvs)
        corr = z.corr().abs()

        for lv_a in context.lvs:
            table.loc[lv_a, lv_a] = 1.0
            for lv_b in context.lvs:
                if lv_a == lv_b:
                    continue
                indicators_a = context.indicators[lv_a]
                indicators_b = context.indicators[lv_b]
                heterotrait = corr.loc[indicators_a, indicators_b].to_numpy(dtype=float).ravel()
                monotrait_a = _upper_triangle_values(corr.loc[indicators_a, indicators_a])
                monotrait_b = _upper_triangle_values(corr.loc[indicators_b, indicators_b])
                denom = np.sqrt(np.nanmean(monotrait_a) * np.nanmean(monotrait_b))
                table.loc[lv_a, lv_b] = np.nanmean(heterotrait) / denom if denom and np.isfinite(denom) else np.nan
        return table

    def _f_square(
        self,
        scores: pd.DataFrame,
        r2: dict[str, float],
        structural: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        structural = self.structural_model if structural is None else structural
        rows: list[dict[str, Any]] = []
        for source, target in structural:
            sources = [s for s, t in structural if t == target]
            if source not in sources or len(sources) == 0:
                continue
            included = r2.get(target, 0.0)
            reduced_sources = [candidate for candidate in sources if candidate != source]
            if reduced_sources:
                _, excluded, _, _ = _ols(scores[reduced_sources], scores[target])
            else:
                excluded = 0.0
            denominator = 1 - included
            f2 = (included - excluded) / denominator if denominator > 0 else np.nan
            rows.append({"Path": f"{source} -> {target}", "Included R2": included, "Excluded R2": excluded, "f2": f2})
        return pd.DataFrame(rows).set_index("Path") if rows else pd.DataFrame(columns=["Included R2", "Excluded R2", "f2"])

    def _inner_vif(
        self,
        scores: pd.DataFrame,
        structural: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        structural = self.structural_model if structural is None else structural
        rows: list[dict[str, Any]] = []
        for target in {target for _, target in structural}:
            sources = [source for source, current_target in structural if current_target == target]
            for source in sources:
                others = [candidate for candidate in sources if candidate != source]
                if not others:
                    vif = 1.0
                else:
                    _, r2, _, _ = _ols(scores[others], scores[source])
                    vif = 1 / (1 - r2) if r2 < 1 else np.inf
                rows.append({"Target": target, "Predictor": source, "VIF": vif})
        return pd.DataFrame(rows).set_index(["Target", "Predictor"]) if rows else pd.DataFrame(columns=["VIF"])

    def _outer_vif(self, z: pd.DataFrame, context: FitContext) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for construct, indicators in context.indicators.items():
            for indicator in indicators:
                others = [candidate for candidate in indicators if candidate != indicator]
                if not others:
                    vif = 1.0
                else:
                    _, r2, _, _ = _ols(z[others], z[indicator])
                    vif = 1 / (1 - r2) if r2 < 1 else np.inf
                rows.append({"Indicator": indicator, "Construct": construct, "VIF": vif})
        return pd.DataFrame(rows).set_index("Indicator") if rows else pd.DataFrame(columns=["Construct", "VIF"])

    def _effects(self, path_coefficients: pd.DataFrame, lvs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        b = path_coefficients.loc[lvs, lvs].to_numpy(dtype=float)
        identity = np.eye(len(lvs))
        try:
            total = np.linalg.inv(identity - b) - identity
        except np.linalg.LinAlgError:
            total = np.zeros_like(b)
            power = b.copy()
            for _ in range(len(lvs)):
                total += power
                power = power @ b
        total_df = pd.DataFrame(total, index=lvs, columns=lvs)
        indirect_df = total_df - path_coefficients.loc[lvs, lvs]
        return total_df, indirect_df

    def _model_fit(
        self,
        z: pd.DataFrame,
        lv_correlations: pd.DataFrame,
        outer_loadings: pd.DataFrame,
        context: FitContext,
    ) -> pd.DataFrame:
        indicators = list(dict.fromkeys(indicator for values in context.indicators.values() for indicator in values))
        observed = z[indicators].corr().to_numpy(dtype=float)
        lambda_matrix = np.zeros((len(indicators), len(context.lvs)))
        for i, indicator in enumerate(indicators):
            for j, lv in enumerate(context.lvs):
                if indicator in context.indicators[lv]:
                    lambda_matrix[i, j] = float(outer_loadings.loc[indicator, lv])
        phi = lv_correlations.loc[context.lvs, context.lvs].to_numpy(dtype=float)
        implied = lambda_matrix @ phi @ lambda_matrix.T
        np.fill_diagonal(implied, 1.0)
        mask = np.triu(np.ones_like(observed, dtype=bool), k=1)
        residual = observed - implied
        if mask.any():
            p = observed.shape[0]
            srmr = float(np.sqrt(np.sum(residual[mask] ** 2) / (p * (p + 1) / 2)))
            d_uls = float(np.sum(residual[mask] ** 2))
        else:
            srmr = d_uls = np.nan
        d_g = self._geodesic_distance(observed, implied)
        return pd.DataFrame(
            [
                {"Metric": "SRMR", "Saturated Model": srmr, "Estimated Model": srmr},
                {"Metric": "d_ULS", "Saturated Model": d_uls, "Estimated Model": d_uls},
                {"Metric": "d_G", "Saturated Model": d_g, "Estimated Model": d_g},
            ]
        ).set_index("Metric")

    @staticmethod
    def _geodesic_distance(observed: np.ndarray, implied: np.ndarray) -> float:
        """Geodesic discrepancy d_G = 0.5 * sum (ln eigenvalue_k)^2 of inv(observed)·implied."""
        try:
            eigenvalues = np.linalg.eigvals(np.linalg.solve(observed, implied))
        except np.linalg.LinAlgError:
            return np.nan
        eigenvalues = np.real(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if eigenvalues.size == 0:
            return np.nan
        return float(0.5 * np.sum(np.log(eigenvalues) ** 2))

    def _bootstrap(
        self,
        frame: pd.DataFrame,
        settings: dict[str, Any],
        base_result: dict[str, Any],
        subsamples: int,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        progress_callback = settings.get("progress_callback")
        progress_interval = int(settings.get("progress_interval", max(1, subsamples // 200)))
        rng = np.random.default_rng(int(settings.get("random_seed", 12345)))
        ci_level = float(settings.get("confidence_level", 0.95))
        alpha = 1 - ci_level
        boot_settings = dict(settings)
        boot_settings["bootstrap_enabled"] = False
        boot_settings.pop("progress_callback", None)
        sample_base = self._make_context(frame, boot_settings).original.reset_index(drop=True)
        rows = len(sample_base)
        original = self._flatten_result(base_result)
        values: dict[str, list[float]] = {key: [] for key in original}

        if bool(settings.get("bootstrap_parallel", True)) and subsamples >= 50:
            try:
                values = self._bootstrap_parallel(
                    sample_base,
                    boot_settings,
                    values,
                    subsamples,
                    int(settings.get("random_seed", 12345)),
                    progress_callback,
                )
            except ModelValidationError:
                raise
            except Exception:
                values = {key: [] for key in original}
                self._bootstrap_serial(sample_base, boot_settings, values, subsamples, rng, progress_callback, progress_interval)
        else:
            self._bootstrap_serial(sample_base, boot_settings, values, subsamples, rng, progress_callback, progress_interval)

        one_tailed = "one" in str(settings.get("test_type", "two-tailed")).lower()
        samples: dict[str, np.ndarray] = {}
        summary_rows: list[dict[str, Any]] = []
        for key, original_value in original.items():
            sample_values = np.array(values.get(key, []), dtype=float)
            sample_values = sample_values[np.isfinite(sample_values)]
            samples[key] = sample_values
            if sample_values.size < 2:
                mean = stdev = bias = t_value = p_value = np.nan
                ci_lower = ci_upper = bc_lower = bc_upper = np.nan
            else:
                mean = float(np.mean(sample_values))
                stdev = float(np.std(sample_values, ddof=1))
                bias = mean - original_value
                t_value = float(abs(original_value) / stdev) if stdev > 0 else np.nan
                if np.isfinite(t_value):
                    sf = float(stats.t.sf(abs(t_value), df=max(sample_values.size - 1, 1)))
                    p_value = sf if one_tailed else 2 * sf
                else:
                    p_value = np.nan
                ci_lower = float(np.quantile(sample_values, alpha / 2))
                ci_upper = float(np.quantile(sample_values, 1 - alpha / 2))
                bc_lower, bc_upper = _bias_corrected_bounds(sample_values, original_value, alpha)
            summary_rows.append(
                {
                    "Metric": key,
                    "Original": original_value,
                    "Mean": mean,
                    "STDEV": stdev,
                    "Bias": bias,
                    "T statistic": t_value,
                    "P value": p_value,
                    "CI lower": ci_lower,
                    "CI upper": ci_upper,
                    "CI lower BC": bc_lower,
                    "CI upper BC": bc_upper,
                    "Valid samples": int(sample_values.size),
                }
            )
        return pd.DataFrame(summary_rows).set_index("Metric"), samples

    def _bootstrap_serial(
        self,
        sample_base: pd.DataFrame,
        boot_settings: dict[str, Any],
        values: dict[str, list[float]],
        subsamples: int,
        rng: np.random.Generator,
        progress_callback,
        progress_interval: int,
    ) -> dict[str, list[float]]:
        rows = len(sample_base)
        for completed in range(1, subsamples + 1):
            sample_index = rng.integers(0, rows, size=rows)
            sample = sample_base.iloc[sample_index].reset_index(drop=True)
            try:
                flat = self._fit_bootstrap_flat_numeric(sample, boot_settings)
            except Exception:
                continue
            for key in values:
                value = flat.get(key, np.nan)
                if pd.notna(value):
                    values[key].append(float(value))
            if progress_callback and (completed == 1 or completed == subsamples or completed % progress_interval == 0):
                valid_samples = max((len(sample_values) for sample_values in values.values()), default=0)
                should_continue = progress_callback(completed, subsamples, valid_samples)
                if should_continue is False:
                    raise ModelValidationError("Bootstrapping đã bị hủy.")
        return values

    def _bootstrap_parallel(
        self,
        sample_base: pd.DataFrame,
        boot_settings: dict[str, Any],
        values: dict[str, list[float]],
        subsamples: int,
        seed: int,
        progress_callback,
    ) -> dict[str, list[float]]:
        cpu_count = os.cpu_count() or 1
        requested = int(boot_settings.get("bootstrap_workers", 0) or 0)
        workers = requested if requested > 0 else max(1, cpu_count - 1)
        workers = max(1, min(workers, cpu_count, subsamples))
        if workers <= 1:
            raise RuntimeError("Parallel bootstrap requires more than one worker.")

        # Split into several small chunks (≈ workers × 4) so futures complete frequently
        # for smooth progress and good load balancing; tasks carry only (count, seed).
        target_chunks = min(subsamples, max(workers, workers * 4))
        base_chunk = subsamples // target_chunks
        remainder = subsamples % target_chunks
        sizes = [base_chunk + (1 if index < remainder else 0) for index in range(target_chunks)]
        sizes = [size for size in sizes if size > 0]
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds = [int(seq.generate_state(1)[0]) for seq in seed_sequence.spawn(len(sizes))]
        tasks = [(size, child_seeds[index]) for index, size in enumerate(sizes)]

        common = {
            "sample_base": sample_base,
            "keys": list(values.keys()),
            "settings": boot_settings,
            "measurement_model": self.measurement_model,
            "structural_model": self.structural_model,
            "measurement_modes": self.measurement_modes,
            "effects": self.effects,
        }

        completed = 0
        saved_env = {var: os.environ.get(var) for var in _THREAD_ENV_VARS}
        for var in _THREAD_ENV_VARS:
            os.environ[var] = "1"  # children inherit at spawn → single-threaded BLAS
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(common,)) as pool:
                futures = [pool.submit(_bootstrap_chunk_worker, task) for task in tasks]
                for future in as_completed(futures):
                    chunk_values, chunk_completed = future.result()
                    completed += int(chunk_completed)
                    for key, sample_values in chunk_values.items():
                        values[key].extend(sample_values)
                    if progress_callback:
                        valid_samples = max((len(sample_values) for sample_values in values.values()), default=0)
                        should_continue = progress_callback(min(completed, subsamples), subsamples, valid_samples)
                        if should_continue is False:
                            for pending in futures:
                                pending.cancel()
                            raise ModelValidationError("Bootstrapping đã bị hủy.")
        finally:
            for var, value in saved_env.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
        return values

    def _flatten_result(self, result: dict[str, Any]) -> dict[str, float]:
        """Flatten the base result into the SmartPLS-style metric set.

        Keys here MUST match those produced by ``_fit_bootstrap_flat_numeric`` so the
        bootstrap distribution lines up with each "Original" value.
        """
        flat: dict[str, float] = {}
        structural: list[tuple[str, str]] = result.get("structural_model", self.structural_model)
        paths: pd.DataFrame = result["path_coefficients"]
        lvs = list(paths.index)
        lv_index = {lv: index for index, lv in enumerate(lvs)}
        path_matrix = paths.loc[lvs, lvs].to_numpy(dtype=float)
        for source, target in structural:
            if source in paths.index and target in paths.columns:
                flat[f"path:{source} -> {target}"] = float(paths.loc[source, target])

        for chain in _indirect_chains(structural, lvs):
            flat[f"specific:{' -> '.join(chain)}"] = _chain_specific(path_matrix, lv_index, chain)

        loadings: pd.DataFrame = result["outer_loadings"]
        measurement: dict[str, list[str]] = result["measurement_model"]
        for construct, indicators in measurement.items():
            for indicator in indicators:
                flat[f"loading:{indicator} <- {construct}"] = float(loadings.loc[indicator, construct])

        weights: pd.DataFrame = result.get("outer_weights")
        if weights is not None:
            for construct, indicators in measurement.items():
                for indicator in indicators:
                    if indicator in weights.index and construct in weights.columns:
                        flat[f"weight:{indicator} <- {construct}"] = float(weights.loc[indicator, construct])

        r2: pd.Series = result["r_square"]
        for construct, value in r2.items():
            flat[f"r2:{construct}"] = float(value)
        adjusted: pd.Series | None = result.get("adjusted_r_square")
        if adjusted is not None:
            for construct, value in adjusted.items():
                flat[f"radj:{construct}"] = float(value)

        f_square: pd.DataFrame | None = result.get("f_square")
        if isinstance(f_square, pd.DataFrame) and not f_square.empty:
            for label, row in f_square.iterrows():
                flat[f"f2:{label}"] = float(row.get("f2", np.nan))

        reliability: pd.DataFrame = result["reliability"]
        rel_metrics = [
            ("Cronbach alpha", "Cronbach alpha"),
            ("Composite reliability", "Composite reliability"),
            ("AVE", "AVE"),
            ("rho_A approx.", "rho_A"),
        ]
        for construct in reliability.index:
            for column, name in rel_metrics:
                if column in reliability.columns:
                    flat[f"{name}:{construct}"] = float(reliability.loc[construct, column])

        modes = result.get("measurement_modes", {})
        eligible = [lv for lv in measurement if modes.get(lv, "reflective") != "formative" and len(measurement[lv]) >= 2]
        htmt: pd.DataFrame | None = result.get("htmt")
        if isinstance(htmt, pd.DataFrame):
            for i in range(len(eligible)):
                for j in range(i + 1, len(eligible)):
                    a, b = eligible[i], eligible[j]
                    if a in htmt.index and b in htmt.columns:
                        flat[f"htmt:{a} <-> {b}"] = float(htmt.loc[a, b])

        lv_correlations: pd.DataFrame | None = result.get("lv_correlations")
        if isinstance(lv_correlations, pd.DataFrame):
            base_lvs = [lv for lv in measurement if lv in lv_correlations.index]
            for i in range(len(base_lvs)):
                for j in range(i + 1, len(base_lvs)):
                    a, b = base_lvs[i], base_lvs[j]
                    flat[f"lvcorr:{a} <-> {b}"] = float(lv_correlations.loc[a, b])

        total_effects: pd.DataFrame = result["total_effects"]
        indirect_effects: pd.DataFrame = result["indirect_effects"]
        for source, target in _total_pairs(structural, lvs):
            if source in total_effects.index and target in total_effects.columns:
                flat[f"total:{source} -> {target}"] = float(total_effects.loc[source, target])
        for source, target in _indirect_pairs(structural, lvs):
            if source in indirect_effects.index and target in indirect_effects.columns:
                flat[f"total_indirect:{source} -> {target}"] = float(indirect_effects.loc[source, target])

        model_fit: pd.DataFrame | None = result.get("model_fit")
        if isinstance(model_fit, pd.DataFrame):
            column = "Estimated Model" if "Estimated Model" in model_fit.columns else model_fit.columns[0]
            for metric in ["SRMR", "d_ULS", "d_G"]:
                if metric in model_fit.index:
                    flat[f"model_fit:{metric}"] = float(model_fit.loc[metric, column])

        return {key: value for key, value in flat.items() if np.isfinite(value)}

    def _make_interpretation(self, result: dict[str, Any]) -> list[str]:
        notes: list[str] = []
        if result.get("converged"):
            notes.append(f"Thuật toán PLS đã hội tụ sau {result.get('iterations')} vòng lặp.")
        else:
            notes.append("Thuật toán PLS chưa hội tụ trong giới hạn vòng lặp đã chọn.")

        reliability: pd.DataFrame | None = result.get("reliability")
        if reliability is not None and not reliability.empty:
            for construct, row in reliability.iterrows():
                ave = row.get("AVE", np.nan)
                cr = row.get("Composite reliability", np.nan)
                if pd.notna(ave):
                    notes.append(
                        f"{construct}: AVE {ave:.3f} "
                        + ("đạt giá trị hội tụ (>= 0.50)." if ave >= 0.5 else "thấp hơn 0.50; nên xem lại biến quan sát.")
                    )
                if pd.notna(cr):
                    notes.append(
                        f"{construct}: composite reliability {cr:.3f} "
                        + ("đạt mức chấp nhận (>= 0.70)." if cr >= 0.7 else "thấp hơn 0.70.")
                    )

        htmt: pd.DataFrame | None = result.get("htmt")
        if htmt is not None and not htmt.empty:
            for row in htmt.index:
                for column in htmt.columns:
                    if row >= column:
                        continue
                    value = htmt.loc[row, column]
                    if pd.notna(value) and value > 0.9:
                        notes.append(f"HTMT giữa {row} và {column} = {value:.3f}; giá trị phân biệt có thể yếu.")

        bootstrap: pd.DataFrame | None = result.get("bootstrap")
        if bootstrap is not None and not bootstrap.empty:
            for metric, row in bootstrap.iterrows():
                if metric.startswith("path:") and pd.notna(row.get("P value")):
                    p_value = float(row["P value"])
                    notes.append(f"{metric[5:]}: p-value bootstrap = {p_value:.4f}.")

        return notes

    def _has_cycle(self) -> bool:
        graph: dict[str, list[str]] = {construct: [] for construct in self.measurement_model}
        for source, target in self.structural_model:
            graph.setdefault(source, []).append(target)
            graph.setdefault(target, [])

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for next_node in graph.get(node, []):
                if visit(next_node):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in graph)

    @staticmethod
    def _initial_weights(count: int) -> np.ndarray:
        if count <= 0:
            return np.array([])
        return np.ones(count, dtype=float) / np.sqrt(count)


def _ols(x: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, float, float, np.ndarray]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim == 1:
        x_values = x_values.reshape(-1, 1)
    design = np.column_stack([np.ones(x_values.shape[0]), x_values])
    beta_full, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    predicted = design @ beta_full
    residual = y_values - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(y_values)
    p = x_values.shape[1]
    adjusted = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else np.nan
    return beta_full[1:], float(r2), float(adjusted), predicted


def _ols_array(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if x_values.ndim == 1:
        x_values = x_values.reshape(-1, 1)
    design = np.column_stack([np.ones(x_values.shape[0]), x_values])
    beta_full, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    return np.asarray(beta_full[1:], dtype=float)


def _ols_r2_array(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, float]:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if x_values.ndim == 1:
        x_values = x_values.reshape(-1, 1)
    design = np.column_stack([np.ones(x_values.shape[0]), x_values])
    beta_full, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    predicted = design @ beta_full
    residual = y_values - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return np.asarray(beta_full[1:], dtype=float), float(r2)


def _normalize(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm == 0 or not np.isfinite(norm):
        return values
    return values / norm


def _standardize_array(values: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1))
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def _corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    a_values = np.asarray(a, dtype=float)
    b_values = np.asarray(b, dtype=float)
    mask = np.isfinite(a_values) & np.isfinite(b_values)
    if mask.sum() < 2:
        return 0.0
    a_values = a_values[mask]
    b_values = b_values[mask]
    if np.nanstd(a_values, ddof=1) == 0 or np.nanstd(b_values, ddof=1) == 0:
        return 0.0
    return float(np.corrcoef(a_values, b_values)[0, 1])


def _corr_array(a_values: np.ndarray, b_values: np.ndarray) -> float:
    a_values = np.asarray(a_values, dtype=float)
    b_values = np.asarray(b_values, dtype=float)
    mask = np.isfinite(a_values) & np.isfinite(b_values)
    if mask.sum() < 2:
        return 0.0
    a_values = a_values[mask]
    b_values = b_values[mask]
    a_centered = a_values - a_values.mean()
    b_centered = b_values - b_values.mean()
    denom = float(np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2)))
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return float((a_centered @ b_centered) / denom)


def _pred_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / MAPE / Q²predict for a holdout prediction (mean = naive benchmark)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "q2": np.nan}
    err = actual - predicted
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    nz = np.abs(actual) > 1e-12
    mape = float(np.mean(np.abs(err[nz] / actual[nz])) * 100.0) if nz.any() else np.nan
    sso = float(np.sum((actual - float(np.mean(actual))) ** 2))
    q2 = float(1.0 - float(np.sum(err ** 2)) / sso) if sso > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "mape": mape, "q2": q2}


def _describe(values: np.ndarray) -> dict[str, float]:
    """SmartPLS-style descriptive statistics for a 1-D array of predictions or errors."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    keys = ["Mean", "Median", "Minimum", "Maximum",
            "Standard Deviation", "Skewness", "Excess Kurtosis", "Observations"]
    if v.size == 0:
        return {k: (0.0 if k == "Observations" else np.nan) for k in keys}
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    if sd > 0 and v.size > 2:
        z = (v - mean) / sd
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)
    else:
        skew = np.nan
        kurt = np.nan
    return {
        "Mean": mean,
        "Median": float(np.median(v)),
        "Minimum": float(np.min(v)),
        "Maximum": float(np.max(v)),
        "Standard Deviation": sd,
        "Skewness": skew,
        "Excess Kurtosis": kurt,
        "Observations": float(v.size),
    }


def _effects_array(path_coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    identity = np.eye(path_coefficients.shape[0])
    try:
        total = np.linalg.inv(identity - path_coefficients) - identity
    except np.linalg.LinAlgError:
        total = np.zeros_like(path_coefficients)
        power = path_coefficients.copy()
        for _ in range(path_coefficients.shape[0]):
            total += power
            power = power @ path_coefficients
    return total, total - path_coefficients


def _indirect_chains(structural: list[tuple[str, str]], lvs: list[str]) -> list[list[str]]:
    """Enumerate every simple directed path with >=1 mediator (>=3 nodes).

    Deterministic order: depth-first from each construct in model order. Each
    distinct mediation chain (e.g. A -> M -> B, A -> M1 -> M2 -> B) appears once.
    """
    adjacency: dict[str, list[str]] = {lv: [] for lv in lvs}
    for source, target in structural:
        if source in adjacency and target in adjacency:
            adjacency[source].append(target)
    chains: list[list[str]] = []

    def walk(path: list[str]) -> None:
        for nxt in adjacency.get(path[-1], []):
            if nxt in path:
                continue  # skip cycles
            extended = path + [nxt]
            if len(extended) >= 3:
                chains.append(extended)
            walk(extended)

    for lv in lvs:
        walk([lv])
    return chains


def _reachability(structural: list[tuple[str, str]], lvs: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Direct adjacency and transitive reachability (>=1 step) over the structural graph."""
    index = {lv: i for i, lv in enumerate(lvs)}
    n = len(lvs)
    adjacency = np.zeros((n, n), dtype=bool)
    for source, target in structural:
        if source in index and target in index:
            adjacency[index[source], index[target]] = True
    reach = adjacency.copy()
    for k in range(n):  # Warshall transitive closure
        reach |= reach[:, k][:, None] & reach[k, :][None, :]
    return adjacency, reach, index


def _total_pairs(structural: list[tuple[str, str]], lvs: list[str]) -> list[tuple[str, str]]:
    """Every ordered pair where the target is reachable from the source (total effect ≠ 0)."""
    _adjacency, reach, _index = _reachability(structural, lvs)
    n = len(lvs)
    return [(lvs[i], lvs[j]) for i in range(n) for j in range(n) if i != j and reach[i, j]]


def _indirect_pairs(structural: list[tuple[str, str]], lvs: list[str]) -> list[tuple[str, str]]:
    """Ordered pairs that have at least one mediated (length >= 2) path."""
    adjacency, reach, _index = _reachability(structural, lvs)
    n = len(lvs)
    pairs: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if any(adjacency[i, m] and m != j and reach[m, j] for m in range(n)):
                pairs.append((lvs[i], lvs[j]))
    return pairs


def _chain_specific(path_matrix: np.ndarray, lv_index: dict[str, int], chain: list[str]) -> float:
    """Specific indirect effect = product of path coefficients along the chain."""
    value = 1.0
    for source, target in zip(chain[:-1], chain[1:]):
        source_i = lv_index.get(source)
        target_i = lv_index.get(target)
        if source_i is None or target_i is None:
            return float("nan")
        value *= float(path_matrix[source_i, target_i])
    return value


def _rho_a_block(weights: np.ndarray, block_corr: np.ndarray) -> float:
    """Dijkstra-Henseler rho_A from outer weights and the indicator correlation block."""
    w = np.asarray(weights, dtype=float)
    k = w.size
    if k <= 1:
        return 1.0
    wsw = float(w @ block_corr @ w)
    if not np.isfinite(wsw) or wsw <= 0:
        return float("nan")
    w = w / np.sqrt(wsw)  # rescale so the composite has unit variance (w'Sw = 1)
    wtw = float(w @ w)
    diag = np.diag(block_corr)
    inner_s = 1.0 - float(np.sum(w**2 * diag))  # w'(S - diagS)w
    denom = wtw**2 - float(np.sum(w**4))  # w'(ww' - diag(ww'))w
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(wtw**2 * inner_s / denom)


def _htmt_from_corr(abs_corr: np.ndarray, idx_a: list[int], idx_b: list[int]) -> float:
    """HTMT for one construct pair from the absolute indicator-correlation matrix."""
    if not idx_a or not idx_b:
        return float("nan")
    hetero = abs_corr[np.ix_(idx_a, idx_b)]
    heterotrait = float(np.mean(hetero))
    mono_a = _offdiag_mean(abs_corr[np.ix_(idx_a, idx_a)])
    mono_b = _offdiag_mean(abs_corr[np.ix_(idx_b, idx_b)])
    denom = np.sqrt(mono_a * mono_b)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return heterotrait / denom


def _offdiag_mean(block: np.ndarray) -> float:
    n = block.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(block[mask]))


def _bias_corrected_bounds(samples: np.ndarray, original: float, alpha: float) -> tuple[float, float]:
    """Bias-corrected percentile bootstrap CI bounds (BC, no acceleration)."""
    samples = samples[np.isfinite(samples)]
    n = samples.size
    if n < 2:
        return float("nan"), float("nan")
    proportion = float(np.mean(samples < original))
    if proportion <= 0:
        proportion = 0.5 / n
    elif proportion >= 1:
        proportion = 1 - 0.5 / n
    z0 = stats.norm.ppf(proportion)
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)
    lower_q = float(stats.norm.cdf(2 * z0 + z_lo))
    upper_q = float(stats.norm.cdf(2 * z0 + z_hi))
    lower = float(np.quantile(samples, min(max(lower_q, 0.0), 1.0)))
    upper = float(np.quantile(samples, min(max(upper_q, 0.0), 1.0)))
    return lower, upper


def _henseler_pls_mga(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    """Henseler's PLS-MGA one-tailed p-value: P(beta_A <= beta_B) from bootstrap distributions.

    Significant difference at 5% when p < 0.05 (B greater) or p > 0.95 (A greater).
    """
    a = np.sort(arr_a[np.isfinite(arr_a)])
    b = arr_b[np.isfinite(arr_b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    # For each bootstrap estimate of B, count how many A estimates are <= it.
    counts = np.searchsorted(a, b, side="right")
    return float(np.mean(counts) / a.size)


def _vec_corr(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of `matrix` with `vector` (vectorized)."""
    centered = matrix - matrix.mean(axis=0)
    target = vector - vector.mean()
    col_norm = np.sqrt(np.sum(centered ** 2, axis=0))
    target_norm = np.sqrt(np.sum(target ** 2))
    denom = col_norm * target_norm
    numerator = centered.T @ target
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom > 0, numerator / denom, 0.0)
    return result


def _upper_triangle_values(frame: pd.DataFrame) -> np.ndarray:
    if frame.shape[0] < 2:
        return np.array([np.nan])
    mask = np.triu(np.ones(frame.shape, dtype=bool), k=1)
    return frame.to_numpy(dtype=float)[mask]
