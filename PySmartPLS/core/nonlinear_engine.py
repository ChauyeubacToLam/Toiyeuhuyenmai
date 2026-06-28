"""Nonlinear ML pipeline engine for PySmartPLS.

Wraps the full "phi tuyến tính" (nonlinear) workflow that previously lived in a
Jupyter notebook into a clean, thread-callable engine:

    1. XGBoost regression with optional GridSearchCV (R²/RMSE/MAE on Train/CV/Test)
    2. SHAP explainability (beeswarm, %importance, dependence scatter + LOWESS)
    3. Symbolic Regression (PySR) -> human-readable formula + Pareto front
    4. Sobol global sensitivity (SALib) + convergence + directional impact
    5. Optimization of the discovered model (SciPy differential evolution / brute force)

Heavy scientific libraries (xgboost, shap, pysr, scikit-learn, SALib, sympy,
matplotlib, statsmodels) are imported lazily so the desktop app always launches.
On this machine they live in the *global* CPython 3.13 site-packages while the
app runs from a venv of the same interpreter, so :func:`ensure_ml_libs` bridges
``sys.path`` (ABI-safe — identical CPython) before importing them.

Charts are rendered to PNG bytes with the Agg backend (no pyplot global state),
which is safe to do inside a worker thread; the GUI just shows the pixmaps.
"""
from __future__ import annotations

import io
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Dependency bridge
# --------------------------------------------------------------------------- #
ML_LIBS = ["sklearn", "xgboost", "shap", "SALib", "sympy", "matplotlib", "statsmodels", "pysr"]

_BRIDGED = False


def _candidate_site_packages() -> list[str]:
    """Best-effort locations of a same-version global CPython site-packages."""
    minor = sys.version_info.minor
    tag = f"Python3{minor}"  # e.g. Python313
    candidates: list[str] = []
    override = os.environ.get("PYSMARTPLS_ML_SITE")
    if override:
        candidates.append(override)
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        candidates.append(os.path.join(local, "Programs", "Python", tag, "Lib", "site-packages"))
    program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    candidates.append(os.path.join(program_files, tag, "Lib", "site-packages"))
    candidates.append(os.path.join("C:\\", tag, "Lib", "site-packages"))
    return candidates


def ensure_ml_libs() -> None:
    """Add a global site-packages dir to ``sys.path`` if the ML libs are missing.

    Idempotent and cheap after the first successful bridge. Never raises.
    """
    global _BRIDGED
    if _BRIDGED:
        return
    import importlib.util

    if importlib.util.find_spec("xgboost") is not None:
        _BRIDGED = True
        return
    for path in _candidate_site_packages():
        try:
            if path and os.path.isdir(path) and os.path.isdir(os.path.join(path, "xgboost")):
                if path not in sys.path:
                    sys.path.append(path)
                break
        except Exception:
            continue
    _BRIDGED = True


def dependency_report() -> dict[str, dict[str, Any]]:
    """Return ``{lib: {available, version}}`` for every heavy dependency."""
    ensure_ml_libs()
    import importlib

    report: dict[str, dict[str, Any]] = {}
    for name in ML_LIBS:
        try:
            module = importlib.import_module(name)
            report[name] = {"available": True, "version": getattr(module, "__version__", "?")}
        except Exception as exc:  # pragma: no cover - import errors are environment specific
            report[name] = {"available": False, "version": "", "error": str(exc)[:160]}
    return report


def core_ready() -> bool:
    """True when the non-PySR core (XGBoost + SHAP + sklearn) can run."""
    report = dependency_report()
    return all(report[name]["available"] for name in ("sklearn", "xgboost", "shap", "matplotlib"))


# --------------------------------------------------------------------------- #
# Premium chart styling (applied per-figure, Agg backend)
# --------------------------------------------------------------------------- #
ACCENT = "#7C5CFC"          # nonlinear signature (violet)
ACCENT_SOFT = "#B9A8FF"
BLUE = "#1D6FE0"
INK = "#1B2433"
SUBINK = "#5B6776"
GRID = "#E6EAF2"
RED = "#E11D48"


def _new_figure(width: float, height: float, dpi: int = 130):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(width, height), dpi=dpi)
    fig.set_facecolor("#FFFFFF")
    FigureCanvasAgg(fig)
    return fig


def _style_axes(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D5DCE8")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=SUBINK, labelsize=8.5, length=0)
    ax.grid(True, color=GRID, linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(INK)
        label.set_fontsize(9.5)
    if ax.get_title():
        ax.title.set_color(INK)
        ax.title.set_fontsize(11)
        ax.title.set_fontweight("bold")


def _fig_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Defaults derived from the notebook
# --------------------------------------------------------------------------- #
DEFAULT_PARAM_GRID: dict[str, list] = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 400],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
}

QUICK_PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
}


def _xgb_thread_count() -> int:
    """Use most local cores while leaving the UI responsive."""
    cpu_count = os.cpu_count() or 2
    default_threads = min(8, max(1, cpu_count - 1))
    raw = os.environ.get("PYSMARTPLS_XGB_THREADS", str(default_threads))
    try:
        requested = int(raw)
    except ValueError:
        requested = default_threads
    return max(1, min(requested, cpu_count))


def _xgb_grid_jobs() -> int:
    """Parallel GridSearchCV workers; each worker runs one single-thread XGB fit."""
    cpu_count = os.cpu_count() or 2
    default_jobs = min(8, max(1, cpu_count - 1))
    raw = os.environ.get("PYSMARTPLS_XGB_GRID_JOBS", str(default_jobs))
    try:
        requested = int(raw)
    except ValueError:
        requested = default_jobs
    if requested <= 0:
        return max(1, cpu_count - 1)
    return max(1, min(requested, cpu_count))


@dataclass
class NonlinearState:
    features: list[str]
    target: str
    display_names: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class NonlinearEngine:
    """Holds the dataset + intermediate artefacts across pipeline stages."""

    def __init__(
        self,
        frame: pd.DataFrame,
        features: list[str],
        target: str,
        display_names: dict[str, str] | None = None,
    ) -> None:
        ensure_ml_libs()
        from core.data_manager import coerce_numeric_frame

        self.features = list(features)
        self.target = target
        self.display_names = dict(display_names or {})

        used = [c for c in (self.features + [self.target]) if c in frame.columns]
        numeric, _warn = coerce_numeric_frame(frame[used])
        numeric = numeric.dropna()
        self.frame = numeric.reset_index(drop=True)
        self.X = self.frame[self.features]
        self.y = self.frame[self.target]

        # Filled by stages.
        self.model = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.best_params: dict[str, Any] = {}
        self.shap_values = None
        self.pysr_model = None
        self.formula_sympy = None
        self.formula_callable: Callable[[np.ndarray], np.ndarray] | None = None

    # ---- helpers ---------------------------------------------------------- #
    def label(self, col: str) -> str:
        return self.display_names.get(col, col)

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    def bounds(self) -> list[tuple[float, float]]:
        return [(float(self.X[c].min()), float(self.X[c].max())) for c in self.features]

    # ---- stage 1: XGBoost ------------------------------------------------- #
    def train_xgboost(
        self,
        *,
        use_grid: bool = True,
        param_grid: dict[str, list] | None = None,
        params: dict[str, Any] | None = None,
        cv_folds: int = 5,
        test_size: float = 0.20,
        seed: int = 42,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        import xgboost as xgb
        from joblib import parallel_backend
        from sklearn.model_selection import train_test_split, GridSearchCV, KFold
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        def say(msg: str) -> None:
            if progress:
                progress(msg)

        say("Đang chia dữ liệu train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=seed
        )
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        xgb_threads = _xgb_thread_count()
        base = xgb.XGBRegressor(
            random_state=seed,
            objective="reg:squarederror",
            n_jobs=1,
            tree_method="hist",
        )

        if use_grid:
            grid = {k: list(v) for k, v in (param_grid or DEFAULT_PARAM_GRID).items()}
            n_candidates = int(np.prod([len(v) for v in grid.values()]))
            grid_jobs = _xgb_grid_jobs()
            say(f"GridSearchCV dang chay song song voi {grid_jobs} worker.")
            say(f"Đang quét lưới {n_candidates} tổ hợp × {cv_folds}-fold...")
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            scoring = {
                "R2": "r2",
                "neg_RMSE": "neg_root_mean_squared_error",
                "neg_MAE": "neg_mean_absolute_error",
            }
            search = GridSearchCV(
                estimator=base, param_grid=grid, cv=cv, scoring=scoring,
                refit="R2", n_jobs=grid_jobs, pre_dispatch=grid_jobs, verbose=0,
            )
            with parallel_backend("threading", n_jobs=grid_jobs):
                search.fit(X_train, y_train)
            self.model = search.best_estimator_
            self.model.set_params(n_jobs=xgb_threads)
            self.best_params = dict(search.best_params_)
            idx = search.best_index_
            cv_r2 = float(search.cv_results_["mean_test_R2"][idx])
            cv_rmse = float(-search.cv_results_["mean_test_neg_RMSE"][idx])
            cv_mae = float(-search.cv_results_["mean_test_neg_MAE"][idx])
            n_fits = n_candidates * cv_folds
        else:
            merged = dict(QUICK_PARAMS)
            merged.update(params or {})
            say("Đang huấn luyện XGBoost (chế độ nhanh)...")
            self.model = xgb.XGBRegressor(
                random_state=seed,
                objective="reg:squarederror",
                n_jobs=xgb_threads,
                tree_method="hist",
                **merged,
            )
            self.model.fit(X_train, y_train)
            self.best_params = merged
            # Cross-val metrics for honesty even in quick mode.
            from sklearn.model_selection import cross_validate
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            cv_jobs = _xgb_grid_jobs()
            cv_model = xgb.XGBRegressor(
                random_state=seed,
                objective="reg:squarederror",
                n_jobs=1,
                tree_method="hist",
                **merged,
            )
            with parallel_backend("threading", n_jobs=cv_jobs):
                cv_scores = cross_validate(
                    cv_model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring={
                        "R2": "r2",
                        "neg_RMSE": "neg_root_mean_squared_error",
                        "neg_MAE": "neg_mean_absolute_error",
                    },
                    n_jobs=cv_jobs,
                    pre_dispatch=cv_jobs,
                )
            cv_r2 = float(np.mean(cv_scores["test_R2"]))
            cv_rmse = float(-np.mean(cv_scores["test_neg_RMSE"]))
            cv_mae = float(-np.mean(cv_scores["test_neg_MAE"]))
            n_candidates = 1
            n_fits = cv_folds

        say("Đang đánh giá trên tập kiểm tra...")
        pred_train = self.model.predict(X_train)
        pred_test = self.model.predict(X_test)
        train_r2 = float(r2_score(y_train, pred_train))
        test_r2 = float(r2_score(y_test, pred_test))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
        test_mae = float(mean_absolute_error(y_test, pred_test))

        metrics = {
            "train_r2": train_r2,
            "cv_r2": cv_r2, "cv_rmse": cv_rmse, "cv_mae": cv_mae,
            "test_r2": test_r2, "test_rmse": test_rmse, "test_mae": test_mae,
        }

        importance = self._gain_importance()
        figures = {
            "pred_vs_actual": self._fig_pred_vs_actual(y_test, pred_test, test_r2),
            "importance_gain": self._fig_importance_bar(importance, "Độ quan trọng (Gain)"),
        }
        return {
            "best_params": self.best_params,
            "metrics": metrics,
            "n_candidates": n_candidates,
            "n_fits": n_fits,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "importance": importance,
            "figures": figures,
        }

    def _gain_importance(self) -> list[tuple[str, float]]:
        booster = self.model.get_booster()
        score = booster.get_score(importance_type="gain")
        # XGBoost names features f0, f1, ... in column order.
        mapping = {f"f{i}": self.features[i] for i in range(len(self.features))}
        values = {mapping.get(k, k): v for k, v in score.items()}
        for feat in self.features:
            values.setdefault(feat, 0.0)
        total = sum(values.values()) or 1.0
        pairs = [(feat, 100.0 * values[feat] / total) for feat in self.features]
        pairs.sort(key=lambda kv: kv[1])
        return pairs

    # ---- stage 2: SHAP ---------------------------------------------------- #
    def compute_shap(self, *, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        import shap

        if self.model is None:
            raise RuntimeError("Hãy huấn luyện XGBoost trước khi chạy SHAP.")
        if progress:
            progress("Đang tính giá trị SHAP (TreeExplainer)...")
        explainer = shap.TreeExplainer(self.model)
        explanation = explainer(self.X)
        self.shap_values = explanation
        shap_matrix = np.asarray(explanation.values)

        mean_abs = np.abs(shap_matrix).mean(axis=0)
        total = mean_abs.sum() or 1.0
        importance = [(self.label(self.features[i]), 100.0 * mean_abs[i] / total)
                      for i in range(len(self.features))]
        importance.sort(key=lambda kv: kv[1])

        if progress:
            progress("Đang dựng biểu đồ beeswarm & scatter...")
        figures = {
            "beeswarm": self._fig_beeswarm(shap_matrix),
            "importance_bar": self._fig_importance_bar(importance, "Độ quan trọng SHAP (%)", pct=True),
            "scatter_grid": self._fig_shap_scatter(shap_matrix),
        }
        return {"importance": importance, "figures": figures}

    # ---- stage 3: Symbolic Regression (PySR) ------------------------------ #
    def run_symbolic_regression(
        self,
        *,
        niterations: int = 200,
        maxsize: int = 30,
        populations: int = 15,
        parsimony: float = 0.0,
        seed: int = 42,
        use_train: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        import sympy
        from pysr import PySRRegressor
        from sklearn.metrics import r2_score

        if self.X_train is None:
            raise RuntimeError("Hãy huấn luyện XGBoost trước (cần tập train) để chạy hồi quy biểu tượng.")
        if progress:
            progress("Đang tiến hoá phương trình với PySR (Julia)...")

        X_src = self.X_train if use_train else self.X
        y_src = self.y_train if use_train else self.y
        X_np = np.asarray(X_src, dtype=np.float32)
        y_np = np.asarray(y_src, dtype=np.float32)

        model = PySRRegressor(
            niterations=niterations,
            maxsize=maxsize,
            populations=populations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[
                "square", "cube", "sqrt_abs(x) = sqrt(abs(x))",
                "log_abs(x) = log(abs(x) + 1f-8)", "exp",
            ],
            extra_sympy_mappings={
                "sqrt_abs": lambda x: sympy.sqrt(sympy.Abs(x)),
                "log_abs": lambda x: sympy.log(sympy.Abs(x) + 1e-8),
            },
            model_selection="accuracy",
            parsimony=parsimony,
            random_state=seed,
            deterministic=True,
            parallelism="serial",
            verbosity=0,
            progress=False,
        )
        model.fit(X_np, y_np, variable_names=self.features)
        self.pysr_model = model
        self.formula_sympy = model.sympy()

        def predictor(matrix: np.ndarray) -> np.ndarray:
            return model.predict(np.asarray(matrix, dtype=np.float32))

        self.formula_callable = predictor
        r2 = float(r2_score(y_np, model.predict(X_np)))

        table = []
        try:
            eqs = model.equations_
            for _, row in eqs.iterrows():
                table.append({
                    "complexity": int(row["complexity"]),
                    "loss": float(row["loss"]),
                    "score": float(row.get("score", float("nan"))),
                    "equation": str(row["equation"]),
                })
        except Exception:
            pass

        figures = {}
        if table:
            figures["pareto"] = self._fig_pareto(table)
        return {
            "equation": str(self.formula_sympy),
            "latex": self._safe_latex(self.formula_sympy),
            "r2": r2,
            "table": table,
            "figures": figures,
        }

    @staticmethod
    def _safe_latex(expr) -> str:
        try:
            import sympy
            return sympy.latex(expr)
        except Exception:
            return ""

    # ---- stage 4: Sobol sensitivity --------------------------------------- #
    def sobol_sensitivity(
        self,
        *,
        predictor: str = "model",
        max_pow: int = 13,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        from SALib.sample import saltelli
        from SALib.analyze import sobol

        predict = self._resolve_predictor(predictor)
        names = self.features
        # SALib rejects equal lo/hi bounds; widen any constant feature slightly so a
        # constant column yields ~zero sensitivity instead of crashing the stage.
        safe_bounds = [[lo, hi if hi > lo else lo + 1e-6] for lo, hi in self.bounds()]
        problem = {"num_vars": len(names), "names": names, "bounds": safe_bounds}

        powers = [p for p in (5, 7, 9, 11, 13) if p <= max_pow]
        if not powers:
            powers = [max_pow]
        results_s1 = {n: [] for n in names}
        results_st = {n: [] for n in names}
        conf_s1 = {n: [] for n in names}
        conf_st = {n: [] for n in names}

        final = None
        for power in powers:
            if progress:
                progress(f"Đang lấy mẫu Sobol N = 2^{power}...")
            samples = saltelli.sample(problem, 2 ** power)
            y_pred = predict(samples)
            si = sobol.analyze(problem, np.asarray(y_pred, dtype=float), print_to_console=False)
            for i, name in enumerate(names):
                results_s1[name].append(float(si["S1"][i]))
                results_st[name].append(float(si["ST"][i]))
                conf_s1[name].append(float(si["S1_conf"][i]))
                conf_st[name].append(float(si["ST_conf"][i]))
            final = si

        table = [
            {"feature": self.label(name), "S1": float(final["S1"][i]), "ST": float(final["ST"][i])}
            for i, name in enumerate(names)
        ]
        directional = self._directional_impact(predict)
        figures = {
            "convergence": self._fig_sobol_convergence(powers, names, results_s1, results_st, conf_s1, conf_st),
            "directional": directional["figure"],
        }
        return {
            "table": table,
            "powers": powers,
            "directional_table": directional["table"],
            "figures": figures,
        }

    def _directional_impact(self, predict) -> dict[str, Any]:
        names = self.features
        bounds = self.bounds()
        base = np.array([(lo + hi) / 2.0 for lo, hi in bounds])
        num_points = 100
        rows = []
        curves = {}
        for i, name in enumerate(names):
            lo, hi = bounds[i]
            if hi <= lo:  # constant feature → no slope to classify
                grid = np.array([lo, lo])
                matrix = np.tile(base, (2, 1))
                matrix[:, i] = grid
                y_eval = np.asarray(predict(matrix), dtype=float)
                rows.append({
                    "feature": self.label(name), "dominant": "Không đổi",
                    "Tăng tốc": 0.0, "Tăng chậm dần": 0.0,
                    "Giảm tăng tốc": 0.0, "Giảm chậm dần": 0.0, "Không đổi": 100.0,
                })
                curves[name] = (grid, y_eval)
                continue
            grid = np.linspace(lo, hi, num_points)
            matrix = np.tile(base, (num_points, 1))
            matrix[:, i] = grid
            y_eval = np.asarray(predict(matrix), dtype=float)
            dx = grid[1] - grid[0]
            dy = np.gradient(y_eval, dx)
            d2y = np.gradient(dy, dx)
            # A flat tolerance keeps step-function (tree) predictors from being
            # mislabelled: their many exact-zero slopes count as "no change".
            span = float(np.ptp(y_eval))
            tol = max(span / (hi - lo) * 1e-3, 1e-9)
            flat = np.abs(dy) <= tol
            trends = {
                "Tăng tốc": float(np.sum((dy > tol) & (d2y > 0)) / num_points * 100),
                "Tăng chậm dần": float(np.sum((dy > tol) & (d2y <= 0)) / num_points * 100),
                "Giảm tăng tốc": float(np.sum((dy < -tol) & (d2y < 0)) / num_points * 100),
                "Giảm chậm dần": float(np.sum((dy < -tol) & (d2y >= 0)) / num_points * 100),
                "Không đổi": float(np.sum(flat) / num_points * 100),
            }
            dominant = max(trends, key=trends.get)
            rows.append({"feature": self.label(name), "dominant": dominant, **trends})
            curves[name] = (grid, y_eval)
        figure = self._fig_directional(curves)
        return {"table": rows, "figure": figure}

    def _resolve_predictor(self, which: str):
        if which == "formula":
            if self.formula_callable is None:
                raise RuntimeError("Chưa có công thức PySR. Hãy chạy Hồi quy biểu tượng trước.")
            return self.formula_callable
        if self.model is None:
            raise RuntimeError("Chưa có mô hình XGBoost. Hãy huấn luyện trước.")
        return lambda matrix: self.model.predict(np.asarray(matrix, dtype=np.float32))

    # ---- stage 5: Optimization -------------------------------------------- #
    def optimize(
        self,
        *,
        predictor: str = "formula",
        integer: bool = True,
        bounds: list[tuple[float, float]] | None = None,
        seed: int = 42,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        from scipy.optimize import differential_evolution

        predict = self._resolve_predictor(predictor)
        raw_box = bounds or self.bounds()

        # Build per-feature integrality. SciPy's differential_evolution rejects an
        # integral dimension whose [lo, hi] spans no integer (ceil(lo) > floor(hi)),
        # so only mark a feature integral when its observed values are integer-valued
        # AND its range actually admits an integer; otherwise keep it continuous.
        box: list[tuple[float, float]] = []
        integrality: list[bool] = []
        for i, (lo, hi) in enumerate(raw_box):
            if hi <= lo:  # constant/degenerate feature → pin via a tiny window
                hi = lo + 1e-6
            col = np.asarray(self.X.iloc[:, i], dtype=float)
            col = col[~np.isnan(col)]
            int_col = integer and col.size > 0 and np.allclose(col, np.round(col))
            ilo, ihi = math.ceil(lo), math.floor(hi)
            if int_col and ilo <= ihi:
                box.append((float(ilo), float(ihi)))
                integrality.append(True)
            else:
                box.append((float(lo), float(hi)))
                integrality.append(False)

        def scalar(x: np.ndarray) -> float:
            return float(np.asarray(predict(np.asarray(x, dtype=float).reshape(1, -1)), dtype=float)[0])

        if progress:
            progress("Đang tối ưu hoá (Differential Evolution) — tìm GIÁ TRỊ NHỎ NHẤT...")
        res_min = differential_evolution(
            scalar, box, integrality=integrality, popsize=20,
            mutation=(0.5, 1.5), recombination=0.7, seed=seed,
        )
        if progress:
            progress("Đang tối ưu hoá — tìm GIÁ TRỊ LỚN NHẤT...")
        res_max = differential_evolution(
            lambda x: -scalar(x), box, integrality=integrality, popsize=20,
            mutation=(0.5, 1.5), recombination=0.7, seed=seed,
        )

        min_vars = [int(round(v)) if integrality[i] else round(float(v), 4)
                    for i, v in enumerate(res_min.x)]
        max_vars = [int(round(v)) if integrality[i] else round(float(v), 4)
                    for i, v in enumerate(res_max.x)]

        result = {
            "min_value": float(res_min.fun),
            "max_value": float(-res_max.fun),
            "min_vars": dict(zip(self.features, min_vars)),
            "max_vars": dict(zip(self.features, max_vars)),
            "method": "Differential Evolution (SciPy)",
            "figures": {
                "profile": self._fig_optimum_profile(
                    dict(zip(self.features, min_vars)), dict(zip(self.features, max_vars))
                )
            },
        }
        return result

    # ================================================================== #
    # Figures
    # ================================================================== #
    def _fig_pred_vs_actual(self, y_true, y_pred, r2: float) -> bytes:
        fig = _new_figure(5.4, 4.4)
        ax = fig.add_subplot(111)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ax.scatter(y_true, y_pred, s=34, color=ACCENT, alpha=0.55, edgecolor="white", linewidth=0.6)
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], color=SUBINK, linestyle="--", linewidth=1.4)
        ax.set_xlabel("Giá trị thực tế")
        ax.set_ylabel("Giá trị dự đoán")
        ax.set_title(f"Dự đoán vs Thực tế  ·  R² = {r2:.3f}")
        _style_axes(ax)
        return _fig_to_png(fig)

    def _fig_importance_bar(self, importance: list[tuple[str, float]], title: str, *, pct: bool = False) -> bytes:
        labels = [self.label(name) for name, _ in importance]
        values = [v for _, v in importance]
        height = max(2.6, 0.55 * len(labels) + 1.4)
        fig = _new_figure(5.6, height)
        ax = fig.add_subplot(111)
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color=ACCENT, height=0.66, edgecolor="white", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        span = (max(values) if values else 1.0) or 1.0
        for i, v in enumerate(values):
            ax.text(v + span * 0.012, i, f"{v:.2f}%" if pct else f"{v:.1f}",
                    va="center", fontsize=8.6, color=INK)
        ax.set_xlim(0, span * 1.18)
        ax.set_title(title)
        _style_axes(ax)
        ax.grid(axis="y", visible=False)
        return _fig_to_png(fig)

    def _fig_beeswarm(self, shap_matrix: np.ndarray) -> bytes:
        import matplotlib
        order = np.argsort(np.abs(shap_matrix).mean(axis=0))
        names = [self.label(self.features[i]) for i in order]
        cmap = matplotlib.colormaps["coolwarm"]
        fig = _new_figure(6.2, max(2.8, 0.6 * len(order) + 1.6))
        ax = fig.add_subplot(111)
        rng = np.random.default_rng(7)
        for row, feat_idx in enumerate(order):
            vals = shap_matrix[:, feat_idx]
            raw = np.asarray(self.X.iloc[:, feat_idx], dtype=float)
            vmin, vmax = np.nanmin(raw), np.nanmax(raw)
            norm = (raw - vmin) / (vmax - vmin) if vmax > vmin else np.full_like(raw, 0.5)
            jitter = (rng.random(len(vals)) - 0.5) * 0.6
            ax.scatter(vals, np.full(len(vals), row) + jitter, c=norm, cmap=cmap,
                       s=15, alpha=0.7, edgecolor="none")
        ax.axvline(0, color=SUBINK, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Giá trị SHAP (tác động lên dự đoán)")
        ax.set_title("SHAP beeswarm")
        _style_axes(ax)
        ax.grid(axis="y", visible=False)
        mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Thấp", "Cao"])
        cbar.ax.tick_params(length=0, labelsize=8, colors=SUBINK)
        cbar.outline.set_visible(False)
        cbar.set_label("Giá trị biến", color=SUBINK, fontsize=8.5)
        return _fig_to_png(fig)

    def _fig_shap_scatter(self, shap_matrix: np.ndarray) -> bytes:
        try:
            import statsmodels.api as sm
        except Exception:  # LOWESS overlay is optional — degrade gracefully
            sm = None
        n = len(self.features)
        cols = 2 if n > 1 else 1
        rows = math.ceil(n / cols)
        fig = _new_figure(6.4, 3.0 * rows + 0.4)
        for i, feat in enumerate(self.features):
            ax = fig.add_subplot(rows, cols, i + 1)
            x_vals = np.asarray(self.X.iloc[:, i], dtype=float)
            y_vals = shap_matrix[:, i]
            ax.scatter(x_vals, y_vals, s=20, color=BLUE, alpha=0.5, edgecolor="none")
            valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
            if sm is not None and valid.sum() > 3:
                lowess = sm.nonparametric.lowess(y_vals[valid], x_vals[valid], frac=0.4)
                ax.plot(lowess[:, 0], lowess[:, 1], color=RED, linewidth=2.2)
            ax.axhline(0, color=SUBINK, linewidth=1.0, linestyle="--", alpha=0.6)
            ax.set_title(self.label(feat), fontsize=10)
            ax.set_xlabel("Giá trị biến")
            ax.set_ylabel("SHAP")
            _style_axes(ax)
        fig.tight_layout(pad=1.1)
        return _fig_to_png(fig)

    def _fig_pareto(self, table: list[dict]) -> bytes:
        comp = [r["complexity"] for r in table]
        loss = [r["loss"] for r in table]
        fig = _new_figure(5.6, 4.0)
        ax = fig.add_subplot(111)
        ax.plot(comp, loss, marker="o", color=ACCENT, linewidth=2.0, markersize=6,
                markerfacecolor="white", markeredgecolor=ACCENT, markeredgewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlabel("Độ phức tạp")
        ax.set_ylabel("Sai số (loss, log)")
        ax.set_title("Mặt trận Pareto — độ phức tạp vs sai số")
        _style_axes(ax)
        return _fig_to_png(fig)

    def _fig_sobol_convergence(self, powers, names, s1, st, c_s1, c_st) -> bytes:
        fig = _new_figure(7.2, 3.6)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        x_pos = np.arange(len(names))
        palette = [ACCENT, BLUE, "#0E9F6E", "#D98A00", "#E11D48", "#7d3cff", "#00A6A6"]
        markers = ["D", "v", "^", ">", "<", "o", "s"]
        for j, power in enumerate(powers):
            color = palette[j % len(palette)]
            marker = markers[j % len(markers)]
            offset = (j - len(powers) / 2.0) * 0.12
            s1_vals = [s1[n][j] for n in names]
            s1_err = [c_s1[n][j] for n in names]
            st_vals = [st[n][j] for n in names]
            st_err = [c_st[n][j] for n in names]
            ax1.errorbar(x_pos + offset, s1_vals, yerr=s1_err, fmt=marker, color=color,
                         label=f"N=2^{power}", capsize=2.5, markersize=5, alpha=0.85)
            ax2.errorbar(x_pos + offset, st_vals, yerr=st_err, fmt=marker, color=color,
                         label=f"N=2^{power}", capsize=2.5, markersize=5, alpha=0.85)
        labels = [self.label(n) for n in names]
        for ax, title in ((ax1, "(a) Bậc một  S1"), (ax2, "(b) Tổng  ST")):
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
            ax.set_title(title)
            _style_axes(ax)
        ax2.legend(fontsize=7, frameon=False, loc="best")
        fig.tight_layout(pad=1.0)
        return _fig_to_png(fig)

    def _fig_directional(self, curves: dict) -> bytes:
        names = list(curves.keys())
        cols = 3 if len(names) > 2 else len(names)
        cols = max(1, cols)
        rows = math.ceil(len(names) / cols)
        fig = _new_figure(6.8, 2.6 * rows + 0.4)
        for i, name in enumerate(names):
            ax = fig.add_subplot(rows, cols, i + 1)
            grid, y_eval = curves[name]
            ax.plot(grid, y_eval, color=ACCENT, linewidth=2.2)
            ax.set_title(self.label(name), fontsize=9.5)
            ax.set_xlabel("Giá trị")
            ax.set_ylabel("Y dự đoán")
            _style_axes(ax)
        fig.tight_layout(pad=1.0)
        return _fig_to_png(fig)

    def _fig_optimum_profile(self, min_vars: dict, max_vars: dict) -> bytes:
        names = list(self.features)
        labels = [self.label(n) for n in names]
        fig = _new_figure(6.0, max(2.8, 0.5 * len(names) + 1.6))
        ax = fig.add_subplot(111)
        y_pos = np.arange(len(names))
        width = 0.4
        ax.barh(y_pos - width / 2, [max_vars[n] for n in names], height=width,
                color=ACCENT, label="Tối đa Y", edgecolor="white", linewidth=0.7)
        ax.barh(y_pos + width / 2, [min_vars[n] for n in names], height=width,
                color="#9AA6B5", label="Tối thiểu Y", edgecolor="white", linewidth=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Mức biến tại điểm tối ưu")
        ax.set_title("Cấu hình biến tại điểm tối ưu")
        _style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=8, frameon=False)
        return _fig_to_png(fig)
