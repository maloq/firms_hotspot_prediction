from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


EPS = 1e-12


def sigmoid(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def logit(prob: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


@dataclass
class CalibrationMetadata:
    model_name: str
    model_type: str
    calibrator_type: str
    calibration_start_date: str | None
    calibration_end_date: str | None
    test_start_date: str | None
    test_end_date: str | None
    feature_columns_used_by_calibrator: list[str]
    n_rows_unweighted: int
    n_rows_weighted: float
    unweighted_prevalence: float | None
    weighted_prevalence: float | None
    raw_score_mean: float | None
    prob_raw_mean: float | None
    calibrator_path: str


class ConstantPrevalenceCalibrator:
    def __init__(self, prevalence: float, method: str) -> None:
        self.prevalence = float(np.clip(prevalence, EPS, 1.0 - EPS))
        self.method = method
        self.feature_names_: list[str] = ["constant_prevalence"]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.prevalence, dtype=float)


class PlattCalibrator:
    def __init__(self, method: str = "platt_month") -> None:
        self.method = method
        self.model: LogisticRegression | None = None
        self.feature_names_: list[str] = []

    def _design_matrix(self, frame: pd.DataFrame, *, fit: bool = False) -> pd.DataFrame:
        if "raw_score" not in frame.columns:
            if "prob_raw" not in frame.columns:
                raise KeyError("Calibration frame must contain raw_score or prob_raw.")
            raw_score = logit(frame["prob_raw"])
        else:
            raw_score = pd.to_numeric(frame["raw_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        design = pd.DataFrame({"raw_score": raw_score})
        method = str(self.method).lower()
        if method in {"platt_month", "platt_month_country"} and "month" in frame.columns:
            month = pd.to_numeric(frame["month"], errors="coerce").fillna(-1).astype(int).astype(str)
            design = pd.concat([design, pd.get_dummies(month, prefix="month", dtype=float)], axis=1)
        if method == "platt_month_country" and "country" in frame.columns:
            country = frame["country"].fillna("missing").astype(str)
            design = pd.concat([design, pd.get_dummies(country, prefix="country", dtype=float)], axis=1)

        if fit:
            self.feature_names_ = list(design.columns)
        else:
            for col in self.feature_names_:
                if col not in design.columns:
                    design[col] = 0.0
            design = design.reindex(columns=self.feature_names_, fill_value=0.0)
        return design.astype(float)

    def fit(self, frame: pd.DataFrame) -> "PlattCalibrator | ConstantPrevalenceCalibrator":
        y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
        w = (
            pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            if "eval_weight" in frame.columns
            else np.ones(len(frame), dtype=float)
        )
        finite = np.isfinite(y) & np.isfinite(w) & (w > 0)
        y = y[finite]
        w = w[finite]
        work = frame.loc[finite].reset_index(drop=True)
        if len(y) == 0:
            return ConstantPrevalenceCalibrator(0.0, self.method)
        weighted_prev = float(np.average(y, weights=w))
        if np.unique(y).size < 2:
            return ConstantPrevalenceCalibrator(weighted_prev, self.method)

        X = self._design_matrix(work, fit=True)
        try:
            model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            model.fit(X, y, sample_weight=w)
        except TypeError:
            model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
            model.fit(X, y, sample_weight=w)
        self.model = model
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator is not fitted.")
        X = self._design_matrix(frame, fit=False)
        return np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)


class PriorOffsetCalibrator:
    """Rare-event calibrator with raw-score slope fixed to one.

    The fitted offset is the intercept shift that makes weighted expected
    counts match weighted observed counts on the calibration grid. This is more
    stable than an unconstrained Platt slope when deployment positives are very
    rare.
    """

    def __init__(self, method: str = "prior_offset_month") -> None:
        self.method = method
        self.global_offset_: float | None = None
        self.offsets_: dict[Any, float] = {}
        self.group_columns_: list[str] = []
        self.feature_names_: list[str] = []

    def _group_columns(self, frame: pd.DataFrame) -> list[str]:
        method = str(self.method).lower()
        cols: list[str] = []
        if method in {"prior_offset_month", "prior_offset_month_country"} and "month" in frame.columns:
            cols.append("month")
        if method == "prior_offset_month_country" and "country" in frame.columns:
            cols.append("country")
        return cols

    @staticmethod
    def _offset_for_expected_count(raw_score: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
        observed = float(np.sum(y * weight))
        if observed <= 0:
            prevalence = EPS
        else:
            prevalence = min(max(observed / float(np.sum(weight)), EPS), 1.0 - EPS)
        # If there are no positives, match a tiny prevalence rather than
        # returning an infinite offset.
        target = prevalence * float(np.sum(weight)) if observed <= 0 else observed

        lo, hi = -80.0, 80.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            expected = float(np.sum(sigmoid(raw_score + mid) * weight))
            if expected > target:
                hi = mid
            else:
                lo = mid
        return float((lo + hi) / 2.0)

    @staticmethod
    def _group_key(values: tuple[Any, ...] | Any) -> Any:
        if isinstance(values, tuple):
            return tuple(str(value) for value in values)
        return str(values)

    def fit(self, frame: pd.DataFrame) -> "PriorOffsetCalibrator":
        if "raw_score" not in frame.columns:
            if "prob_raw" not in frame.columns:
                raise KeyError("Calibration frame must contain raw_score or prob_raw.")
            raw_all = logit(frame["prob_raw"])
        else:
            raw_all = pd.to_numeric(frame["raw_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_all = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
        w_all = (
            pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            if "eval_weight" in frame.columns
            else np.ones(len(frame), dtype=float)
        )
        finite = np.isfinite(raw_all) & np.isfinite(y_all) & np.isfinite(w_all) & (w_all > 0)
        work = frame.loc[finite].reset_index(drop=True)
        raw = raw_all[finite]
        y = y_all[finite]
        w = w_all[finite]
        if len(work) == 0:
            self.global_offset_ = 0.0
            self.feature_names_ = ["raw_score", "global_offset"]
            return self

        self.global_offset_ = self._offset_for_expected_count(raw, y, w)
        self.group_columns_ = self._group_columns(work)
        self.feature_names_ = ["raw_score"] + [f"offset:{col}" for col in self.group_columns_]
        if not self.group_columns_:
            return self

        raw_series = pd.Series(raw, index=work.index)
        y_series = pd.Series(y, index=work.index)
        w_series = pd.Series(w, index=work.index)
        group_by: str | list[str] = self.group_columns_[0] if len(self.group_columns_) == 1 else self.group_columns_
        for keys, idx in work.groupby(group_by, observed=True, dropna=False).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            self.offsets_[self._group_key(keys)] = self._offset_for_expected_count(
                raw_series.iloc[idx_arr].to_numpy(dtype=float),
                y_series.iloc[idx_arr].to_numpy(dtype=int),
                w_series.iloc[idx_arr].to_numpy(dtype=float),
            )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.global_offset_ is None:
            raise RuntimeError("Calibrator is not fitted.")
        if "raw_score" not in frame.columns:
            raw = logit(frame["prob_raw"])
        else:
            raw = pd.to_numeric(frame["raw_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        offsets = np.full(len(frame), float(self.global_offset_), dtype=float)
        if self.group_columns_ and all(col in frame.columns for col in self.group_columns_):
            keys = frame[self.group_columns_].astype(str).agg("\x1f".join, axis=1)
            lookup = {"\x1f".join(key if isinstance(key, tuple) else (key,)): value for key, value in self.offsets_.items()}
            mapped = keys.map(lookup)
            mask = mapped.notna().to_numpy()
            offsets[mask] = mapped.loc[mask].to_numpy(dtype=float)
        return sigmoid(raw + offsets)


def fit_calibrator(frame: pd.DataFrame, method: str) -> PlattCalibrator | ConstantPrevalenceCalibrator:
    method_key = str(method or "platt_month").lower()
    if method_key in {"prior_offset_global", "prior_offset_month", "prior_offset_month_country"}:
        return PriorOffsetCalibrator(method_key).fit(frame)
    if method_key not in {"platt_global", "platt_month", "platt_month_country"}:
        raise ValueError(
            f"Unsupported calibration_method={method!r}; expected platt_global, platt_month, platt_month_country, "
            "prior_offset_global, prior_offset_month, or prior_offset_month_country."
        )
    return PlattCalibrator(method_key).fit(frame)


def apply_calibrator(calibrator: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(calibrator.predict_proba(frame), dtype=float), 0.0, 1.0)


def save_calibrator(calibrator: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)
    return path


def build_calibration_metadata(
    *,
    frame: pd.DataFrame,
    model_name: str,
    model_type: str,
    calibrator_type: str,
    calibration_start_date: str | None,
    calibration_end_date: str | None,
    test_start_date: str | None,
    test_end_date: str | None,
    calibrator_path: Path,
    calibrator: Any,
) -> CalibrationMetadata:
    weights = (
        pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if "eval_weight" in frame.columns
        else np.ones(len(frame), dtype=float)
    )
    y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    raw = pd.to_numeric(frame.get("raw_score", pd.Series(dtype=float)), errors="coerce")
    prob_raw = pd.to_numeric(frame.get("prob_raw", pd.Series(dtype=float)), errors="coerce")

    weighted_prev = float(np.average(y[valid], weights=weights[valid])) if valid.any() else None
    unweighted_prev = float(np.mean(y[valid])) if valid.any() else None
    return CalibrationMetadata(
        model_name=model_name,
        model_type=model_type,
        calibrator_type=calibrator_type,
        calibration_start_date=calibration_start_date,
        calibration_end_date=calibration_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        feature_columns_used_by_calibrator=list(getattr(calibrator, "feature_names_", [])),
        n_rows_unweighted=int(valid.sum()),
        n_rows_weighted=float(weights[valid].sum()) if valid.any() else 0.0,
        unweighted_prevalence=unweighted_prev,
        weighted_prevalence=weighted_prev,
        raw_score_mean=float(raw.mean()) if len(raw) and math.isfinite(float(raw.mean())) else None,
        prob_raw_mean=float(prob_raw.mean()) if len(prob_raw) and math.isfinite(float(prob_raw.mean())) else None,
        calibrator_path=str(calibrator_path),
    )


def write_calibration_metadata(path: Path, metadata: CalibrationMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
