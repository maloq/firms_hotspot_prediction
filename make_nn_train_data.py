from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.feature_generation.prepare_climate_data import (
    prepare_data as prepare_climate_data,
)


@dataclass
class PipelineConfig:
    config_path: Path
    features_path: Path
    output_dir: Path
    climate_data_dir: Path
    climate_variables: List[str]
    n_days: int
    datetime_column: str
    target_column: str
    train_end: str
    val_end: str
    ignored_features: List[str] = field(
        default_factory=lambda: ["datetime", "day", "latitude", "longitude", "year"]
    )
    cat_one_hot_threshold: int = 30
    scale_static: bool = True
    log_population: bool = True
    embedding_features: List[str] = field(default_factory=list)
    climate_cache_dir: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare neural-network ready dataset from pre-computed features."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/features_config_30d_LSTM_early_fusion.yaml"),
        help="Path to YAML config describing NN preprocessing parameters.",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Optional override for the features parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the directory where NPZ/JSON will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(value: Any, *, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required but not provided.")
    path = Path(value)
    if not path.is_absolute():
        path = path.resolve()
    return path


def build_pipeline_config(
    config_path: Path, cfg: Dict[str, Any], args: argparse.Namespace
) -> PipelineConfig:
    features_path = args.features_path or cfg.get("nn_features_path")
    if features_path is None:
        features_path = cfg.get("train_features_parquet_path")
    if features_path is None:
        raise KeyError(
            "Configuration must provide 'nn_features_path' or "
            "'train_features_parquet_path', or pass --features-path."
        )

    output_dir = args.output_dir or cfg.get("output_train_data_dir")
    if output_dir is None:
        raise KeyError(
            "Configuration must include 'output_train_data_dir' "
            "or provide --output-dir."
        )

    climate_cfg = cfg.get("climate_data_params") or {}
    climate_data_dir = climate_cfg.get("climate_data_dir")
    if climate_data_dir is None:
        raise KeyError(
            "Configuration missing 'climate_data_params.climate_data_dir'."
        )
    climate_variables = climate_cfg.get("climate_variables") or []
    if not climate_variables:
        raise KeyError(
            "Configuration missing 'climate_data_params.climate_variables'."
        )
    n_days = int(climate_cfg.get("n_days") or 0)
    if n_days <= 0:
        raise ValueError(
            "Configuration must provide positive 'climate_data_params.n_days'."
        )

    datetime_column = (
        cfg.get("nn_datetime_column")
        or cfg.get("date_column")
        or cfg.get("datetime_column")
        or "datetime"
    )
    target_column = (
        cfg.get("nn_target_column")
        or cfg.get("target_column")
        or "count"
    )
    train_end = cfg.get("train_end")
    val_end = cfg.get("val_end")
    if train_end is None or val_end is None:
        raise KeyError("Both 'train_end' and 'val_end' must be present in config.")

    ignored_features = cfg.get("ignored_features")
    if ignored_features:
        ignored_features = list(ignored_features)
    else:
        ignored_features = ["datetime", "day", "latitude", "longitude", "year"]
    if "ecoregion_realm" not in ignored_features:
        ignored_features.append("ecoregion_realm")

    cat_one_hot_threshold = int(cfg.get("cat_one_hot_threshold", 30))
    scale_static = bool(cfg.get("scale_static", True))
    log_population = bool(cfg.get("log_population", True))
    climate_cache_dir = cfg.get("climate_features_cache_dir")
    embedding_features = cfg.get("nn_embedding_features")
    if embedding_features:
        embedding_features = list(embedding_features)
    else:
        embedding_features = ["ecoregion_name"]

    return PipelineConfig(
        config_path=config_path,
        features_path=_resolve_path(features_path, label="features_path"),
        output_dir=_resolve_path(output_dir, label="output_dir"),
        climate_data_dir=_resolve_path(climate_data_dir, label="climate_data_dir"),
        climate_variables=list(climate_variables),
        n_days=n_days,
        datetime_column=datetime_column,
        target_column=target_column,
        train_end=str(train_end),
        val_end=str(val_end),
        ignored_features=list(ignored_features),
        cat_one_hot_threshold=cat_one_hot_threshold,
        scale_static=scale_static,
        log_population=log_population,
        climate_cache_dir=_resolve_path(climate_cache_dir, label="climate_cache_dir")
        if climate_cache_dir
        else None,
        embedding_features=embedding_features,
    )


def load_features_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features parquet not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Features dataframe at '{path}' is empty.")
    return df


def ensure_datetime_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Datetime column '{column}' not found in features dataframe.")
    dt_series = pd.to_datetime(df[column], errors="coerce")
    if dt_series.isna().any():
        raise ValueError(
            f"Failed to parse datetime values in column '{column}'. "
            "Ensure it contains valid timestamps."
        )
    return dt_series


def ensure_coordinate_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str, str]:
    lat_candidates = ("lat_rounded", "latitude", "lat")
    lon_candidates = ("lon_rounded", "longitude", "lon")

    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)

    if lat_col is None or lon_col is None:
        raise KeyError(
            "Features dataframe must contain latitude and longitude columns. "
            "Tried: lat_rounded/latitude/lat and lon_rounded/longitude/lon."
        )

    df_out = df.copy()
    if lat_col != "lat_rounded":
        df_out["lat_rounded"] = pd.to_numeric(df_out[lat_col], errors="coerce")
        lat_col = "lat_rounded"
    else:
        df_out["lat_rounded"] = pd.to_numeric(df_out["lat_rounded"], errors="coerce")

    if lon_col != "lon_rounded":
        df_out["lon_rounded"] = pd.to_numeric(df_out[lon_col], errors="coerce")
        lon_col = "lon_rounded"
    else:
        df_out["lon_rounded"] = pd.to_numeric(df_out["lon_rounded"], errors="coerce")

    if df_out[lat_col].isna().any() or df_out[lon_col].isna().any():
        raise ValueError("Latitude/longitude columns contain non-numeric values.")

    return df_out, lat_col, lon_col


def prepare_climate_matrix(
    df: pd.DataFrame,
    pipeline_cfg: PipelineConfig,
    *,
    lat_col: str,
    lon_col: str,
) -> np.ndarray:
    climate_cache = (
        pipeline_cfg.climate_cache_dir
        if pipeline_cfg.climate_cache_dir is not None
        else pipeline_cfg.output_dir / "climate_cache"
    )
    target_df = pd.DataFrame(
        {
            "acq_date": pd.to_datetime(df[pipeline_cfg.datetime_column], errors="coerce"),
            "lat_rounded": df[lat_col].astype(float),
            "lon_rounded": df[lon_col].astype(float),
        }
    )
    if target_df["acq_date"].isna().any():
        raise ValueError(
            "Unable to create 'acq_date' series required for climate extraction."
        )

    ts_matrix = prepare_climate_data(
        climate_data_dir=str(pipeline_cfg.climate_data_dir),
        climate_variables=pipeline_cfg.climate_variables,
        target_df=target_df,
        n_days=pipeline_cfg.n_days,
        prep_climate=True,
        test_mode=False,
        cache_dir=str(climate_cache),
        return_features_df=False,
    )
    ts_matrix = np.asarray(ts_matrix, dtype=np.float32)
    if ts_matrix.ndim != 2:
        raise ValueError(
            f"Expected climate matrix with ndim=2, got shape {ts_matrix.shape}."
        )
    if ts_matrix.shape[0] != len(df):
        raise ValueError(
            "Climate time-series matrix length does not match number of samples."
        )
    return ts_matrix


def _fill_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].fillna(0.0)
    return df


def _fill_categorical(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].astype(str).fillna("__nan__")
    return df


def prepare_feature_for_models(
    *,
    features_df: pd.DataFrame,
    climate_matrix: np.ndarray,
    n_days: int,
    target_col: str,
    train_end: str,
    val_end: str,
    datetime_col: str,
    static_exclude_cols: Optional[List[str]] = None,
    ignored_features: Optional[List[str]] = None,
    cat_one_hot_threshold: int = 100,
    scale_static: bool = True,
    log_population: bool = True,
    embedding_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    n_samples, total_len = climate_matrix.shape
    if total_len % n_days != 0:
        raise ValueError(
            "Climate matrix width must be divisible by n_days "
            f"(shape {climate_matrix.shape}, n_days={n_days})."
        )
    n_channels = total_len // n_days

    if target_col not in features_df.columns:
        raise KeyError(f"Target column '{target_col}' not present in features dataframe.")

    features_df = features_df.copy()
    embedding_features = [f for f in (embedding_features or []) if f in features_df.columns]

    if datetime_col not in features_df.columns:
        raise KeyError(f"Datetime column '{datetime_col}' not present in features dataframe.")
    dt_series = pd.to_datetime(features_df[datetime_col], errors="coerce")
    if dt_series.isna().any():
        raise ValueError(
            f"Failed to parse datetime values in column '{datetime_col}'. "
            "Ensure timestamps are valid."
        )
    features_df[datetime_col] = dt_series

    day_of_year = dt_series.dt.dayofyear.fillna(0).astype(np.int32)
    angle = 2.0 * np.pi * day_of_year.to_numpy(dtype=np.float32) / 366.0
    features_df["dayofyear_sin"] = np.sin(angle).astype(np.float32)
    features_df["dayofyear_cos"] = np.cos(angle).astype(np.float32)

    if ignored_features is None:
        ignored_features = ["datetime", "day", "latitude", "longitude", "year"]
    ignored_features = list(ignored_features)

    coord_cols = []
    if datetime_col in features_df.columns:
        coord_cols.append(datetime_col)
    for column in ("lat_rounded", "lon_rounded"):
        if column in features_df.columns:
            coord_cols.append(column)

    if coord_cols:
        coord_mask_df = features_df[coord_cols].reset_index(drop=True).copy()
        for col in coord_mask_df.columns:
            if col == datetime_col:
                coord_mask_df[col] = pd.to_datetime(coord_mask_df[col], errors="coerce")
            else:
                coord_mask_df[col] = (
                    pd.to_numeric(coord_mask_df[col], errors="coerce")
                    .astype("float32")
                    .fillna(0.0)
                )
    else:
        coord_mask_df = pd.DataFrame(index=range(n_samples))

    embedding_arrays: List[np.ndarray] = []
    embedding_info: List[Dict[str, Any]] = []
    for feature in embedding_features:
        values = features_df[feature].astype(str).fillna("__nan__")
        codes, uniques = pd.factorize(values, sort=True)
        if codes.ndim != 1:
            raise ValueError(f"Unexpected shape for factorized values of '{feature}'.")
        if len(uniques) == 0:
            continue
        codes = codes.astype(np.int64)
        embedding_arrays.append(codes)
        embedding_info.append(
            {
                "name": feature,
                "cardinality": int(len(uniques)),
                "vocabulary": [str(u) for u in uniques],
            }
        )

    if embedding_arrays:
        embedding_matrix_full = np.column_stack(embedding_arrays).astype(np.int64)
    else:
        embedding_matrix_full = np.zeros((n_samples, 0), dtype=np.int64)

    if static_exclude_cols is None:
        static_exclude_cols = [target_col]
    else:
        static_exclude_cols = list(static_exclude_cols)
        if target_col not in static_exclude_cols:
            static_exclude_cols.append(target_col)

    ignored_present = [c for c in ignored_features if c in features_df.columns]
    drop_cols = sorted(set(static_exclude_cols + ignored_present + embedding_features))

    x_stat_df_full = (
        features_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    )

    if log_population:
        if "population" in x_stat_df_full.columns:
            x_stat_df_full["population_log"] = np.log1p(
                x_stat_df_full["population"].astype(float)
            )
        else:
            raise KeyError("Expected 'population' column for log transformation.")

    if datetime_col in x_stat_df_full.columns:
        x_stat_df_full = x_stat_df_full.drop(columns=[datetime_col])

    cat_cols = [
        col
        for col in x_stat_df_full.columns
        if x_stat_df_full[col].dtype == object
        or str(x_stat_df_full[col].dtype).startswith("category")
    ]
    num_cols = [
        col
        for col in x_stat_df_full.columns
        if col not in cat_cols
        and not pd.api.types.is_datetime64_any_dtype(x_stat_df_full[col])
        and not pd.api.types.is_period_dtype(x_stat_df_full[col])
        and not pd.api.types.is_timedelta64_dtype(x_stat_df_full[col])
    ]

    x_stat_df_full = _fill_numeric(x_stat_df_full, num_cols)
    x_stat_df_full = _fill_categorical(x_stat_df_full, cat_cols)

    count_cat = {col: int(x_stat_df_full[col].nunique()) for col in cat_cols}
    ohe_cols = [col for col, k in count_cat.items() if k <= cat_one_hot_threshold]
    label_cols = [col for col in cat_cols if col not in ohe_cols]

    x_dyn_all = np.nan_to_num(climate_matrix.astype(np.float32), nan=0.0)

    dt_series = pd.to_datetime(features_df[datetime_col], errors="coerce")
    train_cut = pd.to_datetime(train_end)
    val_cut = pd.to_datetime(val_end)

    train_mask = dt_series <= train_cut
    val_mask = (dt_series > train_cut) & (dt_series <= val_cut)
    test_mask = dt_series > val_cut

    train_idx = features_df[train_mask].index.to_numpy()
    val_idx = features_df[val_mask].index.to_numpy()
    test_idx = features_df[test_mask].index.to_numpy()

    if len(train_idx) == 0:
        raise ValueError(
            "Training split is empty. Adjust 'train_end' to cover available data."
        )

    coord_mask_train_df = coord_mask_df.iloc[train_idx].reset_index(drop=True)
    coord_mask_val_df = coord_mask_df.iloc[val_idx].reset_index(drop=True)
    coord_mask_test_df = coord_mask_df.iloc[test_idx].reset_index(drop=True)

    x_dyn_all = (
        x_dyn_all.reshape((n_samples, n_channels, n_days))
        .transpose(0, 2, 1)
        .astype(np.float32)
    )
    x_dyn_train = x_dyn_all[train_idx]
    x_dyn_val = x_dyn_all[val_idx]
    x_dyn_test = x_dyn_all[test_idx]

    mean_dyn = x_dyn_train.mean(axis=(0, 1), keepdims=True)
    std_dyn = x_dyn_train.std(axis=(0, 1), keepdims=True)
    std_dyn[std_dyn == 0.0] = 1.0
    x_dyn_train = ((x_dyn_train - mean_dyn) / std_dyn).astype(np.float32)
    x_dyn_val = ((x_dyn_val - mean_dyn) / std_dyn).astype(np.float32)
    x_dyn_test = ((x_dyn_test - mean_dyn) / std_dyn).astype(np.float32)
    dyn_scaler = {"mean": mean_dyn.astype(np.float32), "std": std_dyn.astype(np.float32)}

    x_stat_df_full = x_stat_df_full.reset_index(drop=True)
    x_stat_train_df = x_stat_df_full.loc[train_idx].reset_index(drop=True)
    x_stat_val_df = x_stat_df_full.loc[val_idx].reset_index(drop=True)
    x_stat_test_df = x_stat_df_full.loc[test_idx].reset_index(drop=True)

    num_scaler = None
    if num_cols:
        x_num_train = x_stat_train_df[num_cols].astype(np.float32).values
        x_num_val = x_stat_val_df[num_cols].astype(np.float32).values
        x_num_test = x_stat_test_df[num_cols].astype(np.float32).values

        if scale_static:
            num_scaler = StandardScaler()
            x_num_train = num_scaler.fit_transform(x_num_train).astype(np.float32)
            x_num_val = (
                num_scaler.transform(x_num_val).astype(np.float32)
                if len(x_num_val) > 0
                else x_num_val.astype(np.float32)
            )
            x_num_test = (
                num_scaler.transform(x_num_test).astype(np.float32)
                if len(x_num_test) > 0
                else x_num_test.astype(np.float32)
            )
        else:
            x_num_train = x_num_train.astype(np.float32)
            x_num_val = x_num_val.astype(np.float32)
            x_num_test = x_num_test.astype(np.float32)
    else:
        x_num_train = np.zeros((len(train_idx), 0), dtype=np.float32)
        x_num_val = np.zeros((len(val_idx), 0), dtype=np.float32)
        x_num_test = np.zeros((len(test_idx), 0), dtype=np.float32)

    ohe_encoders: Dict[str, OneHotEncoder] = {}
    x_ohe_train_list: List[np.ndarray] = []
    x_ohe_val_list: List[np.ndarray] = []
    x_ohe_test_list: List[np.ndarray] = []
    n_val_rows = len(x_stat_val_df)
    n_test_rows = len(x_stat_test_df)

    for col in ohe_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc.fit(x_stat_train_df[[col]])
        train_arr = enc.transform(x_stat_train_df[[col]])
        val_arr = (
            enc.transform(x_stat_val_df[[col]])
            if n_val_rows > 0
            else np.zeros((0, train_arr.shape[1]), dtype=train_arr.dtype)
        )
        test_arr = (
            enc.transform(x_stat_test_df[[col]])
            if n_test_rows > 0
            else np.zeros((0, train_arr.shape[1]), dtype=train_arr.dtype)
        )
        x_ohe_train_list.append(train_arr)
        x_ohe_val_list.append(val_arr)
        x_ohe_test_list.append(test_arr)
        ohe_encoders[col] = enc

    if x_ohe_train_list:
        x_ohe_train = np.hstack(x_ohe_train_list).astype(np.float32)
        x_ohe_val = np.hstack(x_ohe_val_list).astype(np.float32)
        x_ohe_test = np.hstack(x_ohe_test_list).astype(np.float32)
    else:
        x_ohe_train = np.zeros((len(train_idx), 0), dtype=np.float32)
        x_ohe_val = np.zeros((len(val_idx), 0), dtype=np.float32)
        x_ohe_test = np.zeros((len(test_idx), 0), dtype=np.float32)

    ord_encoders: Dict[str, OrdinalEncoder] = {}
    x_ord_train_list: List[np.ndarray] = []
    x_ord_val_list: List[np.ndarray] = []
    x_ord_test_list: List[np.ndarray] = []

    for col in label_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(x_stat_train_df[[col]])
        x_ord_train_list.append(enc.transform(x_stat_train_df[[col]]))
        x_ord_val_list.append(
            enc.transform(x_stat_val_df[[col]])
            if n_val_rows > 0
            else np.zeros((0, 1), dtype=np.float32)
        )
        x_ord_test_list.append(
            enc.transform(x_stat_test_df[[col]])
            if n_test_rows > 0
            else np.zeros((0, 1), dtype=np.float32)
        )
        ord_encoders[col] = enc

    if x_ord_train_list:
        x_ord_train = np.hstack(x_ord_train_list).astype(np.float32)
        x_ord_val = np.hstack(x_ord_val_list).astype(np.float32)
        x_ord_test = np.hstack(x_ord_test_list).astype(np.float32)
    else:
        x_ord_train = np.zeros((len(train_idx), 0), dtype=np.float32)
        x_ord_val = np.zeros((len(val_idx), 0), dtype=np.float32)
        x_ord_test = np.zeros((len(test_idx), 0), dtype=np.float32)

    x_stat_train = np.hstack([x_num_train, x_ohe_train, x_ord_train]).astype(np.float32)
    x_stat_val = np.hstack([x_num_val, x_ohe_val, x_ord_val]).astype(np.float32)
    x_stat_test = np.hstack([x_num_test, x_ohe_test, x_ord_test]).astype(np.float32)

    if embedding_matrix_full.shape[1] > 0:
        x_cat_train = embedding_matrix_full[train_idx]
        x_cat_val = embedding_matrix_full[val_idx]
        x_cat_test = embedding_matrix_full[test_idx]
    else:
        x_cat_train = np.zeros((len(train_idx), 0), dtype=np.int64)
        x_cat_val = np.zeros((len(val_idx), 0), dtype=np.int64)
        x_cat_test = np.zeros((len(test_idx), 0), dtype=np.int64)

    y_all = features_df[target_col].to_numpy()
    y_train = y_all[train_idx].astype(np.float32)
    y_val = y_all[val_idx].astype(np.float32)
    y_test = y_all[test_idx].astype(np.float32)

    static_feature_names: List[str] = []
    static_feature_names.extend(num_cols)
    for col in ohe_cols:
        enc = ohe_encoders[col]
        cats = enc.categories_[0]
        static_feature_names.extend([f"{col}__{str(cat)}" for cat in cats])
    static_feature_names.extend([f"{col}__ord" for col in label_cols])

    dyn_channel_names = [f"dyn_ch_{i}" for i in range(n_channels)]
    dyn_expanded = [
        f"day{day:02d}_ch{ch}" for day in range(n_days) for ch in range(n_channels)
    ]

    features_mask = {
        "static": static_feature_names,
        "dyn_channels": dyn_channel_names,
        "dyn_expanded": dyn_expanded,
        "n_days": n_days,
        "n_channels": n_channels,
    }

    categorical_embeddings_meta: List[Dict[str, Any]] = []
    for idx, info in enumerate(embedding_info):
        cardinality = max(1, int(info["cardinality"]))
        recommended_dim = min(64, max(8, cardinality // 2))
        categorical_embeddings_meta.append(
            {
                "name": info["name"],
                "cardinality": cardinality,
                "embedding_dim": recommended_dim,
                "column_index": idx,
            }
        )

    return {
        "x_dyn_train": x_dyn_train,
        "x_dyn_val": x_dyn_val,
        "x_dyn_test": x_dyn_test,
        "x_stat_train": x_stat_train,
        "x_stat_val": x_stat_val,
        "x_stat_test": x_stat_test,
        "x_cat_train": x_cat_train,
        "x_cat_val": x_cat_val,
        "x_cat_test": x_cat_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "num_cols": num_cols,
        "ohe_cols": ohe_cols,
        "label_cols": label_cols,
        "num_scaler": num_scaler,
        "ohe_encoders": ohe_encoders,
        "ord_encoders": ord_encoders,
        "dyn_scaler": dyn_scaler,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "n_days": n_days,
        "n_channels": n_channels,
        "embedding_features": [info["name"] for info in embedding_info],
        "embedding_cardinalities": [int(info["cardinality"]) for info in embedding_info],
        "embedding_vocabulary": {info["name"]: info["vocabulary"] for info in embedding_info},
        "categorical_embeddings_meta": categorical_embeddings_meta,
        "coord_mask_df": coord_mask_df,
        "coord_mask_train_df": coord_mask_train_df,
        "coord_mask_val_df": coord_mask_val_df,
        "coord_mask_test_df": coord_mask_test_df,
        "features_mask": features_mask,
        "ignored_features": ignored_present,
    }


def serialize_scaler(scaler: Optional[StandardScaler]) -> Optional[Dict[str, Any]]:
    if scaler is None:
        return None
    return {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
    }


def serialize_one_hot(encoders: Dict[str, OneHotEncoder]) -> Dict[str, List[List[Any]]]:
    result: Dict[str, List[List[Any]]] = {}
    for col, enc in encoders.items():
        result[col] = [cats.tolist() for cats in enc.categories_]
    return result


def serialize_ordinal(encoders: Dict[str, OrdinalEncoder]) -> Dict[str, List[List[Any]]]:
    result: Dict[str, List[List[Any]]] = {}
    for col, enc in encoders.items():
        result[col] = [cats.tolist() for cats in enc.categories_]
    return result


def extract_coord_arrays(
    coord_df: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    arrays: Dict[str, np.ndarray] = {}
    columns_meta: List[Dict[str, Any]] = []

    for col in coord_df.columns:
        values = coord_df[col]
        key = f"coord_{col}"
        if pd.api.types.is_datetime64_any_dtype(values):
            arr = values.astype("datetime64[ns]").view("int64")
            arrays[key] = arr
            columns_meta.append(
                {"name": col, "npz_key": key, "dtype": "datetime64[ns]", "unit": "ns"}
            )
        else:
            if values.dtype == object:
                arr = values.astype(str).to_numpy()
                dtype = "str"
            else:
                arr = values.to_numpy()
                dtype = str(arr.dtype)
            arrays[key] = arr
            columns_meta.append({"name": col, "npz_key": key, "dtype": dtype})

    return arrays, columns_meta


def build_metadata(
    pipeline_cfg: PipelineConfig,
    prepared: Dict[str, Any],
    coord_columns_meta: List[Dict[str, Any]],
) -> Dict[str, Any]:
    dyn_scaler = prepared.get("dyn_scaler") or {}
    dyn_scaler_serialized = {
        "mean": np.asarray(dyn_scaler.get("mean", []), dtype=np.float32)
        .ravel()
        .astype(float)
        .tolist(),
        "std": np.asarray(dyn_scaler.get("std", []), dtype=np.float32)
        .ravel()
        .astype(float)
        .tolist(),
    }

    metadata: Dict[str, Any] = {
        "source": {
            "config_path": str(pipeline_cfg.config_path),
            "features_path": str(pipeline_cfg.features_path),
            "climate_data_dir": str(pipeline_cfg.climate_data_dir),
        },
        "dataset": {
            "n_samples": int(len(prepared["coord_mask_df"])),
            "n_days": int(prepared["n_days"]),
            "n_channels": int(prepared["n_channels"]),
            "target_column": pipeline_cfg.target_column,
            "datetime_column": pipeline_cfg.datetime_column,
        },
        "splits": {
            "train_idx": prepared["train_idx"].astype(int).tolist(),
            "val_idx": prepared["val_idx"].astype(int).tolist(),
            "test_idx": prepared["test_idx"].astype(int).tolist(),
        },
        "features": {
            "numeric": prepared["num_cols"],
            "one_hot": prepared["ohe_cols"],
            "ordinal": prepared["label_cols"],
            "ignored": prepared["ignored_features"],
            "mask": prepared["features_mask"],
        },
        "preprocessing": {
            "numeric_scaler": serialize_scaler(prepared.get("num_scaler")),
            "dynamic_scaler": dyn_scaler_serialized,
            "one_hot_categories": serialize_one_hot(prepared.get("ohe_encoders", {})),
            "ordinal_categories": serialize_ordinal(prepared.get("ord_encoders", {})),
        },
        "coordinates": {"columns": coord_columns_meta},
        "config": {
            "train_end": pipeline_cfg.train_end,
            "val_end": pipeline_cfg.val_end,
            "cat_one_hot_threshold": pipeline_cfg.cat_one_hot_threshold,
            "scale_static": pipeline_cfg.scale_static,
            "log_population": pipeline_cfg.log_population,
        },
    }
    categorical_embeddings_meta = prepared.get("categorical_embeddings_meta") or []
    metadata["categorical_embeddings"] = categorical_embeddings_meta
    metadata["categorical_embedding_vocabulary"] = prepared.get("embedding_vocabulary", {})
    return metadata


def save_outputs(
    pipeline_cfg: PipelineConfig,
    prepared: Dict[str, Any],
    coord_arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    *,
    overwrite: bool,
) -> Tuple[Path, Path]:
    output_dir = pipeline_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "prepared_data.npz"
    metadata_path = output_dir / "prepared_metadata.json"

    if not overwrite:
        for path in (npz_path, metadata_path):
            if path.exists():
                raise FileExistsError(
                    f"File '{path}' already exists. Pass --overwrite to replace it."
                )

    payload: Dict[str, np.ndarray] = {
        "x_dyn_train": prepared["x_dyn_train"],
        "x_dyn_val": prepared["x_dyn_val"],
        "x_dyn_test": prepared["x_dyn_test"],
        "x_stat_train": prepared["x_stat_train"],
        "x_stat_val": prepared["x_stat_val"],
        "x_stat_test": prepared["x_stat_test"],
        "x_cat_train": np.asarray(prepared["x_cat_train"], dtype=np.int64),
        "x_cat_val": np.asarray(prepared["x_cat_val"], dtype=np.int64),
        "x_cat_test": np.asarray(prepared["x_cat_test"], dtype=np.int64),
        "y_train": prepared["y_train"],
        "y_val": prepared["y_val"],
        "y_test": prepared["y_test"],
        "train_idx": prepared["train_idx"].astype(np.int64),
        "val_idx": prepared["val_idx"].astype(np.int64),
        "test_idx": prepared["test_idx"].astype(np.int64),
    }
    payload.update(coord_arrays)

    np.savez_compressed(npz_path, **payload)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return npz_path, metadata_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_cfg = build_pipeline_config(args.config, config, args)

    print(f"Loading features from {pipeline_cfg.features_path} ...")
    features_df = load_features_dataframe(pipeline_cfg.features_path)

    print(f"Preparing coordinate columns and datetime '{pipeline_cfg.datetime_column}' ...")
    features_df[pipeline_cfg.datetime_column] = ensure_datetime_column(
        features_df, pipeline_cfg.datetime_column
    )
    features_df, lat_col, lon_col = ensure_coordinate_columns(features_df)

    print("Extracting climate time-series ...")
    climate_matrix = prepare_climate_matrix(
        features_df,
        pipeline_cfg,
        lat_col=lat_col,
        lon_col=lon_col,
    )
    print(f"Climate matrix shape: {climate_matrix.shape}")

    print("Preparing model-ready tensors ...")
    prepared = prepare_feature_for_models(
        features_df=features_df,
        climate_matrix=climate_matrix,
        n_days=pipeline_cfg.n_days,
        target_col=pipeline_cfg.target_column,
        train_end=pipeline_cfg.train_end,
        val_end=pipeline_cfg.val_end,
        datetime_col=pipeline_cfg.datetime_column,
        static_exclude_cols=None,
        ignored_features=pipeline_cfg.ignored_features,
        cat_one_hot_threshold=pipeline_cfg.cat_one_hot_threshold,
        scale_static=pipeline_cfg.scale_static,
        log_population=pipeline_cfg.log_population,
        embedding_features=pipeline_cfg.embedding_features,
    )

    coord_arrays, coord_columns_meta = extract_coord_arrays(prepared["coord_mask_df"])
    metadata = build_metadata(pipeline_cfg, prepared, coord_columns_meta)

    npz_path, metadata_path = save_outputs(
        pipeline_cfg,
        prepared,
        coord_arrays,
        metadata,
        overwrite=args.overwrite,
    )

    print("Prepared data saved:")
    print(f"  NPZ:  {npz_path}")
    print(f"  JSON: {metadata_path}")
    print("Done.")


if __name__ == "__main__":
    main()
