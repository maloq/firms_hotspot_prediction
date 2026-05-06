import pandas as pd
import numpy as np
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import yaml
import joblib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

DEFAULT_IGNORED_FEATURES = ["datetime", "day", "latitude", "longitude", "year"]
MISSING_CATEGORY = "__nan__"
PREPROCESSING_VERSION = 2


def load_data(features_parquet_path):
    parquet_file = features_parquet_path
    npz_file = features_parquet_path.replace('.parquet', '.npz')

    final_features_df = None
    ts_matrix = None

    if os.path.exists(parquet_file):
        final_features_df = pd.read_parquet(parquet_file)
        print(f"Loaded features DataFrame from '{parquet_file}'.")
        print(f"DataFrame shape: {final_features_df.shape}")
    else:
        print(f"Error: Parquet file not found at '{parquet_file}'")

    if os.path.exists(npz_file):
        with np.load(npz_file) as data:
            if 'climate_features' in data:
                ts_matrix = data['climate_features']
                print(f"Loaded climate features from '{npz_file}'.")
                print(f"Climate features matrix shape: {ts_matrix.shape}")
            else:
                raise ValueError(f"Error: 'climate_features' not found in {npz_file}. Available keys: {data.files}")

    else:
        print(f"Error: NPZ file not found at '{npz_file}'")

    return final_features_df, ts_matrix


def _make_coord_mask_df(
    features_df: pd.DataFrame,
    datetime_col: Optional[str],
) -> pd.DataFrame:
    coord_cols = []
    if datetime_col is not None and datetime_col in features_df.columns:
        coord_cols.append(datetime_col)

    for col in ["lat_rounded", "lon_rounded"]:
        if col in features_df.columns:
            coord_cols.append(col)

    if not coord_cols:
        return pd.DataFrame(index=range(len(features_df)))

    coord_mask_df = features_df[coord_cols].reset_index(drop=True).copy()
    for col in coord_mask_df.columns:
        if col == datetime_col:
            coord_mask_df[col] = pd.to_datetime(coord_mask_df[col], errors="coerce")
        else:
            coord_mask_df[col] = pd.to_numeric(coord_mask_df[col], errors="coerce").astype("float32")
    return coord_mask_df


def _build_static_frame(
    features_df: pd.DataFrame,
    target_col: str,
    static_exclude_cols: Optional[List[str]],
    ignored_features: List[str],
    datetime_col: Optional[str],
    log_population: bool,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if static_exclude_cols is None:
        static_exclude_cols = [target_col]
    else:
        static_exclude_cols = list(static_exclude_cols)
        if target_col not in static_exclude_cols:
            static_exclude_cols.append(target_col)

    ignored_present = [col for col in ignored_features if col in features_df.columns]
    drop_cols = list(dict.fromkeys(static_exclude_cols + ignored_present))
    x_stat_df_full = features_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)

    if log_population:
        if "population" not in x_stat_df_full.columns:
            raise KeyError("No 'population' in static features")
        population = pd.to_numeric(x_stat_df_full["population"], errors="coerce")
        population = population.clip(lower=0)
        x_stat_df_full["population_log"] = np.log1p(population)

    if datetime_col is not None and datetime_col in x_stat_df_full.columns:
        x_stat_df_full = x_stat_df_full.drop(columns=[datetime_col])

    cat_cols: list[str] = []
    num_cols: list[str] = []
    skipped_cols: list[str] = []
    for col in x_stat_df_full.columns:
        series = x_stat_df_full[col]
        is_temporal = (
            pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_period_dtype(series)
            or pd.api.types.is_timedelta64_dtype(series)
        )
        if is_temporal:
            skipped_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    if skipped_cols:
        x_stat_df_full = x_stat_df_full.drop(columns=skipped_cols)

    return x_stat_df_full, num_cols, cat_cols, ignored_present


def _split_indices_by_date(
    features_df: pd.DataFrame,
    datetime_col: str,
    train_end: str,
    val_end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = pd.to_datetime(features_df[datetime_col], errors="coerce")
    train_cut = pd.to_datetime(train_end)
    val_cut = pd.to_datetime(val_end)

    train_mask = dt <= train_cut
    val_mask = (dt > train_cut) & (dt <= val_cut)
    test_mask = dt > val_cut

    train_idx = features_df[train_mask].index.to_numpy()
    val_idx = features_df[val_mask].index.to_numpy()
    test_idx = features_df[test_mask].index.to_numpy()

    print(
        "Split sizes -> train: {train} | val: {val} | test: {test}".format(
            train=len(train_idx), val=len(val_idx), test=len(test_idx)
        )
    )
    if len(train_idx) == 0:
        raise ValueError(
            "Training split is empty. Please adjust 'train_end' to cover available data."
        )
    if len(val_idx) == 0:
        print(
            "Warning: validation split is empty. Check 'train_end'/'val_end' or dataset coverage."
        )
    if len(test_idx) == 0:
        print("Warning: test split is empty. Check 'val_end' or dataset coverage.")

    return train_idx, val_idx, test_idx


def _reshape_dynamic_features(climate_matrix: np.ndarray, n_days: int) -> tuple[np.ndarray, int]:
    n_samples, total_len = climate_matrix.shape
    if total_len % n_days != 0:
        raise ValueError("climate_matrix shape must be divisible by n_days")

    n_channels = total_len // n_days
    x_dyn_all = climate_matrix.astype(np.float32)
    x_dyn_all = x_dyn_all.reshape((n_samples, n_channels, n_days)).transpose(0, 2, 1)
    return x_dyn_all.astype(np.float32), n_channels


def _fit_dynamic_scaler(x_dyn_all: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    x_dyn_train_raw = x_dyn_all[train_idx]
    fill = np.nanmedian(x_dyn_train_raw, axis=(0, 1), keepdims=True).astype(np.float32)
    fill = np.nan_to_num(fill, nan=0.0)

    x_dyn_filled = np.where(np.isnan(x_dyn_all), fill, x_dyn_all).astype(np.float32)
    x_dyn_train = x_dyn_filled[train_idx]

    mean = x_dyn_train.mean(axis=(0, 1), keepdims=True)
    std = x_dyn_train.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    x_dyn_scaled = ((x_dyn_filled - mean) / std).astype(np.float32)
    dyn_scaler = {"mean": mean, "std": std, "fill": fill}
    return x_dyn_scaled, dyn_scaler


def _transform_dynamic_features(
    climate_matrix: np.ndarray,
    n_days: int,
    dyn_scaler: dict,
) -> tuple[np.ndarray, int]:
    x_dyn_all, n_channels = _reshape_dynamic_features(climate_matrix, n_days)

    fill = dyn_scaler.get("fill")
    if fill is None:
        fill = np.zeros((1, 1, n_channels), dtype=np.float32)
    fill = np.asarray(fill, dtype=np.float32)

    mean = np.asarray(dyn_scaler["mean"], dtype=np.float32)
    std = np.asarray(dyn_scaler["std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std).astype(np.float32)

    x_dyn_all = np.where(np.isnan(x_dyn_all), fill, x_dyn_all).astype(np.float32)
    x_dyn_all = ((x_dyn_all - mean) / std).astype(np.float32)
    return x_dyn_all, n_channels


def _numeric_frame(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in num_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = np.nan
    return out


def _fit_transform_numeric_static(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list[str],
    *,
    scale_static: bool,
    add_missing_indicators: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler | None, dict[str, float], list[str]]:
    if not num_cols:
        empty_train = np.zeros((len(train_df), 0), dtype=np.float32)
        empty_val = np.zeros((len(val_df), 0), dtype=np.float32)
        empty_test = np.zeros((len(test_df), 0), dtype=np.float32)
        return empty_train, empty_val, empty_test, None, {}, []

    train_num = _numeric_frame(train_df, num_cols)
    val_num = _numeric_frame(val_df, num_cols)
    test_num = _numeric_frame(test_df, num_cols)

    fill_values = train_num.median(axis=0, skipna=True).fillna(0.0).astype(float).to_dict()
    indicator_cols = [f"{col}__missing" for col in num_cols] if add_missing_indicators else []

    def fill_and_indicator(num_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = num_df.isna().astype(np.float32).to_numpy()
        filled = num_df.fillna(fill_values).astype(np.float32).to_numpy()
        return filled, missing

    x_num_train, miss_train = fill_and_indicator(train_num)
    x_num_val, miss_val = fill_and_indicator(val_num)
    x_num_test, miss_test = fill_and_indicator(test_num)

    num_scaler = None
    if scale_static:
        num_scaler = StandardScaler()
        x_num_train = num_scaler.fit_transform(x_num_train).astype(np.float32)
        x_num_val = num_scaler.transform(x_num_val).astype(np.float32) if len(val_df) else x_num_val
        x_num_test = num_scaler.transform(x_num_test).astype(np.float32) if len(test_df) else x_num_test

    if add_missing_indicators:
        x_num_train = np.hstack([x_num_train, miss_train]).astype(np.float32)
        x_num_val = np.hstack([x_num_val, miss_val]).astype(np.float32)
        x_num_test = np.hstack([x_num_test, miss_test]).astype(np.float32)

    return x_num_train, x_num_val, x_num_test, num_scaler, fill_values, indicator_cols


def _transform_numeric_static(df: pd.DataFrame, meta: dict) -> np.ndarray:
    num_cols = list(meta.get("num_cols") or [])
    if not num_cols:
        return np.zeros((len(df), 0), dtype=np.float32)

    num_df = _numeric_frame(df, num_cols)
    fill_values = dict(meta.get("num_fill_values") or {})
    fill_values = {col: float(fill_values.get(col, 0.0)) for col in num_cols}

    missing = num_df.isna().astype(np.float32).to_numpy()
    filled = num_df.fillna(fill_values).astype(np.float32).to_numpy()

    num_scaler = meta.get("num_scaler")
    if meta.get("scale_static", num_scaler is not None) and num_scaler is not None:
        filled = num_scaler.transform(filled).astype(np.float32)

    indicator_cols = list(meta.get("numeric_missing_indicator_cols") or [])
    if indicator_cols:
        return np.hstack([filled, missing]).astype(np.float32)
    return filled.astype(np.float32)


def _categorical_frame(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cat_cols:
        if col in df.columns:
            out[col] = df[col].fillna(MISSING_CATEGORY).astype(str)
        else:
            out[col] = MISSING_CATEGORY
    return out


def _fit_transform_categorical_static(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str],
    *,
    cat_one_hot_threshold: int,
    allow_ordinal_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict, list[str], list[str], list[str]]:
    if not cat_cols:
        empty_train = np.zeros((len(train_df), 0), dtype=np.float32)
        empty_val = np.zeros((len(val_df), 0), dtype=np.float32)
        empty_test = np.zeros((len(test_df), 0), dtype=np.float32)
        return empty_train, empty_val, empty_test, {}, {}, [], [], []

    train_cat = _categorical_frame(train_df, cat_cols)
    val_cat = _categorical_frame(val_df, cat_cols)
    test_cat = _categorical_frame(test_df, cat_cols)

    count_cat = {col: int(train_cat[col].nunique()) for col in cat_cols}
    print(count_cat)

    ohe_cols = [col for col, n_unique in count_cat.items() if n_unique <= cat_one_hot_threshold]
    label_cols = [col for col in cat_cols if col not in ohe_cols]
    if label_cols and not allow_ordinal_fallback:
        raise ValueError(
            "Categorical columns exceed cat_one_hot_threshold and ordinal fallback is disabled: "
            f"{label_cols}. Increase nn_preprocessing.cat_one_hot_threshold or explicitly set "
            "allow_ordinal_fallback: true."
        )

    ohe_encoders: dict[str, OneHotEncoder] = {}
    ord_encoders: dict[str, OrdinalEncoder] = {}
    static_feature_names: list[str] = []

    x_ohe_train_list = []
    x_ohe_val_list = []
    x_ohe_test_list = []
    for col in ohe_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc.fit(train_cat[[col]])
        train_arr = enc.transform(train_cat[[col]])
        val_arr = enc.transform(val_cat[[col]]) if len(val_df) else np.zeros((0, train_arr.shape[1]))
        test_arr = enc.transform(test_cat[[col]]) if len(test_df) else np.zeros((0, train_arr.shape[1]))

        x_ohe_train_list.append(train_arr)
        x_ohe_val_list.append(val_arr)
        x_ohe_test_list.append(test_arr)
        ohe_encoders[col] = enc
        static_feature_names.extend([f"{col}__{str(cat)}" for cat in enc.categories_[0]])

    x_ord_train_list = []
    x_ord_val_list = []
    x_ord_test_list = []
    for col in label_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(train_cat[[col]])
        x_ord_train_list.append(enc.transform(train_cat[[col]]))
        x_ord_val_list.append(enc.transform(val_cat[[col]]) if len(val_df) else np.zeros((0, 1)))
        x_ord_test_list.append(enc.transform(test_cat[[col]]) if len(test_df) else np.zeros((0, 1)))
        ord_encoders[col] = enc
        static_feature_names.append(f"{col}__ord")

    if x_ohe_train_list:
        x_ohe_train = np.hstack(x_ohe_train_list).astype(np.float32)
        x_ohe_val = np.hstack(x_ohe_val_list).astype(np.float32)
        x_ohe_test = np.hstack(x_ohe_test_list).astype(np.float32)
    else:
        x_ohe_train = np.zeros((len(train_df), 0), dtype=np.float32)
        x_ohe_val = np.zeros((len(val_df), 0), dtype=np.float32)
        x_ohe_test = np.zeros((len(test_df), 0), dtype=np.float32)

    if x_ord_train_list:
        x_ord_train = np.hstack(x_ord_train_list).astype(np.float32)
        x_ord_val = np.hstack(x_ord_val_list).astype(np.float32)
        x_ord_test = np.hstack(x_ord_test_list).astype(np.float32)
    else:
        x_ord_train = np.zeros((len(train_df), 0), dtype=np.float32)
        x_ord_val = np.zeros((len(val_df), 0), dtype=np.float32)
        x_ord_test = np.zeros((len(test_df), 0), dtype=np.float32)

    x_cat_train = np.hstack([x_ohe_train, x_ord_train]).astype(np.float32)
    x_cat_val = np.hstack([x_ohe_val, x_ord_val]).astype(np.float32)
    x_cat_test = np.hstack([x_ohe_test, x_ord_test]).astype(np.float32)

    return (
        x_cat_train,
        x_cat_val,
        x_cat_test,
        ohe_encoders,
        ord_encoders,
        ohe_cols,
        label_cols,
        static_feature_names,
    )


def _transform_categorical_static(df: pd.DataFrame, meta: dict) -> np.ndarray:
    ohe_cols = list(meta.get("ohe_cols") or [])
    label_cols = list(meta.get("label_cols") or [])
    ohe_encoders = meta.get("ohe_encoders") or {}
    ord_encoders = meta.get("ord_encoders") or {}

    cat_cols = ohe_cols + label_cols
    if not cat_cols:
        return np.zeros((len(df), 0), dtype=np.float32)

    cat_df = _categorical_frame(df, cat_cols)
    parts = []
    for col in ohe_cols:
        enc = ohe_encoders[col]
        parts.append(enc.transform(cat_df[[col]]).astype(np.float32))
    for col in label_cols:
        enc = ord_encoders[col]
        parts.append(enc.transform(cat_df[[col]]).astype(np.float32))

    if not parts:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.hstack(parts).astype(np.float32)


def transform_features_with_metadata(
    features_df: pd.DataFrame,
    climate_matrix: np.ndarray,
    metadata: dict,
    *,
    datetime_col: Optional[str] = None,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a saved NN preprocessing contract without refitting anything."""

    n_days = int(metadata["n_days"])
    expected_channels = int(metadata["n_channels"])
    datetime_col = datetime_col or metadata.get("datetime_col") or "datetime"
    target_col = target_col or metadata.get("target_col") or "count"
    ignored_features = list(metadata.get("ignored_features") or DEFAULT_IGNORED_FEATURES)
    log_population = bool(metadata.get("log_population", True))
    static_exclude_cols = metadata.get("static_exclude_cols")

    x_dyn_all, n_channels = _transform_dynamic_features(
        climate_matrix=climate_matrix,
        n_days=n_days,
        dyn_scaler=metadata["dyn_scaler"],
    )
    if n_channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} dynamic channels from metadata, got {n_channels}."
        )

    x_stat_df_full, _, _, _ = _build_static_frame(
        features_df=features_df,
        target_col=target_col,
        static_exclude_cols=static_exclude_cols,
        ignored_features=ignored_features,
        datetime_col=datetime_col,
        log_population=log_population,
    )

    x_num = _transform_numeric_static(x_stat_df_full, metadata)
    x_cat = _transform_categorical_static(x_stat_df_full, metadata)
    x_stat = np.hstack([x_num, x_cat]).astype(np.float32)

    expected_static = len(metadata.get("features_mask", {}).get("static", []))
    if expected_static and x_stat.shape[1] != expected_static:
        raise ValueError(
            f"Static feature width mismatch: metadata expects {expected_static}, got {x_stat.shape[1]}."
        )

    coord_mask_df = _make_coord_mask_df(features_df, datetime_col)
    out: Dict[str, Any] = {
        "x_dyn": x_dyn_all,
        "x_stat": x_stat,
        "coord_mask_df": coord_mask_df,
        "n_days": n_days,
        "n_channels": n_channels,
        "features_mask": metadata.get("features_mask", {}),
    }
    if target_col in features_df.columns:
        out["y"] = features_df[target_col].values.astype(np.float32)
    return out


def ensure_feature_files(parquet_path: Path, config_path: Path) -> None:
    parquet_path = Path(parquet_path)
    config_path = Path(config_path)
    npz_path = parquet_path.with_suffix('.npz')

    if parquet_path.exists() and npz_path.exists():
        return

    print(f"Features files not found. Generating with make_features_nn.py -> {parquet_path} and {npz_path}")
    cmd = [
        sys.executable,
        "src/feature_generation/make_features_nn.py",
        "--config",
        str(config_path),
        "--output",
        str(parquet_path),
    ]
    subprocess.run(cmd, check=True)

    if not parquet_path.exists() or not npz_path.exists():
        raise RuntimeError(
            "make_features_nn.py finished but expected feature files are still missing"
        )


def plot_and_save_visualizations(climate_variables, output_dir):
    """Generates and saves histograms and plots for climate variables."""
    # --- Plot and save histograms ---
    print("Generating and saving histograms of mean values...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (name, data) in enumerate(climate_variables.items()):
        mean_values = np.mean(data, axis=1)
        axes[i].hist(mean_values, bins=50)
        axes[i].set_title(f"Mean {name}")
        axes[i].set_xlabel("Mean value")
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    histogram_path = os.path.join(output_dir, "mean_values_histograms.png")
    plt.savefig(histogram_path)
    plt.close(fig)
    print(f"Histograms saved to {histogram_path}")

    print("Generating and saving plot of mean values...")
    fig, axes = plt.subplots(len(climate_variables), 1, figsize=(12, 16), sharex=True)

    for i, (name, data) in enumerate(climate_variables.items()):
        mean_values = np.mean(data, axis=0)
        axes[i].plot(mean_values)
        axes[i].set_title(f"Mean {name} along time dim")
        axes[i].set_ylabel("Mean value")

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "mean_values_plot.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Plot of mean values saved to {plot_path}")


def plot_time_series(climate_matrix: np.ndarray,
                     n_days: int,
                     feature_names: List[str] = None,
                     n_samples: int = 5,
                     out_dir: str = 'plots') -> List[str]:
    """Построение и сохранение графиков временных рядов.
    Построение для сэмпла.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    if climate_matrix is None:
        raise ValueError("climate_matrix is None")

    total_cols = climate_matrix.shape[1]
    if total_cols % n_days != 0:
        raise ValueError("climate_matrix columns not divisible by n_days")

    n_features = total_cols // n_days

    if feature_names is None:
        feature_names = [f'f{i}' for i in range(n_features)]

    # seq = (n_samples, n_days, n_features)
    seq = np.stack([climate_matrix[:, i*n_days:(i+1)*n_days] for i in range(n_features)], axis=-1)

    # index of samples
    N = seq.shape[0]
    samples_idx = np.linspace(0, max(0, N-1), min(n_samples, N), dtype=int)

    for idx in samples_idx:
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 3*n_features), sharex=True)
        if n_features == 1:
            axes = [axes]

        for i in range(n_features):
            axes[i].plot(np.arange(n_days), seq[idx, :, i])
            axes[i].set_ylabel(feature_names[i])
            axes[i].grid(True)

        axes[-1].set_xlabel('day')
        fig.suptitle(f'Sample {idx} time series')
        out_path = os.path.join(out_dir, f'sample_{idx}_timeseries.png')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
        print(f"Saved plot {out_path}")

    return saved


def plot_target_distribution(target: np.ndarray, out_dir: str = 'plots', bins: int = 30) -> str:
    """
    Гистограмма распределения целевой переменной.
    Сохраняет в папку out_dir и возвращает путь к файлу.
    """
    os.makedirs(out_dir, exist_ok=True)  # создаём папку, если нет

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(target, bins=bins)
    ax.set_xlabel("target")
    ax.set_ylabel("count")
    ax.grid(True)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "target_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {out_path}")
    return out_path


def prepare_feature_for_models(
    features_df: pd.DataFrame,
    climate_matrix: np.ndarray,
    n_days: int,
    static_exclude_cols: Optional[List[str]] = None,
    target_col: str = 'count',
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
    datetime_col: Optional[str] = None,
    cat_one_hot_threshold: int = 100,
    scale_static: bool = True,
    ignored_features: Optional[List[str]] = None,
    log_population: bool = True,
    add_missing_indicators: bool = True,
    allow_ordinal_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Return dict:
      x_dyn_train/val/test (n_samples, n_days, n_channels),
      x_stat_train/val/test (n_samples, n_static_features),
      y_train/val/test,
      num_scaler, ohe_encoders, ord_encoders, dyn_scaler, indices.
    """

    n_samples = climate_matrix.shape[0]
    if target_col not in features_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in features_df")

    if ignored_features is None:
        ignored_features = DEFAULT_IGNORED_FEATURES

    print("start prepare data...")

    coord_mask_df = _make_coord_mask_df(features_df, datetime_col)
    x_stat_df_full, num_cols, cat_cols, ignored_present = _build_static_frame(
        features_df=features_df,
        target_col=target_col,
        static_exclude_cols=static_exclude_cols,
        ignored_features=list(ignored_features),
        datetime_col=datetime_col,
        log_population=log_population,
    )

    print(f"size of cat_cols: {len(cat_cols)}")
    print(f"size of num_cols: {len(num_cols)}")

    print("start split data...")
    train_idx, val_idx, test_idx = _split_indices_by_date(
        features_df=features_df,
        datetime_col=datetime_col,
        train_end=train_end,
        val_end=val_end,
    )

    if not coord_mask_df.empty:
        coord_mask_train_df = coord_mask_df.iloc[train_idx].reset_index(drop=True)
        coord_mask_val_df = coord_mask_df.iloc[val_idx].reset_index(drop=True)
        coord_mask_test_df = coord_mask_df.iloc[test_idx].reset_index(drop=True)
    else:
        coord_mask_train_df = coord_mask_val_df = coord_mask_test_df = pd.DataFrame()


    # split dynamic data
    print("split dynamic data...")
    x_dyn_all, n_channels = _reshape_dynamic_features(climate_matrix, n_days)

    # scale dynamics
    print("scale dynamics...")
    x_dyn_all, dyn_scaler = _fit_dynamic_scaler(x_dyn_all, train_idx)
    x_dyn_train = x_dyn_all[train_idx]
    x_dyn_val = x_dyn_all[val_idx]
    x_dyn_test = x_dyn_all[test_idx]

    # split static data
    print("split static  data...")
    x_stat_df_full = x_stat_df_full.reset_index(drop=True)
    x_stat_train_df = x_stat_df_full.loc[train_idx].reset_index(drop=True)
    x_stat_val_df = x_stat_df_full.loc[val_idx].reset_index(drop=True)
    x_stat_test_df = x_stat_df_full.loc[test_idx].reset_index(drop=True)

    # print(f"Train start year: {x_stat_train_df['year'].min()}")
    # print(f"Val start year: {x_stat_val_df['year'].min()}")
    # print(f"Test start year: {x_stat_test_df['year'].min()}")


    # scaling numeric columns
    print("scaling numeric columns...")
    (
        x_num_train,
        x_num_val,
        x_num_test,
        num_scaler,
        num_fill_values,
        numeric_missing_indicator_cols,
    ) = _fit_transform_numeric_static(
        x_stat_train_df,
        x_stat_val_df,
        x_stat_test_df,
        num_cols,
        scale_static=scale_static,
        add_missing_indicators=add_missing_indicators,
    )

    # one-hot and label for categories columns
    print("one-hot and label for categories columns...")
    (
        x_cat_train,
        x_cat_val,
        x_cat_test,
        ohe_encoders,
        ord_encoders,
        ohe_cols,
        label_cols,
        cat_feature_names,
    ) = _fit_transform_categorical_static(
        x_stat_train_df,
        x_stat_val_df,
        x_stat_test_df,
        cat_cols,
        cat_one_hot_threshold=cat_one_hot_threshold,
        allow_ordinal_fallback=allow_ordinal_fallback,
    )


    # STATIC
    x_stat_train = np.hstack([x_num_train, x_cat_train]).astype(np.float32)
    x_stat_val = np.hstack([x_num_val, x_cat_val]).astype(np.float32)
    x_stat_test = np.hstack([x_num_test, x_cat_test]).astype(np.float32)

    # target splits
    y = features_df[target_col].values.copy()
    y_train = y[train_idx].astype(np.float32)
    y_val = y[val_idx].astype(np.float32)
    y_test = y[test_idx].astype(np.float32)

    # Names
    static_feature_names = []
    static_feature_names.extend(num_cols)
    static_feature_names.extend(numeric_missing_indicator_cols)
    static_feature_names.extend(cat_feature_names)

    dyn_channel_names = [f"dyn_ch_{i}" for i in range(n_channels)]
    dyn_expanded = []
    for day in range(n_days):
        for ch in range(n_channels):
            dyn_expanded.append(f"day{day:02d}_ch{ch}")

    features_mask = {
        'static': static_feature_names,
        'dyn_channels': dyn_channel_names,
        'dyn_expanded': dyn_expanded,
        'n_days': n_days,
        'n_channels': n_channels
    }


    out = {
        'x_dyn_train': x_dyn_train, 'x_dyn_val': x_dyn_val, 'x_dyn_test': x_dyn_test,
        'x_stat_train': x_stat_train, 'x_stat_val': x_stat_val, 'x_stat_test': x_stat_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'num_cols': num_cols, 'ohe_cols': ohe_cols, 'label_cols': label_cols,
        'num_scaler': num_scaler, 'ohe_encoders': ohe_encoders, 'ord_encoders': ord_encoders,
        'dyn_scaler': dyn_scaler,
        'num_fill_values': num_fill_values,
        'numeric_missing_indicator_cols': numeric_missing_indicator_cols,
        'scale_static': scale_static,
        'log_population': log_population,
        'add_missing_indicators': add_missing_indicators,
        'cat_one_hot_threshold': cat_one_hot_threshold,
        'allow_ordinal_fallback': allow_ordinal_fallback,
        'datetime_col': datetime_col,
        'target_col': target_col,
        'static_exclude_cols': static_exclude_cols,
        'preprocessing_version': PREPROCESSING_VERSION,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
        'n_days': n_days, 'n_channels': n_channels,

        # mask before scaling and split masks
        'coord_mask_df': coord_mask_df,
        'coord_mask_train_df': coord_mask_train_df,
        'coord_mask_val_df': coord_mask_val_df,
        'coord_mask_test_df': coord_mask_test_df,
        'features_mask': features_mask,
        'ignored_features': ignored_present
    }
    return out


if __name__ == '__main__':

    # First, load config to get paths
    default_config_path = Path('configs/features_config_30d_nn.yaml')

    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Read paths from config
    data_path = Path(config.get('train_features_parquet_path', 'data/saved_features/train_features_nn_30d.parquet'))
    config_path = Path(config.get('config_path', 'configs/features_config_30d_nn.yaml'))

    ensure_feature_files(data_path, config_path)

    N_DAYS = config['generate_climate_params']['n_days']
    print(f"Attempting to load data from '{data_path}'...")

    features_df, climate_features = load_data(str(data_path))

    print("Data loaded successfully.")
    print("Data shapes:", features_df.shape, climate_features.shape)
    print(features_df.head())
    print("Climate Features Matrix:")
    print(climate_features[:5, :5])
    print(climate_features.shape)
    print(features_df.info())
    print(features_df[:5])

    t2m = climate_features[:, :N_DAYS]
    d2m = climate_features[:, N_DAYS:2*N_DAYS]
    tp = climate_features[:, 2*N_DAYS:3*N_DAYS]
    stl1 = climate_features[:, 3*N_DAYS:4*N_DAYS]

    target = features_df['count'].values
    print(np.unique(target))

    print("target mean(class balance):", np.mean(target))

    output_dir = config.get('output_plots_dir', 'outputs/plots')
    os.makedirs(output_dir, exist_ok=True)

    climate_variables = {
        "t2m": t2m,
        "d2m": d2m,
        "tp": tp,
        "stl1": stl1,
    }

    plot_and_save_visualizations(climate_variables, output_dir)

    print("target mean(class balance):", np.mean(target))
    plot_target_distribution(target, out_dir=output_dir)

    feature_names = ['t2m', 'd2m', 'tp', 'stl1']
    plot_time_series(climate_features, n_days=N_DAYS, feature_names=feature_names, n_samples=4, out_dir=output_dir)

    static_exclude_cols=None
    target_col = 'count'
    nn_preprocessing_cfg = config.get('nn_preprocessing') or {}

    prepared = prepare_feature_for_models(
        features_df=features_df,
        climate_matrix=climate_features,
        n_days=N_DAYS,
        static_exclude_cols=static_exclude_cols,
        target_col=target_col,
        train_end=config['train_end'],
        val_end=config['val_end'],
        datetime_col='datetime',
        cat_one_hot_threshold=int(nn_preprocessing_cfg.get('cat_one_hot_threshold', 200)),
        scale_static=bool(nn_preprocessing_cfg.get('scale_static', True)),
        ignored_features=nn_preprocessing_cfg.get('ignored_features', DEFAULT_IGNORED_FEATURES),
        log_population=bool(nn_preprocessing_cfg.get('log_population', True)),
        add_missing_indicators=bool(nn_preprocessing_cfg.get('add_missing_indicators', True)),
        allow_ordinal_fallback=bool(nn_preprocessing_cfg.get('allow_ordinal_fallback', False)),
    )

    print("Shapes:")
    print("x_dyn_train:", prepared['x_dyn_train'].shape)
    print("x_dyn_val:  ", prepared['x_dyn_val'].shape)
    print("x_dyn_test: ", prepared['x_dyn_test'].shape)
    print("x_stat_train:", prepared['x_stat_train'].shape)
    print("x_stat_val:  ", prepared['x_stat_val'].shape)
    print("x_stat_test: ", prepared['x_stat_test'].shape)
    print("y_train unique:", len(np.unique(prepared['y_train'])), " y_train mean", prepared['y_train'].mean(), " y_train shape:", prepared['y_train'].shape)
    print("y_val unique:", len(np.unique(prepared['y_val'])), " y_val mean", prepared['y_val'].mean(), " y_val shape:", prepared['y_val'].shape)
    print("y_test unique:", len(np.unique(prepared['y_test'])), " y_test mean", prepared['y_test'].mean(), " y_test shape:", prepared['y_test'].shape)

    output_dir = Path(config.get('output_train_data_dir', 'data/saved_features/nn_train_data'))
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / 'prepared_data.npz',
        x_dyn_train=prepared['x_dyn_train'],
        x_dyn_val=prepared['x_dyn_val'],
        x_dyn_test=prepared['x_dyn_test'],
        x_stat_train=prepared['x_stat_train'],
        x_stat_val=prepared['x_stat_val'],
        x_stat_test=prepared['x_stat_test'],
        y_train=prepared['y_train'],
        y_val=prepared['y_val'],
        y_test=prepared['y_test']
    )
    print(f"Saved arrays to {output_dir / 'prepared_data.npz'}")


    mask_out_dir = output_dir
    prepared['coord_mask_df'].to_parquet(mask_out_dir / 'coord_mask_before_split.parquet', index=False)
    prepared['coord_mask_train_df'].to_parquet(mask_out_dir / 'coord_mask_train.parquet', index=False)
    prepared['coord_mask_val_df'].to_parquet(mask_out_dir / 'coord_mask_val.parquet', index=False)
    prepared['coord_mask_test_df'].to_parquet(mask_out_dir / 'coord_mask_test.parquet', index=False)
    print(f"Saved coord_mask parquet files to {mask_out_dir}")


    features_mask = prepared['features_mask']
    with open(mask_out_dir / 'features_mask.json', 'w', encoding='utf8') as fh:
        json.dump(features_mask, fh, ensure_ascii=False, indent=2)
    pd.Series(features_mask.get('static', [])).to_csv(mask_out_dir / 'features_mask_static.csv', index=False, header=False)
    pd.Series(features_mask.get('dyn_expanded', [])).to_csv(mask_out_dir / 'features_mask_dyn_expanded.csv', index=False, header=False)


    encoders = {
        'num_scaler': prepared.get('num_scaler'),
        'ohe_encoders': prepared.get('ohe_encoders'),
        'ord_encoders': prepared.get('ord_encoders'),
        'dyn_scaler': prepared.get('dyn_scaler'),
        'num_fill_values': prepared.get('num_fill_values'),
        'numeric_missing_indicator_cols': prepared.get('numeric_missing_indicator_cols'),
        'num_cols': prepared.get('num_cols'),
        'ohe_cols': prepared.get('ohe_cols'),
        'label_cols': prepared.get('label_cols'),
        'n_days': prepared.get('n_days'),
        'n_channels': prepared.get('n_channels'),
        'scale_static': prepared.get('scale_static'),
        'log_population': prepared.get('log_population'),
        'add_missing_indicators': prepared.get('add_missing_indicators'),
        'cat_one_hot_threshold': prepared.get('cat_one_hot_threshold'),
        'allow_ordinal_fallback': prepared.get('allow_ordinal_fallback'),
        'datetime_col': prepared.get('datetime_col'),
        'target_col': prepared.get('target_col'),
        'static_exclude_cols': prepared.get('static_exclude_cols'),
        'ignored_features': prepared.get('ignored_features'),
        'preprocessing_version': prepared.get('preprocessing_version'),
        'train_idx': prepared.get('train_idx'),
        'val_idx': prepared.get('val_idx'),
        'test_idx': prepared.get('test_idx'),
        # 'mask_columns': list(prepared.get('mask_df').columns) if not prepared.get('mask_df').empty else []
        'coord_mask_columns': list(prepared.get('coord_mask_df').columns) if not prepared.get('coord_mask_df').empty else [],
        'features_mask': prepared.get('features_mask', {}),
    }
    joblib.dump(encoders, output_dir / 'encoders_meta.joblib', compress=('gzip', 3))
    print(f"Saved encoders/meta to {output_dir / 'encoders_meta.joblib'}")
