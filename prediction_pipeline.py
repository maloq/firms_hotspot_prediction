import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

try:  # geopandas is optional; plots fall back without borders
    import geopandas as gpd
except ImportError:  # pragma: no cover - geopandas optional
    gpd = None

from make_nn_train_data import load_data, transform_features_with_metadata
from src.neural_net.models import SequenceStaticLightningModule
from src.neural_net.prediction_features_builder import generate_prediction_features
from src.neural_net.train_nn import predict_probs
from src.target_generation.create_grid_target import load_modis_data_for_period
from src.utils.prediction_adjustments import (
    DEFAULT_DEPLOY_PRIOR,
    DEFAULT_TRAIN_PRIOR,
    adjust_probabilities_for_prior,
)
from src.utils.prediction_visuals import (
    plot_feature_map,
    plot_modis_fires,
    plot_prediction_map,
    save_prediction_as_netcdf,
)

DEFAULT_IGNORED_FEATURES = ["datetime", "day", "latitude", "longitude", "year"]
PREDICTION_COL = "prediction"

save_feature_maps = True
print_features_mean = False
MODIS_PLOT_DAYS_WINDOW = 5
TRAIN_PRIOR = DEFAULT_TRAIN_PRIOR
DEPLOY_PRIOR = DEFAULT_DEPLOY_PRIOR
ENABLE_PRIOR_CORRECTION = True


def _series_from_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Return the first column with the given name as a Series."""

    column = df.loc[:, column_name]
    if isinstance(column, pd.DataFrame):
        return column.iloc[:, 0]
    return column



def _extract_feature_columns(
    features_df: pd.DataFrame,
    candidate_columns: Sequence[str] | None,
    lat_col: str,
    lon_col: str,
) -> list[str]:
    """Determine which feature columns to visualise on maps."""

    if candidate_columns:
        filtered = [col for col in candidate_columns if col in features_df.columns]
    else:
        filtered = []

    if not filtered:
        excluded = set(DEFAULT_IGNORED_FEATURES + [lat_col, lon_col, PREDICTION_COL])
        filtered = [
            col
            for col in features_df.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(features_df[col])
        ]

    seen: set[str] = set()
    ordered_unique: list[str] = []
    for col in filtered:
        if col in seen:
            continue
        seen.add(col)
        ordered_unique.append(col)
    return ordered_unique


def _save_feature_maps(
    features_df: pd.DataFrame,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    lat_col: str,
    lon_col: str,
    plot_dir: str,
    date_str: str,
    borders_data: "gpd.GeoDataFrame | None",
    feature_columns: Sequence[str] | None,
) -> None:
    """Persist feature maps for the provided day to disk."""

    feature_cols = _extract_feature_columns(features_df, feature_columns, lat_col, lon_col)
    if not feature_cols:
        print("Warning: No numeric feature columns found for feature map plotting.")
        return

    feature_plot_dir = os.path.join(plot_dir, "feature_maps")
    os.makedirs(feature_plot_dir, exist_ok=True)

    lat_index = pd.Index(lat_coords, name=lat_col)
    lon_index = pd.Index(lon_coords, name=lon_col)

    lat_series = _series_from_column(features_df, lat_col)
    lon_series = _series_from_column(features_df, lon_col)

    for feature_name in feature_cols:
        column_series = _series_from_column(features_df, feature_name)
        if not pd.api.types.is_numeric_dtype(column_series):
            continue

        subset = pd.DataFrame({
            lat_col: lat_series,
            lon_col: lon_series,
            feature_name: column_series,
        }).dropna(subset=[feature_name])
        if subset.empty:
            continue

        pivot = (
            subset.groupby([lat_col, lon_col])[feature_name]
            .mean()
            .unstack(fill_value=np.nan)
            .reindex(index=lat_index, columns=lon_index)
        )

        feature_grid = pivot.to_numpy(dtype=float)
        if np.isnan(feature_grid).all():
            continue

        feature_filename_safe = feature_name.replace("/", "-").replace(" ", "_")
        save_path = os.path.join(feature_plot_dir, f"{feature_filename_safe}_{date_str}.png")
        feature_title = f"{feature_name} ({date_str})"
        plot_feature_map(
            lat_coords=lat_coords,
            lon_coords=lon_coords,
            feature_grid=feature_grid,
            feature_name=feature_name,
            title=feature_title,
            save_path=save_path,
            borders_gdf=borders_data,
        )


@dataclass
class NNForecastDataset:
    features: pd.DataFrame
    x_dyn: np.ndarray
    x_stat: np.ndarray
    n_days: int
    n_channels: int
    min_date: pd.Timestamp
    max_date: pd.Timestamp

    @property
    def available_dates(self) -> np.ndarray:
        return self.features["datetime"].to_numpy()


def resolve_coordinate_columns(df: pd.DataFrame) -> Tuple[str, str]:
    lat_candidates = ("lat_rounded", "latitude", "lat")
    lon_candidates = ("lon_rounded", "longitude", "lon")

    lat_col = next((col for col in lat_candidates if col in df.columns), None)
    lon_col = next((col for col in lon_candidates if col in df.columns), None)

    if lat_col is None or lon_col is None:
        raise KeyError(
            "Latitude/longitude columns not found in features data. Expected one of "
            "'lat_rounded', 'latitude', 'lat' and 'lon_rounded', 'longitude', 'lon'."
        )

    return lat_col, lon_col



def build_prediction_grid(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    value_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert row-wise predictions into a lat/lon grid."""

    values = df[[lat_col, lon_col, value_col]].to_numpy(dtype=float)

    lat_values = np.unique(values[:, 0])
    lon_values = np.unique(values[:, 1])
    lat_values.sort()
    lon_values.sort()

    grid = np.full((lat_values.size, lon_values.size), np.nan, dtype=np.float32)
    lat_index = {lat: i for i, lat in enumerate(lat_values)}
    lon_index = {lon: j for j, lon in enumerate(lon_values)}

    for lat_value, lon_value, val in values:
        i = lat_index[lat_value]
        j = lon_index[lon_value]
        grid[i, j] = val

    return lat_values, lon_values, grid


def _prepare_output_dirs(base_dir: str, date_str: str) -> dict[str, str]:
    """Create per-day output subdirectories and return their paths."""

    day_dir = os.path.join(base_dir, date_str)
    paths = {
        "root": day_dir,
        "csv": os.path.join(day_dir, "csv"),
        "netcdf": os.path.join(day_dir, "netcdf"),
        "plots": os.path.join(day_dir, "plots"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def parse_start_date(value: Optional[str], default: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return default.normalize()

    for fmt in ("%d-%m-%y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return pd.to_datetime(value, format=fmt).normalize()
        except ValueError:
            continue

    return pd.to_datetime(value).normalize()


def load_trained_nn_model(
    model_path: str,
    device: torch.device,
    nn_model_cfg: Optional[dict] = None,
) -> SequenceStaticLightningModule:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Neural network checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    hyperparams = checkpoint.get("hyper_parameters") or {}
    saved_arch = hyperparams.get("model_name", "lstm_mlp")
    saved_model_config = hyperparams.get("model_config") or {}

    if nn_model_cfg:
        requested_arch = nn_model_cfg.get("architecture")
        if requested_arch and requested_arch != saved_arch:
            raise ValueError(
                f"Configured architecture '{requested_arch}' does not match checkpoint architecture '{saved_arch}'."
            )

    lightning_defaults = {
        key: hyperparams.get(key)
        for key in ("learning_rate", "decay_rate", "decay_steps", "l2", "clip_gradient_norm")
        if key in hyperparams
    }

    del checkpoint

    model = SequenceStaticLightningModule.load_from_checkpoint(
        model_path,
        map_location=device,
        model_name=saved_arch,
        model_config=saved_model_config,
        **lightning_defaults,
    )
    print(f"Loaded NN checkpoint '{os.path.basename(model_path)}' with architecture '{saved_arch}'.")
    model.eval()
    model.to(device)
    return model


def validate_requested_date(dataset: NNForecastDataset, target_date: pd.Timestamp) -> None:
    if target_date < dataset.min_date or target_date > dataset.max_date:
        raise ValueError(
            f"Requested prediction date {target_date.date()} lies outside available range "
            f"{dataset.min_date.date()} – {dataset.max_date.date()} in the precomputed dataset."
        )

    history_start = target_date - pd.Timedelta(days=dataset.n_days - 1)
    if history_start < dataset.min_date:
        available_dates = pd.to_datetime(dataset.available_dates).astype("datetime64[ns]")
        unique_days = pd.Index(available_dates).normalize().unique()
        if unique_days.size == 1 and unique_days[0] == target_date.normalize():
            print(
                "Warning: Forecast dataset contains only the target day; "
                "assuming climate history is embedded in provided features."
            )
        else:
            raise ValueError(
                "Insufficient temporal coverage for requested date. Prediction requires climate data "
                f"back to {history_start.date()}, but dataset starts at {dataset.min_date.date()}."
            )


def _resolve_preprocessing_meta_path(config: dict) -> str:
    nn_cfg = config.get("nn_preprocessing", {}) or {}
    candidates = [
        nn_cfg.get("encoders_meta_path"),
        config.get("nn_encoders_meta_path"),
    ]

    output_train_data_dir = config.get("output_train_data_dir")
    if output_train_data_dir:
        candidates.append(os.path.join(output_train_data_dir, "encoders_meta.joblib"))

    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate

    checked = [os.path.abspath(path) for path in candidates if path]
    raise FileNotFoundError(
        "NN preprocessing metadata not found. Run make_nn_train_data.py after training-data "
        "generation, or set nn_preprocessing.encoders_meta_path. Checked: "
        f"{checked}"
    )


def _model_hparam(model: SequenceStaticLightningModule, key: str, default=None):
    hparams = getattr(model, "hparams", {})
    if hasattr(hparams, "get"):
        return hparams.get(key, default)
    return getattr(hparams, key, default)


def validate_dataset_model_compatibility(
    dataset: NNForecastDataset,
    model: SequenceStaticLightningModule,
) -> None:
    model_config = _model_hparam(model, "model_config", {}) or {}
    expected_seq_len = model_config.get("seq_len")
    expected_channels = model_config.get("n_channels")
    expected_static = model_config.get("n_static")

    mismatches = []
    if expected_seq_len is not None and int(expected_seq_len) != dataset.n_days:
        mismatches.append(f"seq_len checkpoint={expected_seq_len}, data={dataset.n_days}")
    if expected_channels is not None and int(expected_channels) != dataset.n_channels:
        mismatches.append(
            f"n_channels checkpoint={expected_channels}, data={dataset.n_channels}"
        )
    if expected_static is not None and int(expected_static) != dataset.x_stat.shape[1]:
        mismatches.append(
            f"n_static checkpoint={expected_static}, data={dataset.x_stat.shape[1]}"
        )

    if mismatches:
        raise ValueError(
            "NN checkpoint and preprocessing metadata are incompatible: "
            + "; ".join(mismatches)
            + ". Use the encoders_meta.joblib produced with the same training run as the checkpoint."
        )


def load_prepared_dataset(config: dict, dataset_path: Optional[str] = None) -> NNForecastDataset:
    nn_cfg = config.get("nn_preprocessing", {})

    resolved_path = (
        dataset_path
        or nn_cfg.get("features_parquet")
        or config.get("nn_features_parquet")
        or config.get("prepared_prediction_features_path")
        or "data/saved_features/train_features_nn_30d.parquet"
    )

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"NN features parquet not found: {resolved_path}")

    features_df, climate_matrix = load_data(resolved_path)
    if features_df is None or climate_matrix is None:
        raise RuntimeError(
            f"Failed to load features or climate matrix from '{resolved_path}'. Ensure both parquet and NPZ exist."
        )

    features_df = features_df.reset_index(drop=True)

    datetime_col = nn_cfg.get("datetime_col", "datetime")
    if datetime_col not in features_df.columns:
        raise KeyError(f"Column '{datetime_col}' not found in features data.")

    datetime_series = pd.to_datetime(features_df[datetime_col], errors="coerce")
    if datetime_series.isna().any():
        raise ValueError("Some datetime values could not be parsed in the NN feature dataset.")
    features_df["datetime"] = datetime_series.dt.normalize()

    target_col = nn_cfg.get("target_col", config.get("target_col", "count"))
    meta_path = _resolve_preprocessing_meta_path(config)
    metadata = joblib.load(meta_path)
    print(f"Loaded NN preprocessing metadata from '{meta_path}'.")

    prepared = transform_features_with_metadata(
        features_df=features_df,
        climate_matrix=climate_matrix,
        metadata=metadata,
        datetime_col="datetime",
        target_col=target_col,
    )

    min_date = features_df["datetime"].min()
    max_date = features_df["datetime"].max()

    return NNForecastDataset(
        features=features_df,
        x_dyn=prepared["x_dyn"],
        x_stat=prepared["x_stat"],
        n_days=prepared["n_days"],
        n_channels=prepared["n_channels"],
        min_date=min_date,
        max_date=max_date,
    )


def make_n_day_forecast(
    forecast_horizon: int,
    config: dict,
    model_path: str,
    output_base_dir: str,
    start_date: Optional[str] = None,
    dataset_path: Optional[str] = None,
    batch_size: int = 1024,
    train_prior: float = TRAIN_PRIOR,
    deploy_prior: float = DEPLOY_PRIOR,
    enable_prior_correction: bool = ENABLE_PRIOR_CORRECTION,
) -> None:
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1 day.")

    borders_data = None
    country_shapes_path = config.get("country_shapes_path")
    if country_shapes_path:
        if gpd is None:
            print(
                "Warning: country_shapes_path provided but geopandas is unavailable. "
                "Borders will not be rendered."
            )
        else:
            try:
                borders_data = gpd.read_file(country_shapes_path)
                print("Loaded border data for plotting.")
            except Exception as exc:  # pragma: no cover - plotting fallback
                print(
                    f"Warning: Could not load border data from {country_shapes_path}: {exc}. "
                    "Border overlays will be skipped."
                )

    selected_columns = None
    selected_cols_path = config.get("selected_feature_columns_path")
    if selected_cols_path and os.path.exists(selected_cols_path):
        try:
            with open(selected_cols_path, "r", encoding="utf-8") as fh:
                selected_columns = [line.strip() for line in fh if line.strip()]
        except OSError as exc:
            print(f"Warning: Could not read feature columns from {selected_cols_path}: {exc}")

    default_start_reference = config.get("prediction_default_date") or config.get("target_end_date")
    if default_start_reference is None:
        raise KeyError(
            "Config must provide 'target_end_date' (or prediction_default_date) to seed NN prediction start date."
        )
    default_start_reference = pd.to_datetime(default_start_reference).normalize()
    requested_start_dt = parse_start_date(start_date, default_start_reference)
    forecast_dates = [
        (requested_start_dt + pd.Timedelta(days=offset)).normalize()
        for offset in range(forecast_horizon)
    ]

    features_output_dir = config.get("prepared_prediction_features_path")
    if features_output_dir:
        if features_output_dir.endswith(".parquet"):
            features_output_dir = os.path.dirname(features_output_dir)
        features_output_dir = os.path.abspath(features_output_dir)
    else:
        features_output_dir = os.path.abspath(os.path.join(output_base_dir, "prepared_features"))

    if dataset_path:
        dataset_path_to_use = os.path.abspath(dataset_path)
        if not os.path.exists(dataset_path_to_use):
            raise FileNotFoundError(f"Specified features parquet not found: {dataset_path_to_use}")
    else:
        generated = generate_prediction_features(
            config=config,
            forecast_dates=forecast_dates,
            output_dir=features_output_dir,
            borders_gdf=borders_data,
        )
        dataset_path_to_use = generated.features_path
        print(f"Generated NN prediction features at {dataset_path_to_use}")

    dataset = load_prepared_dataset(config, dataset_path_to_use)
    lat_col, lon_col = resolve_coordinate_columns(dataset.features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_nn_model(model_path, device, config.get("nn_model"))
    validate_dataset_model_compatibility(dataset, model)

    os.makedirs(output_base_dir, exist_ok=True)

    date_array = dataset.features["datetime"].to_numpy()

    if enable_prior_correction:
        print(
            f"Prior correction enabled: train_prior={train_prior}, deploy_prior={deploy_prior}"
        )
    else:
        print("Prior correction disabled for NN pipeline.")

    for current_date in forecast_dates:
        validate_requested_date(dataset, current_date)

        indices = np.flatnonzero(date_array == current_date.to_datetime64())
        if indices.size == 0:
            raise ValueError(
                f"No precomputed features available for prediction date {current_date.date()} in dataset."
            )

        date_str = current_date.strftime("%Y-%m-%d")
        x_dyn = dataset.x_dyn[indices]
        x_stat = dataset.x_stat[indices]

        probs = predict_probs(model, x_dyn, x_stat, batch_size=batch_size, device=device)

        if enable_prior_correction:
            probs = adjust_probabilities_for_prior(
                probs,
                train_prior=train_prior,
                deploy_prior=deploy_prior,
            )

        day_features_df = dataset.features.loc[indices].copy()
        if print_features_mean:
            means_dict = day_features_df.select_dtypes(include=["number"]).mean().to_dict()
            print(f"Feature means for {date_str}: {means_dict}")

        day_df = day_features_df[["datetime", lat_col, lon_col]].copy()
        day_df[PREDICTION_COL] = probs.astype(np.float32)

        lat_coords, lon_coords, grid = build_prediction_grid(day_df, lat_col, lon_col, PREDICTION_COL)

        output_paths = _prepare_output_dirs(output_base_dir, date_str)

        csv_path = os.path.join(output_paths["csv"], f"predictions_{date_str}.csv")
        day_df.to_csv(csv_path, index=False)

        netcdf_path = os.path.join(output_paths["netcdf"], f"forecast_nn_{date_str}.nc")
        try:
            save_prediction_as_netcdf(
                lat_coords=lat_coords,
                lon_coords=lon_coords,
                predictions=grid,
                date_timestamp=current_date,
                save_path=netcdf_path,
                prediction_type="nn",
            )
        except RuntimeError as exc:
            print(f"Warning: {exc}")

        plot_dir = output_paths["plots"]

        common_filename_base = f"forecast_nn_{date_str}"
        plot_path = os.path.join(plot_dir, f"{common_filename_base}.png")
        title = f"NN Fire Probability Forecast for {date_str}"
        plot_prediction_map(
            lat_coords=lat_coords,
            lon_coords=lon_coords,
            predictions=grid,
            title=title,
            save_path=plot_path,
            borders_gdf=borders_data,
        )

        try:
            modis_data_dir = config.get("modis_data_path", "data/modis/")
            modis_countries = config.get("prediction_countries", config.get("modis_countries"))

            start_date_modis = (current_date - pd.Timedelta(days=MODIS_PLOT_DAYS_WINDOW)).date()
            end_date_modis = (current_date + pd.Timedelta(days=MODIS_PLOT_DAYS_WINDOW)).date()

            modis_fire_data = load_modis_data_for_period(
                data_dir=modis_data_dir,
                start_date=start_date_modis,
                end_date=end_date_modis,
                countries=modis_countries,
            )

            if not modis_fire_data.empty:
                modis_fire_data = modis_fire_data[
                    (modis_fire_data["latitude"] >= lat_coords.min())
                    & (modis_fire_data["latitude"] <= lat_coords.max())
                    & (modis_fire_data["longitude"] >= lon_coords.min())
                    & (modis_fire_data["longitude"] <= lon_coords.max())
                ]

            if modis_fire_data.empty:
                print(
                    f"Warning: No MODIS fire data available for {date_str} (±{MODIS_PLOT_DAYS_WINDOW} days). "
                    "Skipping MODIS plot."
                )
            else:
                modis_plot_path = os.path.join(plot_dir, f"modis_fires_{date_str}.png")
                modis_title = f"MODIS Fires (±{MODIS_PLOT_DAYS_WINDOW} days around {date_str})"
                plot_modis_fires(
                    lat_coords=lat_coords,
                    lon_coords=lon_coords,
                    modis_data=modis_fire_data,
                    prediction_date=current_date,
                    n_days=MODIS_PLOT_DAYS_WINDOW,
                    title=modis_title,
                    save_path=modis_plot_path,
                    borders_gdf=borders_data,
                )
        except Exception as exc:  # pragma: no cover - plotting fallback
            print(f"Warning: Could not load or plot MODIS fire data for {date_str}: {exc}")

        if save_feature_maps:
            _save_feature_maps(
                features_df=day_features_df,
                lat_coords=lat_coords,
                lon_coords=lon_coords,
                lat_col=lat_col,
                lon_col=lon_col,
                plot_dir=plot_dir,
                date_str=date_str,
                borders_data=borders_data,
                feature_columns=selected_columns,
            )

        print(f"Saved NN predictions for {date_str}: {len(day_df)} grid cells -> {csv_path}")

    print(
        f"\nNN forecast finished. Available NN feature range: "
        f"{dataset.min_date.date()} – {dataset.max_date.date()}."
    )
    print(f"Outputs stored in {output_base_dir}")


def main() -> None:
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser(description="Run neural-net fire forecast over precomputed features.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/features_config_30d_nn.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of consecutive days to predict, starting from start-date.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/forecast_nn",
        help="Directory to store prediction artefacts.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (dd-mm-yy or yyyy-mm-dd). Defaults to latest available.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help=(
            "Override path to precomputed NN feature parquet. "
            "When omitted, the pipeline generates prediction features for the requested dates."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Override inference batch size.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    model_path = config.get("nn_model_path") or config.get("model_path")
    if not model_path:
        raise KeyError("Config must provide 'nn_model_path' pointing to a trained neural-net checkpoint.")

    batch_size = (
        args.batch_size
        or config.get("nn_batch_size")
        or config.get("nn_preprocessing", {}).get("batch_size")
        or 1024
    )

    train_prior = float(config.get("train_prior", TRAIN_PRIOR))
    deploy_prior = float(config.get("deploy_prior", DEPLOY_PRIOR))
    enable_prior_correction = config.get(
        "enable_prior_correction",
        ENABLE_PRIOR_CORRECTION,
    )
    if isinstance(enable_prior_correction, str):
        enable_prior_correction = enable_prior_correction.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    else:
        enable_prior_correction = bool(enable_prior_correction)

    make_n_day_forecast(
        forecast_horizon=args.days,
        config=config,
        model_path=model_path,
        output_base_dir=args.output,
        start_date=args.start_date,
        dataset_path=args.features_path,
        batch_size=int(batch_size),
        train_prior=train_prior,
        deploy_prior=deploy_prior,
        enable_prior_correction=enable_prior_correction,
    )


if __name__ == "__main__":
    main()
