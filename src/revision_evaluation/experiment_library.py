#!/usr/bin/env python3
"""Organize revision experiment outputs into a reader-friendly experiment tree.

The original revision run intentionally produced many flat CSV/Markdown/plot
artifacts. This script turns those outputs into per-experiment folders with:

  description.md
  analysis.md
  tables/*.csv          (compact presentation tables, when present)
  plots/png/*.png       (when present)
  plots/pdf/*.pdf       (when present)
  artifacts/*          (raw JSONL tables, manifests, symlinks to models/data, when present)

It archives wide raw CSV tables as JSONL.GZ artifacts so that all remaining CSV
files are small, readable tables.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .artifacts import prune_empty_dirs


ROOT = Path("results/revision_experiments_complete")
EXPERIMENTS = ROOT / "experiments"
SHARED = ROOT / "shared_artifacts"
RAW_JSONL = SHARED / "raw_tables_jsonl"
RAW_SCHEMAS = SHARED / "raw_table_schemas"
SOURCE_MARKDOWN = SHARED / "source_markdown"
MAX_PRESENTATION_TABLE_COLUMNS = 8
NEURAL_MODEL_MARKERS = ("MLP", "Transformer", "TemporalConvNet", "TSN", "LSTM")


def set_root(root: Path) -> None:
    global ROOT, EXPERIMENTS, SHARED, RAW_JSONL, RAW_SCHEMAS, SOURCE_MARKDOWN
    ROOT = root
    EXPERIMENTS = ROOT / "experiments"
    SHARED = ROOT / "shared_artifacts"
    RAW_JSONL = SHARED / "raw_tables_jsonl"
    RAW_SCHEMAS = SHARED / "raw_table_schemas"
    SOURCE_MARKDOWN = SHARED / "source_markdown"


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    mkdir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    mkdir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def rel_link(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    mkdir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))
    return True


def merge_dir(src: Path, dst: Path, *, overwrite_files: bool = False) -> None:
    if not src.exists():
        return
    mkdir(dst)
    for child in src.iterdir():
        target = dst / child.name
        if target.exists():
            if child.is_dir() and target.is_dir():
                merge_dir(child, target, overwrite_files=overwrite_files)
                try:
                    child.rmdir()
                except OSError:
                    pass
            elif overwrite_files:
                if target.exists() or target.is_symlink():
                    target.unlink()
                shutil.move(str(child), str(target))
            else:
                stem = target.stem
                suffix = target.suffix
                i = 2
                while (dst / f"{stem}_duplicate_{i}{suffix}").exists():
                    i += 1
                shutil.move(str(child), str(dst / f"{stem}_duplicate_{i}{suffix}"))
        else:
            shutil.move(str(child), str(target))
    try:
        src.rmdir()
    except OSError:
        pass


def cleanup_stale_artifacts() -> None:
    """Remove rerun leftovers that should not accumulate in the reader bundle."""
    for duplicate in ROOT.rglob("*_duplicate_*"):
        if duplicate.is_dir():
            shutil.rmtree(duplicate)
        elif duplicate.exists() or duplicate.is_symlink():
            duplicate.unlink()

    stale_patterns = [
        "primary_full_grid_calibrated/calibrators/*platt_month.joblib",
        "primary_full_grid_calibrated/calibrators/ERA5*",
        "primary_full_grid_calibrated/calibrators/SEAS5*",
        "primary_full_grid_calibrated/calibrators/Spline_Logistic_Regression*",
        "primary_full_grid_calibrated/predictions/ERA5*",
        "primary_full_grid_calibrated/predictions/SEAS5*",
        "primary_full_grid_calibrated/predictions/Spline_Logistic_Regression*",
        "primary_full_grid_calibrated/calibration_metadata/ERA5*",
        "primary_full_grid_calibrated/calibration_metadata/SEAS5*",
        "primary_full_grid_calibrated/calibration_metadata/Spline_Logistic_Regression*",
        "models/spline_logistic_regression_full*",
        "predictions/spline_logistic_regression_full*",
        "raw_tables_jsonl/input_source_full_grid_calibrated*.jsonl.gz",
        "raw_tables_jsonl/input_source_legacy_vs_full_grid_summary*.jsonl.gz",
        "raw_table_schemas/input_source_full_grid_calibrated*.schema.json",
        "raw_table_schemas/input_source_legacy_vs_full_grid_summary*.schema.json",
        "logs/source_full_grid_evaluation.log",
    ]
    for pattern in stale_patterns:
        for path in SHARED.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()

    stale_source_markdown = [
        "main_model_comparison.md",
        "paper_ready_tables.md",
        "repo_audit.md",
        "report.md",
    ]
    stale_terms = ("Spline", "spline_logistic")
    for name in stale_source_markdown:
        path = SOURCE_MARKDOWN / name
        if path.exists() and any(term in path.read_text(errors="ignore") for term in stale_terms):
            path.unlink()


def table_stem_from_csv(csv_name: str) -> str:
    return "_".join(Path(csv_name).with_suffix("").parts)


def raw_jsonl_path(csv_name: str) -> Path:
    return RAW_JSONL / f"{table_stem_from_csv(csv_name)}.jsonl.gz"


def raw_schema_path(csv_name: str) -> Path:
    return RAW_SCHEMAS / f"{table_stem_from_csv(csv_name)}.schema.json"


def load_table(name: str) -> pd.DataFrame | None:
    candidates = [
        ROOT / name,
        RAW_JSONL / f"{table_stem_from_csv(name)}.jsonl.gz",
    ]
    for p in candidates:
        if not p.exists():
            continue
        if p.suffix == ".gz":
            return pd.read_json(p, orient="records", lines=True, compression="gzip")
        return pd.read_csv(p)
    return None


def load_all_tables() -> dict[str, pd.DataFrame]:
    names: set[str] = set()
    names.update(p.name for p in ROOT.glob("*.csv"))
    names.update(f"{p.stem.removesuffix('.jsonl')}.csv" for p in RAW_JSONL.glob("*.jsonl.gz"))
    # Expected tables from the complete revision workflow.
    names.update(
        {
            "dataset_statistics.csv",
            "dataset_statistics_by_year.csv",
            "main_model_comparison.csv",
            "main_model_comparison_by_year.csv",
            "legacy_sampled_model_comparison.csv",
            "legacy_sampled_model_comparison_by_year.csv",
            "primary_full_grid_calibrated/model_comparison.csv",
            "primary_full_grid_calibrated/model_comparison_by_year.csv",
            "primary_full_grid_calibrated/probability_metrics.csv",
            "primary_full_grid_calibrated/reliability_bins.csv",
            "primary_full_grid_calibrated/monthly_count_calibration.csv",
            "primary_full_grid_calibrated/country_count_calibration.csv",
            "primary_full_grid_calibrated/region_count_calibration.csv",
            "primary_full_grid_calibrated/prevalence_audit.csv",
            "primary_full_grid_calibrated/risk_concentration.csv",
            "primary_full_grid_calibrated/count_correction.csv",
            "primary_full_grid_calibrated/spatial_scale_evaluation.csv",
            "legacy_sampled_case_control/model_comparison.csv",
            "legacy_sampled_case_control/model_comparison_by_year.csv",
            "legacy_sampled_case_control/threshold_metrics.csv",
            "feature_ablation.csv",
            "feature_ablation_by_year.csv",
            "embedding_fusion_ablation.csv",
            "embedding_fusion_ablation_by_year.csv",
            "neural_feature_ablation.csv",
            "neural_feature_ablation_by_year.csv",
            "label_sensitivity.csv",
            "label_sensitivity_by_year.csv",
            "lead_time_sensitivity.csv",
            "lead_time_sensitivity_by_year.csv",
            "input_source_comparison.csv",
            "input_source_comparison_by_year.csv",
            "representative_model_ablation_era5_to_ecmwf_no_tp.csv",
            "representative_model_ablation_era5_to_ecmwf_no_tp_by_year.csv",
            "feature_importance_native.csv",
            "grouped_permutation_importance.csv",
            "grouped_permutation_importance_by_year.csv",
            "climate_window_permutation_importance.csv",
            "shap_importance.csv",
            "neural_feature_importance.csv",
            "era5_feature_schema.csv",
            "era5_seas5_common_schema.csv",
            "metrics_wide.csv",
            "metrics_long.csv",
            "experiment_registry.csv",
        }
    )
    out: dict[str, pd.DataFrame] = {}
    for name in sorted(names):
        df = load_table(name)
        if df is not None:
            out[name] = df
    return out


def archive_root_outputs(dfs: dict[str, pd.DataFrame]) -> None:
    mkdir(SHARED)

    dir_moves = {
        "plots": "source_plots_mixed",
        "models": "models",
        "predictions": "predictions",
        "neural_data": "neural_data",
        "neural_model_metrics": "neural_model_metrics",
        "target_caches": "target_caches",
        "training_curves": "training_curves",
        "configs_used": "configs_used",
        "primary_full_grid_calibrated": "primary_full_grid_calibrated",
        "legacy_sampled_case_control": "legacy_sampled_case_control",
    }
    for src_name, dst_name in dir_moves.items():
        merge_dir(
            ROOT / src_name,
            SHARED / dst_name,
            overwrite_files=True,
        )

    # Archive wide root CSVs as JSONL.GZ so presentation CSVs stay narrow.
    mkdir(RAW_JSONL)
    mkdir(RAW_SCHEMAS)
    for csv in list(ROOT.glob("*.csv")):
        df = dfs.get(csv.name)
        if df is None:
            df = pd.read_csv(csv)
        out = raw_jsonl_path(csv.name)
        df.to_json(out, orient="records", lines=True, compression="gzip")
        schema = {
            "source_csv": csv.name,
            "rows": int(len(df)),
            "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        }
        write_json(raw_schema_path(csv.name), schema)
        csv.unlink()

    # Archive wide CSVs inside primary/legacy evaluation source directories too;
    # the builder has already loaded them into memory for presentation tables.
    for source_dir in [SHARED / "primary_full_grid_calibrated", SHARED / "legacy_sampled_case_control"]:
        if not source_dir.exists():
            continue
        for csv in list(source_dir.rglob("*.csv")):
            df = pd.read_csv(csv)
            rel_stem = "_".join(csv.relative_to(SHARED).with_suffix("").parts)
            out = RAW_JSONL / f"{rel_stem}.jsonl.gz"
            df.to_json(out, orient="records", lines=True, compression="gzip")
            schema = {
                "source_csv": str(csv.relative_to(SHARED)),
                "rows": int(len(df)),
                "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
            }
            write_json(RAW_SCHEMAS / f"{rel_stem}.schema.json", schema)
            csv.unlink()

    for js in list(ROOT.glob("*.json")):
        mkdir(SHARED / "json")
        target = SHARED / "json" / js.name
        if target.exists():
            target.unlink()
        shutil.move(str(js), str(target))

    for txt in list(ROOT.glob("*.txt")) + list(ROOT.glob("*.log")):
        mkdir(SHARED / "logs")
        target = SHARED / "logs" / txt.name
        if target.exists():
            target.unlink()
        shutil.move(str(txt), str(target))

    for md in list(ROOT.glob("*.md")):
        if md.name == "README.md":
            continue
        mkdir(SOURCE_MARKDOWN)
        target = SOURCE_MARKDOWN / md.name
        if target.exists():
            target = SOURCE_MARKDOWN / f"{md.stem}_root{md.suffix}"
            if target.exists():
                target.unlink()
        shutil.move(str(md), str(target))


def source_text(name: str) -> str:
    for p in [
        ROOT / name,
        SOURCE_MARKDOWN / name,
        SHARED / "logs" / name,
        SHARED / "target_caches" / name,
    ]:
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    return ""


def plot_source(stem: str, suffix: str) -> Path | None:
    for p in [
        SHARED / "source_plots_mixed" / f"{stem}{suffix}",
        ROOT / "plots" / f"{stem}{suffix}",
    ]:
        if p.exists():
            return p
    return None


def round_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            lower = col.lower()
            if "rate" in lower:
                out[col] = out[col].round(8)
            elif pd.to_numeric(out[col], errors="coerce").abs().max(skipna=True) < 0.01:
                out[col] = out[col].round(6)
            else:
                out[col] = out[col].round(4)
        elif pd.api.types.is_integer_dtype(out[col]):
            continue
    return out


def select_table(
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    *,
    sort_by: list[str] | None = None,
    head: int | None = None,
) -> pd.DataFrame:
    selected = []
    rename = {}
    for src, dst in columns:
        if src in df.columns:
            selected.append(src)
            rename[src] = dst
    out = df.loc[:, selected].rename(columns=rename)
    if sort_by:
        actual = [c for c in sort_by if c in out.columns]
        if actual:
            out = out.sort_values(actual)
    if head is not None:
        out = out.head(head)
    if len(out.columns) > MAX_PRESENTATION_TABLE_COLUMNS:
        raise ValueError(f"Presentation table has {len(out.columns)} columns: {list(out.columns)}")
    return round_for_csv(out)


def filter_global(df: pd.DataFrame) -> pd.DataFrame:
    if "Region" in df.columns:
        return df[df["Region"].astype(str).str.lower() == "global"].copy()
    if "region_display" in df.columns:
        return df[df["region_display"].astype(str).str.lower() == "global"].copy()
    if "region" in df.columns:
        return df[df["region"].astype(str).str.lower() == "global"].copy()
    return df.copy()


def filter_non_global(df: pd.DataFrame) -> pd.DataFrame:
    if "Region" in df.columns:
        return df[df["Region"].astype(str).str.lower() != "global"].copy()
    if "region_display" in df.columns:
        return df[df["region_display"].astype(str).str.lower() != "global"].copy()
    if "region" in df.columns:
        return df[df["region"].astype(str).str.lower() != "global"].copy()
    return df.copy()


def annual_only(df: pd.DataFrame) -> pd.DataFrame:
    if "period" not in df.columns:
        return df.copy()
    return df[df["period"].astype(str).isin(["2021", "2022", "2023", "2024", "2025"])].copy()


def combined_periods(df: pd.DataFrame) -> pd.DataFrame:
    if "period" not in df.columns:
        return df.copy()
    return df[df["period"].astype(str).isin(["2021-2023", "2021-2025"])].copy()


def top_metric_sentence(
    df: pd.DataFrame,
    item_col: str,
    metric_col: str,
    *,
    prefix: str = "Best row",
) -> str:
    if df.empty or item_col not in df.columns or metric_col not in df.columns:
        return "No metric rows were available for automatic summarization."
    work = df.dropna(subset=[metric_col])
    if work.empty:
        return "Metric values were unavailable for automatic summarization."
    row = work.loc[work[metric_col].idxmax()]
    return f"{prefix}: {row[item_col]} with {metric_col}={row[metric_col]:.4f}."


def model_inventory_sentence(df: pd.DataFrame, model_col: str) -> str:
    if df.empty or model_col not in df.columns:
        return "No model inventory was available."
    models = sorted(str(value) for value in df[model_col].dropna().unique())
    neural = [name for name in models if any(marker in name for marker in NEURAL_MODEL_MARKERS)]
    if not models:
        return "No model inventory was available."
    if neural:
        return f"Model inventory: {len(models)} models total, including {len(neural)} neural models ({', '.join(neural)})."
    return f"Model inventory: {len(models)} models total; no neural model rows were found."


def worst_drop_sentence(df: pd.DataFrame, item_col: str, drop_col: str) -> str:
    if df.empty or item_col not in df.columns or drop_col not in df.columns:
        return "No drop metric rows were available for automatic summarization."
    work = df.dropna(subset=[drop_col])
    if work.empty:
        return "Drop values were unavailable for automatic summarization."
    row = work.loc[work[drop_col].idxmax()]
    return f"Largest full-minus-variant delta: {row[item_col]} with {drop_col}={row[drop_col]:.4f}."


@dataclass
class Experiment:
    slug: str
    title: str
    purpose: str
    source_tables: list[str] = field(default_factory=list)
    plot_stems: list[str] = field(default_factory=list)
    artifact_globs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Builder:
    def __init__(self, dfs: dict[str, pd.DataFrame]) -> None:
        self.dfs = dfs
        self.index_rows: list[dict[str, str]] = []

    def exp_dir(self, exp: Experiment) -> Path:
        return EXPERIMENTS / exp.slug

    def start(self, exp: Experiment) -> Path:
        d = self.exp_dir(exp)
        notes = "\n".join(f"- {note}" for note in exp.notes)
        sources = "\n".join(f"- `{name}`" for name in exp.source_tables)
        write_text(
            d / "description.md",
            f"""# {exp.title}

## Purpose
{exp.purpose}

## Source Tables
{sources or "- No source table required."}

## Notes
{notes or f"- Presentation CSVs in this folder are capped at {MAX_PRESENTATION_TABLE_COLUMNS} columns."}
""",
        )
        self.copy_plots(d, exp.plot_stems)
        self.write_artifacts(d, exp)
        self.index_rows.append(
            {
                "Experiment": exp.title,
                "Folder": exp.slug,
                "Purpose": exp.purpose,
            }
        )
        return d

    def copy_plots(self, d: Path, stems: Iterable[str]) -> None:
        for stem in stems:
            for suffix, subdir in [(".png", "png"), (".pdf", "pdf")]:
                src = plot_source(stem, suffix)
                if src:
                    mkdir(d / "plots" / subdir)
                    shutil.copy2(src, d / "plots" / subdir / src.name)

    def write_artifacts(self, d: Path, exp: Experiment) -> None:
        artifact_dir = d / "artifacts"
        raw_sources: dict[str, object] = {"source_tables": [], "plots": [], "linked_artifacts": []}
        for name in exp.source_tables:
            raw = raw_jsonl_path(name)
            schema = raw_schema_path(name)
            if raw.exists():
                rel_link(raw, artifact_dir / "raw_tables" / raw.name)
                raw_sources["source_tables"].append(str(raw.relative_to(ROOT)))
            if schema.exists():
                rel_link(schema, artifact_dir / "raw_table_schemas" / schema.name)
        for stem in exp.plot_stems:
            for suffix in [".png", ".pdf"]:
                src = plot_source(stem, suffix)
                if src:
                    raw_sources["plots"].append(str(src.relative_to(ROOT)))
        for js in ["feature_groups.json", "run_manifest.json"]:
            src = SHARED / "json" / js
            if rel_link(src, artifact_dir / "json" / js):
                raw_sources["linked_artifacts"].append(str(src.relative_to(ROOT)))
        for pattern in exp.artifact_globs:
            for src in sorted(SHARED.glob(pattern)):
                if src.is_file():
                    rel = src.relative_to(SHARED)
                    rel_link(src, artifact_dir / "linked_files" / rel)
                    raw_sources["linked_artifacts"].append(str(src.relative_to(ROOT)))
        write_json(artifact_dir / "raw_sources.json", raw_sources)

    def write_table(self, d: Path, filename: str, df: pd.DataFrame) -> None:
        if len(df.columns) > MAX_PRESENTATION_TABLE_COLUMNS:
            raise ValueError(f"{filename} has {len(df.columns)} columns")
        mkdir(d / "tables")
        df.to_csv(d / "tables" / filename, index=False)

    def write_analysis(self, d: Path, lines: Iterable[str]) -> None:
        body = "\n".join(f"- {line}" for line in lines if line)
        write_text(d / "analysis.md", f"# Analysis\n\n{body or '- No automatic analysis available.'}")

    def build_metadata(self) -> None:
        exp = Experiment(
            "00_run_metadata_and_audit",
            "Run Metadata And Repository Audit",
            "Collects the repository audit, command log, environment/config references, feature schemas, and run manifest used by the complete revision workflow.",
            [
                "era5_feature_schema.csv",
                "era5_seas5_common_schema.csv",
                "experiment_registry.csv",
            ],
            artifact_globs=[
                "configs_used/*",
                "json/*",
                "logs/*",
                "raw_tables_jsonl/*",
                "raw_table_schemas/*",
            ],
        )
        d = self.start(exp)
        era = self.dfs.get("era5_feature_schema.csv", pd.DataFrame())
        if not era.empty:
            self.write_table(
                d,
                "era5_feature_schema.csv",
                select_table(
                    era,
                    [
                        ("variable", "Variable"),
                        ("era5_zarr_files", "ERA5 Files"),
                        ("era5_years", "ERA5 Years"),
                        ("ecmwf_zarr_files", "ECMWF Files"),
                        ("status", "Status"),
                    ],
                ),
            )
        common = self.dfs.get("era5_seas5_common_schema.csv", pd.DataFrame())
        if not common.empty:
            self.write_table(
                d,
                "era5_seas5_common_schema.csv",
                select_table(
                    common,
                    [
                        ("variable", "Variable"),
                        ("era5_zarr_files", "ERA5 Files"),
                        ("era5_years", "ERA5 Years"),
                        ("ecmwf_zarr_files", "ECMWF Files"),
                        ("status", "Status"),
                    ],
                ),
            )
        registry = self.dfs.get("experiment_registry.csv", pd.DataFrame())
        if not registry.empty:
            self.write_table(
                d,
                "experiment_registry_summary.csv",
                select_table(
                    registry,
                    [
                        ("experiment_id", "Experiment ID"),
                        ("experiment_type", "Type"),
                        ("model", "Model"),
                        ("feature_set", "Feature Set"),
                        ("status", "Status"),
                        ("threshold", "Threshold"),
                    ],
                    sort_by=["Type", "Experiment ID"],
                ),
            )
        repo_audit = source_text("repo_audit.md")
        if repo_audit:
            write_text(d / "artifacts" / "repo_audit_original.md", repo_audit)
        self.write_analysis(
            d,
            [
                "This folder is the provenance hub for the organized result library.",
                "Wide raw source tables are stored as JSONL.GZ under shared artifacts; presentation CSVs stay narrow for reading.",
                "Presentation tables are written as CSV and Markdown.",
            ],
        )

    def build_dataset(self) -> None:
        stats = self.dfs.get("dataset_statistics.csv", pd.DataFrame())
        by_year = self.dfs.get("dataset_statistics_by_year.csv", pd.DataFrame())
        if not stats.empty:
            exp = Experiment(
                "01_dataset_statistics_splits",
                "Dataset Statistics By Split",
                "Summarizes sample counts, class balance, spatial support, labeling rule, and grid resolution for the train/validation/test splits.",
                ["dataset_statistics.csv"],
            )
            d = self.start(exp)
            self.write_table(
                d,
                "split_statistics.csv",
                select_table(
                    stats,
                    [
                        ("split_region", "Split / Region"),
                        ("years", "Years"),
                        ("unique_grid_cells", "Grid Cells"),
                        ("cell_days", "Cell-Days"),
                        ("positive_samples", "Positives"),
                        ("positive_rate", "Positive Rate"),
                    ],
                ),
            )
            self.write_analysis(d, ["Use this table for compact dataset-size reporting without model metrics."])

        if not by_year.empty:
            exp = Experiment(
                "02_dataset_statistics_by_region",
                "Dataset Statistics By Region",
                "Separates regional dataset statistics from model metrics so spatial coverage can be read independently.",
                ["dataset_statistics_by_year.csv"],
            )
            d = self.start(exp)
            region = by_year.groupby("region_display", as_index=False).agg(
                cell_days=("cell_days", "sum"),
                positive_samples=("positive_samples", "sum"),
                negative_samples=("negative_samples", "sum"),
                unique_grid_cells=("unique_grid_cells", "max"),
            )
            region["positive_rate"] = region["positive_samples"] / (
                region["positive_samples"] + region["negative_samples"]
            )
            self.write_table(
                d,
                "regional_dataset_statistics.csv",
                select_table(
                    region,
                    [
                        ("region_display", "Region"),
                        ("cell_days", "Cell-Days"),
                        ("unique_grid_cells", "Grid Cells"),
                        ("positive_samples", "Positives"),
                        ("negative_samples", "Negatives"),
                        ("positive_rate", "Positive Rate"),
                    ],
                    sort_by=["Region"],
                ),
            )
            self.write_analysis(d, ["Regional coverage is aggregated over the available 2021-2025 test years."])

            exp = Experiment(
                "03_dataset_statistics_by_year",
                "Dataset Statistics By Year",
                "Keeps annual class-balance and sample-count reporting separate from regional and split summaries.",
                ["dataset_statistics_by_year.csv"],
            )
            d = self.start(exp)
            self.write_table(
                d,
                "yearly_dataset_statistics.csv",
                select_table(
                    by_year,
                    [
                        ("region_display", "Region"),
                        ("year", "Year"),
                        ("cell_days", "Cell-Days"),
                        ("positive_samples", "Positives"),
                        ("negative_samples", "Negatives"),
                        ("positive_rate", "Positive Rate"),
                    ],
                    sort_by=["Region", "Year"],
                ),
            )
            self.write_analysis(d, ["This table supports year-by-year reporting for 2021 through 2025."])

    def build_primary_full_grid(self) -> None:
        primary = self.dfs.get("primary_full_grid_calibrated/model_comparison.csv", pd.DataFrame())
        probability = self.dfs.get("primary_full_grid_calibrated/probability_metrics.csv", pd.DataFrame())
        reliability = self.dfs.get("primary_full_grid_calibrated/reliability_bins.csv", pd.DataFrame())
        prevalence = self.dfs.get("primary_full_grid_calibrated/prevalence_audit.csv", pd.DataFrame())
        risk = self.dfs.get("primary_full_grid_calibrated/risk_concentration.csv", pd.DataFrame())
        correction = self.dfs.get("primary_full_grid_calibrated/count_correction.csv", pd.DataFrame())
        spatial = self.dfs.get("primary_full_grid_calibrated/spatial_scale_evaluation.csv", pd.DataFrame())
        legacy = self.dfs.get("legacy_sampled_case_control/model_comparison.csv", pd.DataFrame())
        if legacy.empty:
            legacy = self.dfs.get("legacy_sampled_model_comparison.csv", pd.DataFrame())
        dataset = self.dfs.get("dataset_statistics.csv", pd.DataFrame())
        if primary.empty:
            return

        if prevalence.empty and not probability.empty:
            prevalence = probability.assign(
                denominator="deployment_weighted_grid_cell_days",
                denominator_count=probability.get("weighted_support"),
                event_count=probability.get("positives"),
                event_rate=probability.get("observed_prevalence"),
                event_rate_error=probability.get("observed_prevalence_error"),
                notes="Fallback from probability_metrics.csv.",
            )[
                [
                    "model_name",
                    "denominator",
                    "denominator_count",
                    "event_count",
                    "event_rate",
                    "event_rate_error",
                    "notes",
                ]
            ]

        prevalence_rows: list[dict[str, object]] = []
        if not dataset.empty:
            for _, row in dataset[
                dataset["split_region"].astype(str).str.startswith("Global")
                & dataset["split_region"].astype(str).str.contains("train|validation|test", case=False, regex=True)
            ].iterrows():
                prevalence_rows.append(
                    {
                        "source": "sampled_case_control",
                        "denominator": row.get("split_region"),
                        "event_count": row.get("positive_samples"),
                        "denominator_count": row.get("cell_days"),
                        "event_rate": row.get("positive_rate"),
                        "event_rate_error": None,
                    }
                )
        if not prevalence.empty:
            for _, row in prevalence.iterrows():
                prevalence_rows.append(
                    {
                        "source": row.get("model_name"),
                        "denominator": row.get("denominator"),
                        "event_count": row.get("event_count"),
                        "denominator_count": row.get("denominator_count"),
                        "event_rate": row.get("event_rate"),
                        "event_rate_error": row.get("event_rate_error"),
                    }
                )
        prevalence_view = pd.DataFrame(prevalence_rows)

        contrast = pd.DataFrame()
        if not legacy.empty:
            legacy_global = filter_global(legacy).copy()
            full_global = filter_global(primary).copy()
            if not legacy_global.empty and not full_global.empty:
                contrast = full_global.merge(
                    legacy_global,
                    left_on="Model",
                    right_on="Model",
                    suffixes=("_full_grid", "_sampled"),
                    how="inner",
                )
                if not contrast.empty:
                    sampled_rate = None
                    if "positives_sampled" in contrast.columns and "support_sampled" in contrast.columns:
                        sampled_rate = pd.to_numeric(contrast["positives_sampled"], errors="coerce") / pd.to_numeric(
                            contrast["support_sampled"], errors="coerce"
                        )
                    contrast["sampled_positive_rate"] = sampled_rate
                    contrast["full_grid_prevalence"] = contrast.get("observed_prevalence")
                    contrast["full_grid_ap_lift"] = (
                        pd.to_numeric(contrast.get("average_precision"), errors="coerce")
                        / pd.to_numeric(contrast.get("observed_prevalence"), errors="coerce")
                    )
        exp = Experiment(
            "04_primary_full_grid_calibrated",
            "Calibrated Full-Grid Prediction Study",
            (
                "Keeps deployment-grid calibrated prediction in one place: prevalence audit, sampled-vs-full-grid contrast, "
                "risk concentration, count calibration, and spatial-scale evaluation."
            ),
            [
                "primary_full_grid_calibrated/model_comparison.csv",
                "primary_full_grid_calibrated/probability_metrics.csv",
                "primary_full_grid_calibrated/reliability_bins.csv",
                "primary_full_grid_calibrated/monthly_count_calibration.csv",
                "primary_full_grid_calibrated/country_count_calibration.csv",
                "primary_full_grid_calibrated/prevalence_audit.csv",
                "primary_full_grid_calibrated/risk_concentration.csv",
                "primary_full_grid_calibrated/count_correction.csv",
                "primary_full_grid_calibrated/spatial_scale_evaluation.csv",
            ],
            artifact_globs=[
                "primary_full_grid_calibrated/calibrators/*",
                "primary_full_grid_calibrated/calibration_metadata*",
                "primary_full_grid_calibrated/predictions/*",
            ],
            notes=[
                "Classification F1 is expected to be tiny at the deployment base rate; risk-ranking and count-calibration tables carry the operational interpretation.",
                "The target is a MODIS/FIRMS-derived fire-positive grid cell-day, not a unique ignition event.",
                "Full-grid rows are not copied into the sampled model-comparison experiments.",
            ],
        )
        d = self.start(exp)
        if not prevalence_view.empty:
            self.write_table(
                d,
                "01_prevalence_audit.csv",
                select_table(
                    prevalence_view,
                    [
                        ("source", "Source"),
                        ("denominator", "Denominator"),
                        ("event_count", "Events"),
                        ("denominator_count", "Denominator Count"),
                        ("event_rate", "Event Rate"),
                        ("event_rate_error", "Rate Error"),
                    ],
                    sort_by=["Source", "Denominator"],
                ),
            )
        if not contrast.empty:
            self.write_table(
                d,
                "02_sampled_vs_full_grid_metric_contrast.csv",
                select_table(
                    contrast,
                    [
                        ("Model", "Model"),
                        ("sampled_positive_rate", "Sampled Rate"),
                        ("full_grid_prevalence", "Full-Grid Rate"),
                        ("f1", "Sampled F1"),
                        ("max_f1", "Full-Grid Max F1"),
                        ("PR-AUC_sampled", "Sampled PR-AUC"),
                        ("average_precision", "Full-Grid AP"),
                        ("full_grid_ap_lift", "AP Lift"),
                    ],
                    sort_by=["Model"],
                ),
            )
        if not risk.empty:
            self.write_table(
                d,
                "03_risk_concentration.csv",
                select_table(
                    risk,
                    [
                        ("model_name", "Model"),
                        ("q_label", "Top Q"),
                        ("recall_at_top_q", "Recall@Q"),
                        ("recall_at_top_q_error", "Recall Error"),
                        ("lift_at_q", "Lift@Q"),
                        ("lift_at_q_error", "Lift Error"),
                        ("precision_at_top_q", "Precision@Q"),
                        ("ap_lift", "AP Lift"),
                    ],
                    sort_by=["Model", "Top Q"],
                ),
            )
        if not correction.empty:
            self.write_table(
                d,
                "04_count_correction.csv",
                select_table(
                    correction,
                    [
                        ("model_name", "Model"),
                        ("raw_expected_observed_count_ratio", "Raw E/O"),
                        ("raw_expected_observed_count_ratio_error", "Raw E/O Error"),
                        ("calibrated_expected_observed_count_ratio", "Cal E/O"),
                        ("calibrated_expected_observed_count_ratio_error", "Cal E/O Error"),
                        ("eo_ratio_reduction_factor", "Reduction"),
                    ],
                    sort_by=["Model"],
                ),
            )
        if not spatial.empty:
            self.write_table(
                d,
                "05_spatial_scale_evaluation.csv",
                select_table(
                    spatial,
                    [
                        ("model_name", "Model"),
                        ("scale", "Scale"),
                        ("observed_prevalence", "Rate"),
                        ("average_precision", "AP"),
                        ("average_precision_error", "AP Error"),
                        ("roc_auc", "ROC-AUC"),
                        ("max_f1", "Max F1"),
                        ("max_f1_error", "F1 Error"),
                    ],
                    sort_by=["Model", "Scale"],
                ),
            )
        self.write_table(
            d,
            "supporting_full_grid_global_metrics.csv",
            select_table(
                filter_global(primary),
                [
                    ("Model", "Model"),
                    ("PR-AUC", "PR-AUC"),
                    ("ROC-AUC", "ROC-AUC"),
                    ("weighted_brier_score", "Brier"),
                    ("weighted_logloss", "Logloss"),
                    ("expected_observed_count_ratio", "E/O Count"),
                ],
                sort_by=["PR-AUC"],
            ),
        )
        if not probability.empty:
            self.write_table(
                d,
                "supporting_probability_quality.csv",
                select_table(
                    probability,
                    [
                        ("model_name", "Model"),
                        ("observed_prevalence", "Prevalence"),
                        ("mean_calibrated_predicted_probability", "Mean Prob"),
                        ("calibration_intercept", "Cal Intercept"),
                        ("calibration_slope", "Cal Slope"),
                        ("daily_expected_observed_count_mae", "Daily MAE"),
                    ],
                    sort_by=["Model"],
                ),
            )
        if not reliability.empty:
            self.write_table(
                d,
                "supporting_reliability_bin_sample.csv",
                select_table(
                    reliability.head(60),
                    [
                        ("model_name", "Model"),
                        ("bin", "Bin"),
                        ("n_weighted", "Weight"),
                        ("mean_predicted_probability", "Mean Prob"),
                        ("observed_prevalence", "Observed"),
                        ("expected_observed_count_ratio", "E/O"),
                    ],
                    sort_by=["Model", "Bin"],
                ),
            )
        self.write_analysis(
            d,
            [
                top_metric_sentence(filter_global(primary), "Model", "PR-AUC", prefix="Highest full-grid AP"),
                "Experiment 1 makes the base-rate denominator explicit before interpreting any threshold metric.",
                "Experiment 2 contrasts sampled and deployment-grid metrics; sampled PR-AUC/F1 should not be reported as deployment F1.",
                "Experiment 3 reports whether high-score cells concentrate risk using Recall@top q%, Lift@q, and AP lift.",
                "Experiment 4 compares raw and calibrated E/O ratios to quantify overprediction correction.",
                "Experiment 5 tests whether localization improves when exact 0.1 degree cells are relaxed to neighborhoods or coarser grids.",
            ],
        )

    def build_legacy_sampled(self) -> None:
        legacy = self.dfs.get("legacy_sampled_case_control/model_comparison.csv", pd.DataFrame())
        if legacy.empty:
            legacy = self.dfs.get("legacy_sampled_model_comparison.csv", pd.DataFrame())
        threshold = self.dfs.get("legacy_sampled_case_control/threshold_metrics.csv", pd.DataFrame())
        if legacy.empty:
            return
        exp = Experiment(
            "05_legacy_sampled_case_control",
            "Legacy Sampled Case-Control Evaluation",
            (
                "Retains the old undersampled-negative evaluation for backwards comparison. "
                "These metrics describe discrimination/classification on the sampled case-control table "
                "and should not be interpreted as calibrated deployment probabilities."
            ),
            [
                "legacy_sampled_case_control/model_comparison.csv",
                "legacy_sampled_case_control/threshold_metrics.csv",
                "legacy_sampled_model_comparison.csv",
            ],
            artifact_globs=["legacy_sampled_case_control/*"],
        )
        d = self.start(exp)
        self.write_table(
            d,
            "legacy_global_metrics.csv",
            select_table(
                filter_global(legacy),
                [
                    ("Model", "Model"),
                    ("Feature set", "Feature Set"),
                    ("precision", "Precision"),
                    ("recall", "Recall"),
                    ("f1", "F1"),
                    ("PR-AUC", "PR-AUC"),
                ],
                sort_by=["Model"],
            ),
        )
        if not threshold.empty:
            self.write_table(
                d,
                "legacy_threshold_metrics.csv",
                select_table(
                    filter_global(threshold).head(80),
                    [
                        ("model", "Model"),
                        ("split", "Split"),
                        ("threshold", "Threshold"),
                        ("precision", "Precision"),
                        ("recall", "Recall"),
                        ("f1", "F1"),
                    ],
                    sort_by=["Model", "Split"],
                ),
            )
        self.write_analysis(
            d,
            [
                "Legacy sampled metrics are useful for comparing to older runs but inherit the artificial sampled prevalence.",
                "Use the calibrated full-grid study folder for deployment probability and expected-count claims.",
            ],
        )

    def build_main(self) -> None:
        main = self.dfs.get("main_model_comparison.csv", pd.DataFrame())
        by_year = self.dfs.get("main_model_comparison_by_year.csv", pd.DataFrame())
        if (
            not main.empty
            and "evaluation_type" in main.columns
            and main["evaluation_type"].astype(str).eq("primary_full_grid_calibrated").any()
        ):
            sampled = self.dfs.get("legacy_sampled_case_control/model_comparison.csv", pd.DataFrame())
            sampled_by_year = self.dfs.get("legacy_sampled_case_control/model_comparison_by_year.csv", pd.DataFrame())
            if sampled.empty:
                sampled = self.dfs.get("legacy_sampled_model_comparison.csv", pd.DataFrame())
            if sampled_by_year.empty:
                sampled_by_year = self.dfs.get("legacy_sampled_model_comparison_by_year.csv", pd.DataFrame())
            if not sampled.empty:
                main = sampled
                by_year = sampled_by_year
        if main.empty:
            return
        exp = Experiment(
            "06_main_model_comparison_global",
            "Main Model Comparison Global",
            "Compares the main models on the global combined test set while keeping regional and yearly views in their own folders.",
            ["main_model_comparison.csv"],
            ["pr_curves_global"],
            artifact_globs=[
                "models/*full*",
                "models/catboost_*",
                "models/nn_global_full_*",
                "predictions/*full*",
                "predictions/catboost_*",
                "predictions/nn_global_full_*",
                "neural_model_metrics/*",
                "configs_used/nn_global_full_*.yaml",
            ],
            notes=[
                "This table intentionally includes every model row from the source comparison table; it is not top-N filtered.",
            ],
        )
        d = self.start(exp)
        g = filter_global(main)
        self.write_table(
            d,
            "global_model_metrics.csv",
            select_table(
                g,
                [
                    ("Model", "Model"),
                    ("Feature set", "Feature Set"),
                    ("f1", "F1"),
                    ("F1 error", "F1 Error"),
                    ("PR-AUC", "PR-AUC"),
                    ("PR-AUC error", "PR-AUC Error"),
                ],
                sort_by=["Model"],
            ),
        )
        self.write_table(
            d,
            "global_precision_recall_thresholds.csv",
            select_table(
                g,
                [
                    ("Model", "Model"),
                    ("precision", "Precision"),
                    ("recall", "Recall"),
                    ("ROC-AUC", "ROC-AUC"),
                    ("Brier", "Brier"),
                    ("threshold", "Threshold"),
                ],
                sort_by=["Model"],
            ),
        )
        self.write_analysis(
            d,
            [
                model_inventory_sentence(g, "Model"),
                top_metric_sentence(g, "Model", "PR-AUC", prefix="Highest global PR-AUC"),
            ],
        )

        exp = Experiment(
            "07_main_model_comparison_by_region",
            "Main Model Comparison By Region",
            "Separates the regional model comparison from the global table to make spatial robustness easier to inspect.",
            ["main_model_comparison.csv"],
            ["pr_curves_regions"],
            artifact_globs=[
                "predictions/*full*",
                "predictions/catboost_*",
                "predictions/nn_global_full_*",
                "neural_model_metrics/*",
            ],
            notes=[
                "This table intentionally includes every regional model row from the source comparison table; it is not top-N filtered.",
            ],
        )
        d = self.start(exp)
        r = filter_non_global(main)
        self.write_table(
            d,
            "regional_model_metrics.csv",
            select_table(
                r,
                [
                    ("Region", "Region"),
                    ("Model", "Model"),
                    ("f1", "F1"),
                    ("F1 error", "F1 Error"),
                    ("PR-AUC", "PR-AUC"),
                    ("PR-AUC error", "PR-AUC Error"),
                ],
                sort_by=["Region", "Model"],
            ),
        )
        self.write_analysis(
            d,
            [
                model_inventory_sentence(r, "Model"),
                top_metric_sentence(r, "Model", "PR-AUC", prefix="Highest regional row PR-AUC"),
            ],
        )

        if not by_year.empty:
            exp = Experiment(
                "08_main_model_comparison_by_year",
                "Main Model Comparison By Year",
                "Reports annual model metrics for 2021-2025 and keeps temporal stability separate from the regional comparison.",
                ["main_model_comparison_by_year.csv"],
                ["pr_curves_global", "pr_curves_regions"],
                artifact_globs=[
                    "predictions/*full*",
                    "predictions/catboost_*",
                    "predictions/nn_global_full_*",
                    "neural_model_metrics/*",
                ],
                notes=[
                    "This table intentionally includes every yearly model row from the source comparison table; it is not top-N filtered.",
                ],
            )
            d = self.start(exp)
            annual = annual_only(by_year)
            self.write_table(
                d,
                "yearly_global_model_metrics.csv",
                select_table(
                    filter_global(annual),
                    [
                        ("period", "Year"),
                        ("model", "Model"),
                        ("f1", "F1"),
                        ("f1_error", "F1 Error"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Year", "Model"],
                ),
            )
            self.write_table(
                d,
                "yearly_regional_model_metrics.csv",
                select_table(
                    filter_non_global(annual),
                    [
                        ("region_display", "Region"),
                        ("period", "Year"),
                        ("model", "Model"),
                        ("f1", "F1"),
                        ("f1_error", "F1 Error"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Region", "Year", "Model"],
                ),
            )
            self.write_table(
                d,
                "combined_period_model_metrics.csv",
                select_table(
                    combined_periods(by_year),
                    [
                        ("region_display", "Region"),
                        ("period", "Period"),
                        ("model", "Model"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Region", "Period", "Model"],
                ),
            )
            self.write_analysis(
                d,
                [
                    model_inventory_sentence(annual, "model"),
                    top_metric_sentence(annual, "model", "average_precision", prefix="Highest annual PR-AUC"),
                ],
            )

    def build_feature_ablation(self) -> None:
        ab = self.dfs.get("feature_ablation.csv", pd.DataFrame())
        by = self.dfs.get("feature_ablation_by_year.csv", pd.DataFrame())
        if ab.empty:
            return
        exp = Experiment(
            "09_feature_ablation_global",
            "Feature Ablation Global",
            "Measures how CatBoost performance changes when feature sources are removed or restricted on the global combined test set.",
            ["feature_ablation.csv"],
            ["feature_ablation_pr_auc_drop", "feature_ablation_f1_drop"],
            artifact_globs=["models/ablation_*", "predictions/ablation_*"],
        )
        d = self.start(exp)
        g = filter_global(ab)
        self.write_table(
            d,
            "global_ablation_drops.csv",
            select_table(
                g,
                [
                    ("Experiment", "Experiment"),
                    ("f1", "F1"),
                    ("F1 error", "F1 Error"),
                    ("PR-AUC", "PR-AUC"),
                    ("PR-AUC error", "PR-AUC Error"),
                    ("Delta PR-AUC vs full", "Delta PR-AUC"),
                ],
                sort_by=["Experiment"],
            ),
        )
        self.write_table(
            d,
            "global_ablation_precision_recall.csv",
            select_table(
                g,
                [
                    ("Experiment", "Experiment"),
                    ("Model", "Model"),
                    ("precision", "Precision"),
                    ("recall", "Recall"),
                    ("f1", "F1"),
                    ("PR-AUC", "PR-AUC"),
                ],
                sort_by=["Experiment"],
            ),
        )
        self.write_analysis(
            d,
            [
                worst_drop_sentence(g, "Experiment", "Delta PR-AUC vs full"),
                "Delta columns are full CatBoost minus the variant, so positive values mean the variant is lower and negative values mean the variant is higher.",
            ],
        )

        exp = Experiment(
            "10_feature_ablation_by_region",
            "Feature Ablation By Region",
            "Keeps spatial ablation effects separate from the global ablation table.",
            ["feature_ablation.csv"],
            ["feature_ablation_pr_auc_drop", "feature_ablation_f1_drop"],
            artifact_globs=["predictions/ablation_*"],
        )
        d = self.start(exp)
        r = filter_non_global(ab)
        self.write_table(
            d,
            "regional_ablation_drops.csv",
            select_table(
                r,
                [
                    ("Region", "Region"),
                    ("Experiment", "Experiment"),
                    ("f1", "F1"),
                    ("F1 error", "F1 Error"),
                    ("PR-AUC", "PR-AUC"),
                    ("PR-AUC error", "PR-AUC Error"),
                ],
                sort_by=["Region", "Experiment"],
            ),
        )
        self.write_analysis(d, [worst_drop_sentence(r, "Experiment", "Delta PR-AUC vs full")])

        if not by.empty:
            exp = Experiment(
                "11_feature_ablation_by_year",
                "Feature Ablation By Year",
                "Reports annual ablation effects for 2021-2025 apart from the combined and regional-only summaries.",
                ["feature_ablation_by_year.csv"],
                ["feature_ablation_pr_auc_drop", "feature_ablation_f1_drop"],
                artifact_globs=["predictions/ablation_*"],
            )
            d = self.start(exp)
            annual = annual_only(by)
            self.write_table(
                d,
                "yearly_ablation_drops.csv",
                select_table(
                    annual,
                    [
                        ("region_display", "Region"),
                        ("period", "Year"),
                        ("feature_set", "Feature Set"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Region", "Year", "Feature Set"],
                ),
            )
            self.write_table(
                d,
                "combined_period_ablation_drops.csv",
                select_table(
                    combined_periods(by),
                    [
                        ("region_display", "Region"),
                        ("period", "Period"),
                        ("feature_set", "Feature Set"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Region", "Period", "Feature Set"],
                ),
            )
            self.write_analysis(d, [worst_drop_sentence(annual, "feature_set", "delta_average_precision_vs_full")])

    def build_neural(self) -> None:
        neu = self.dfs.get("embedding_fusion_ablation.csv", pd.DataFrame())
        by = self.dfs.get("embedding_fusion_ablation_by_year.csv", pd.DataFrame())
        if neu.empty:
            return
        artifact_globs = [
            "neural_data/*",
            "training_curves/*",
            "neural_model_metrics/*",
            "models/nn_global_full_*",
            "configs_used/nn_global_full_*.yaml",
            "source_plots_mixed/nn_global_full_*",
        ]
        exp = Experiment(
            "12_neural_embedding_fusion_global",
            "Neural Embedding Fusion Global",
            "Compares temporal-only, static-only, concatenation, one-hot, learned embedding, full fusion, and gated neural variants globally.",
            ["embedding_fusion_ablation.csv"],
            ["embedding_fusion_pr_auc", "embedding_fusion_f1"],
            artifact_globs=artifact_globs,
        )
        d = self.start(exp)
        g = filter_global(neu)
        self.write_table(
            d,
            "global_neural_fusion_metrics.csv",
            select_table(
                g,
                [
                    ("experiment", "Variant"),
                    ("f1", "F1"),
                    ("f1_error", "F1 Error"),
                    ("average_precision", "PR-AUC"),
                    ("average_precision_error", "PR-AUC Error"),
                    ("threshold", "Threshold"),
                ],
                sort_by=["Variant"],
            ),
        )
        self.write_analysis(d, [top_metric_sentence(g, "experiment", "average_precision", prefix="Highest global neural PR-AUC")])

        exp = Experiment(
            "13_neural_embedding_fusion_by_region",
            "Neural Embedding Fusion By Region",
            "Separates spatial neural fusion behavior from the global neural comparison.",
            ["embedding_fusion_ablation.csv"],
            ["embedding_fusion_pr_auc", "embedding_fusion_f1"],
            artifact_globs=artifact_globs,
        )
        d = self.start(exp)
        r = filter_non_global(neu)
        self.write_table(
            d,
            "regional_neural_fusion_metrics.csv",
            select_table(
                r,
                [
                    ("region_display", "Region"),
                    ("experiment", "Variant"),
                    ("f1", "F1"),
                    ("f1_error", "F1 Error"),
                    ("average_precision", "PR-AUC"),
                    ("average_precision_error", "PR-AUC Error"),
                ],
                sort_by=["Region", "Variant"],
            ),
        )
        self.write_analysis(d, [top_metric_sentence(r, "experiment", "average_precision", prefix="Highest regional neural row PR-AUC")])

        if not by.empty:
            exp = Experiment(
                "14_neural_embedding_fusion_by_year",
                "Neural Embedding Fusion By Year",
                "Reports annual neural ablation metrics for 2021-2025 separately from regional and global summaries.",
                ["embedding_fusion_ablation_by_year.csv"],
                ["embedding_fusion_pr_auc", "embedding_fusion_f1"],
                artifact_globs=artifact_globs,
            )
            d = self.start(exp)
            annual = annual_only(by)
            self.write_table(
                d,
                "yearly_neural_fusion_metrics.csv",
                select_table(
                    annual,
                    [
                        ("region_display", "Region"),
                        ("period", "Year"),
                        ("experiment", "Variant"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Region", "Year", "Variant"],
                ),
            )
            self.write_table(
                d,
                "combined_period_neural_fusion_metrics.csv",
                select_table(
                    combined_periods(by),
                    [
                        ("region_display", "Region"),
                        ("period", "Period"),
                        ("experiment", "Variant"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Region", "Period", "Variant"],
                ),
            )
            self.write_analysis(d, [top_metric_sentence(annual, "experiment", "average_precision", prefix="Highest annual neural PR-AUC")])

    def build_neural_feature_ablation(self) -> None:
        ab = self.dfs.get("neural_feature_ablation.csv", pd.DataFrame())
        by = self.dfs.get("neural_feature_ablation_by_year.csv", pd.DataFrame())
        if ab.empty:
            return
        artifact_globs = [
            "neural_model_metrics/nn_feature_ablation_*",
            "models/nn_feature_ablation_*",
            "predictions/nn_feature_ablation_*",
            "configs_used/nn_feature_ablation_*.yaml",
            "source_plots_mixed/nn_feature_ablation_*",
            "generated_configs/nn_feature_ablation_*.yaml",
        ]
        exp = Experiment(
            "15_neural_feature_ablation_global",
            "Neural Feature Ablation Global",
            "Ablates dynamic, static, and categorical input branches for the configured best neural architecture on the global combined test set.",
            ["neural_feature_ablation.csv"],
            ["neural_feature_ablation_pr_auc", "neural_feature_ablation_f1"],
            artifact_globs=artifact_globs,
            notes=[
                "The ablation keeps the neural architecture fixed and zeroes one or more input branches in the prepared NN tensors.",
                "Delta columns are full neural feature set minus variant, so positive values mean the ablated variant is lower.",
            ],
        )
        d = self.start(exp)
        g = filter_global(ab)
        self.write_table(
            d,
            "global_neural_feature_ablation.csv",
            select_table(
                g,
                [
                    ("experiment", "Variant"),
                    ("f1", "F1"),
                    ("f1_error", "F1 Error"),
                    ("average_precision", "PR-AUC"),
                    ("average_precision_error", "PR-AUC Error"),
                    ("delta_average_precision_vs_full", "Delta PR-AUC"),
                    ("threshold", "Threshold"),
                ],
                sort_by=["Variant"],
            ),
        )
        self.write_analysis(
            d,
            [
                worst_drop_sentence(g, "experiment", "delta_average_precision_vs_full"),
                top_metric_sentence(g, "experiment", "average_precision", prefix="Highest global neural-feature PR-AUC"),
            ],
        )

        exp = Experiment(
            "16_neural_feature_ablation_by_region",
            "Neural Feature Ablation By Region",
            "Separates regional effects of neural input-branch ablations from the global summary.",
            ["neural_feature_ablation.csv"],
            ["neural_feature_ablation_pr_auc", "neural_feature_ablation_f1"],
            artifact_globs=artifact_globs,
        )
        d = self.start(exp)
        r = filter_non_global(ab)
        self.write_table(
            d,
            "regional_neural_feature_ablation.csv",
            select_table(
                r,
                [
                    ("region_display", "Region"),
                    ("experiment", "Variant"),
                    ("f1", "F1"),
                    ("f1_error", "F1 Error"),
                    ("average_precision", "PR-AUC"),
                    ("average_precision_error", "PR-AUC Error"),
                    ("delta_average_precision_vs_full", "Delta PR-AUC"),
                ],
                sort_by=["Region", "Variant"],
            ),
        )
        self.write_analysis(d, [worst_drop_sentence(r, "experiment", "delta_average_precision_vs_full")])

        if not by.empty:
            exp = Experiment(
                "17_neural_feature_ablation_by_year",
                "Neural Feature Ablation By Year",
                "Reports annual neural input-branch ablation metrics for 2021-2025 separately from regional and global summaries.",
                ["neural_feature_ablation_by_year.csv"],
                ["neural_feature_ablation_pr_auc", "neural_feature_ablation_f1"],
                artifact_globs=artifact_globs,
            )
            d = self.start(exp)
            annual = annual_only(by)
            self.write_table(
                d,
                "yearly_neural_feature_ablation.csv",
                select_table(
                    annual,
                    [
                        ("region_display", "Region"),
                        ("period", "Year"),
                        ("experiment", "Variant"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                        ("delta_average_precision_vs_full", "Delta PR-AUC"),
                    ],
                    sort_by=["Region", "Year", "Variant"],
                ),
            )
            self.write_table(
                d,
                "combined_period_neural_feature_ablation.csv",
                select_table(
                    combined_periods(by),
                    [
                        ("region_display", "Region"),
                        ("period", "Period"),
                        ("experiment", "Variant"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                        ("delta_average_precision_vs_full", "Delta PR-AUC"),
                    ],
                    sort_by=["Region", "Period", "Variant"],
                ),
            )
            self.write_analysis(d, [worst_drop_sentence(annual, "experiment", "delta_average_precision_vs_full")])

    def build_label(self) -> None:
        lab = self.dfs.get("label_sensitivity.csv", pd.DataFrame())
        by = self.dfs.get("label_sensitivity_by_year.csv", pd.DataFrame())
        if lab.empty:
            return
        exp = Experiment(
            "18_label_sensitivity_global",
            "Label Sensitivity Global",
            "Compares the main labels, no dilation, stricter MODIS thresholds, alternative negative ratio, and no historical-fire feature variants globally.",
            ["label_sensitivity.csv"],
            artifact_globs=["models/label_*"],
        )
        d = self.start(exp)
        g = filter_global(lab)
        self.write_table(
            d,
            "global_label_sensitivity.csv",
            select_table(
                g,
                [
                    ("experiment", "Variant"),
                    ("positive_rate", "Positive Rate"),
                    ("f1", "F1"),
                    ("f1_error", "F1 Error"),
                    ("average_precision", "PR-AUC"),
                    ("average_precision_error", "PR-AUC Error"),
                ],
                sort_by=["Variant"],
            ),
        )
        self.write_analysis(d, [top_metric_sentence(g, "experiment", "average_precision", prefix="Highest global label-sensitivity PR-AUC")])

        exp = Experiment(
            "19_label_sensitivity_by_region",
            "Label Sensitivity By Region",
            "Separates regional robustness of target-construction variants from the global label-sensitivity result.",
            ["label_sensitivity.csv"],
            artifact_globs=["models/label_*"],
        )
        d = self.start(exp)
        r = filter_non_global(lab)
        self.write_table(
            d,
            "regional_label_sensitivity.csv",
            select_table(
                r,
                [
                    ("region_display", "Region"),
                    ("experiment", "Variant"),
                    ("f1", "F1"),
                    ("f1_error", "F1 Error"),
                    ("average_precision", "PR-AUC"),
                    ("average_precision_error", "PR-AUC Error"),
                ],
                sort_by=["Region", "Variant"],
            ),
        )
        self.write_analysis(d, [top_metric_sentence(r, "experiment", "average_precision", prefix="Highest regional label row PR-AUC")])

        if not by.empty:
            exp = Experiment(
                "20_label_sensitivity_by_year",
                "Label Sensitivity By Year",
                "Reports annual target-construction sensitivity metrics for 2021-2025 separately from regional and global summaries.",
                ["label_sensitivity_by_year.csv"],
                artifact_globs=["models/label_*"],
            )
            d = self.start(exp)
            annual = annual_only(by)
            self.write_table(
                d,
                "yearly_label_sensitivity.csv",
                select_table(
                    annual,
                    [
                        ("region_display", "Region"),
                        ("period", "Year"),
                        ("experiment", "Variant"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Region", "Year", "Variant"],
                ),
            )
            self.write_table(
                d,
                "combined_period_label_sensitivity.csv",
                select_table(
                    combined_periods(by),
                    [
                        ("region_display", "Region"),
                        ("period", "Period"),
                        ("experiment", "Variant"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Region", "Period", "Variant"],
                ),
            )
            self.write_analysis(d, [top_metric_sentence(annual, "experiment", "average_precision", prefix="Highest annual label-sensitivity PR-AUC")])

    def build_lead(self) -> None:
        lead = self.dfs.get("lead_time_sensitivity.csv", pd.DataFrame())
        by = self.dfs.get("lead_time_sensitivity_by_year.csv", pd.DataFrame())
        if lead.empty:
            return
        exp = Experiment(
            "21_lead_time_sensitivity_global",
            "Lead-Time Sensitivity Global",
            "Compares CatBoost performance at 7-day, 14-day, and 30-day horizons on the global combined test set.",
            ["lead_time_sensitivity.csv"],
            ["lead_time_pr_auc", "lead_time_f1"],
            artifact_globs=["models/lead_time_*"],
        )
        d = self.start(exp)
        g = filter_global(lead)
        self.write_table(
            d,
            "global_lead_time_metrics.csv",
            select_table(
                g,
                [
                    ("lead_time_days", "Lead Days"),
                    ("f1", "F1"),
                    ("f1_error", "F1 Error"),
                    ("average_precision", "PR-AUC"),
                    ("average_precision_error", "PR-AUC Error"),
                    ("threshold", "Threshold"),
                ],
                sort_by=["Lead Days"],
            ),
        )
        self.write_analysis(
            d,
            [
                top_metric_sentence(g, "lead_time_days", "average_precision", prefix="Highest global lead-time PR-AUC"),
                "The 30-day horizon is intended for strategic preparedness and resource planning rather than same-week tactical dispatch.",
            ],
        )

        exp = Experiment(
            "22_lead_time_sensitivity_by_region",
            "Lead-Time Sensitivity By Region",
            "Separates regional horizon sensitivity from the global lead-time comparison.",
            ["lead_time_sensitivity.csv"],
            ["lead_time_pr_auc", "lead_time_f1"],
            artifact_globs=["models/lead_time_*"],
        )
        d = self.start(exp)
        r = filter_non_global(lead)
        self.write_table(
            d,
            "regional_lead_time_metrics.csv",
            select_table(
                r,
                [
                    ("region_display", "Region"),
                    ("lead_time_days", "Lead Days"),
                    ("f1", "F1"),
                    ("f1_error", "F1 Error"),
                    ("average_precision", "PR-AUC"),
                    ("average_precision_error", "PR-AUC Error"),
                ],
                sort_by=["Region", "Lead Days"],
            ),
        )
        self.write_analysis(d, [top_metric_sentence(r, "lead_time_days", "average_precision", prefix="Highest regional lead-time row PR-AUC")])

        if not by.empty:
            exp = Experiment(
                "23_lead_time_sensitivity_by_year",
                "Lead-Time Sensitivity By Year",
                "Reports annual horizon sensitivity for 2021-2025 separately from regional and global lead-time summaries.",
                ["lead_time_sensitivity_by_year.csv"],
                ["lead_time_pr_auc", "lead_time_f1"],
                artifact_globs=["models/lead_time_*"],
            )
            d = self.start(exp)
            annual = annual_only(by)
            self.write_table(
                d,
                "yearly_lead_time_metrics.csv",
                select_table(
                    annual,
                    [
                        ("region_display", "Region"),
                        ("period", "Year"),
                        ("lead_time_days", "Lead Days"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Region", "Year", "Lead Days"],
                ),
            )
            self.write_table(
                d,
                "combined_period_lead_time_metrics.csv",
                select_table(
                    combined_periods(by),
                    [
                        ("region_display", "Region"),
                        ("period", "Period"),
                        ("lead_time_days", "Lead Days"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Region", "Period", "Lead Days"],
                ),
            )
            self.write_analysis(d, [top_metric_sentence(annual, "lead_time_days", "average_precision", prefix="Highest annual lead-time PR-AUC")])

    def build_input_source(self) -> None:
        src = self.dfs.get("input_source_comparison.csv", pd.DataFrame())
        by = self.dfs.get("input_source_comparison_by_year.csv", pd.DataFrame())
        if src.empty:
            return
        required_source_rows = {
            "SEAS5/ECMWF -> SEAS5/ECMWF",
            "ERA5 -> ERA5",
            "ERA5 -> SEAS5/ECMWF",
            "ERA5 + SEAS5/ECMWF -> SEAS5/ECMWF",
        }
        src_global_check = filter_global(src)
        completed_check = src_global_check[
            src_global_check.get("status", pd.Series(index=src_global_check.index, dtype=str))
            .astype(str)
            .str.startswith("completed")
        ]
        present_source_rows = {
            required
            for required in required_source_rows
            if completed_check.get("experiment", pd.Series(dtype=str)).astype(str).str.startswith(required).any()
        }
        missing_source_rows = sorted(required_source_rows - present_source_rows)
        if missing_source_rows:
            raise RuntimeError(
                "Input source comparison is incomplete; refusing to organize a partial or one-source comparison. "
                f"Missing completed global rows: {missing_source_rows}"
            )
        exp = Experiment(
            "24_input_source_comparison_global",
            "Input Source Comparison Global",
            "Compares ERA5 and SEAS5/ECMWF source settings, including retrospective upper bound, operational setting, domain shift, and mixed training.",
            ["input_source_comparison.csv", "input_source_comparison_by_year.csv", "era5_feature_schema.csv", "era5_seas5_common_schema.csv"],
            ["input_source_comparison", "input_source_pr_auc", "input_source_f1"],
            artifact_globs=[
                "era5_source_comparison/**/*",
                "logs/era5_source_comparison.log",
            ],
            notes=[
                "Rows with status completed_available_domain are real ERA5-domain source-comparison runs on sampled case-control rows.",
                "Rows with blocked/schema status document unavailable full-domain ERA5 parity rather than hiding the attempted experiment.",
            ],
        )
        d = self.start(exp)
        src_global = filter_global(src)
        if src_global.empty:
            src_global = src
        completed = src_global[src_global.get("status", pd.Series(dtype=str)).astype(str).str.startswith("completed")]
        blocked = src_global[~src_global.index.isin(completed.index)]
        self.write_table(
            d,
            "input_source_metrics.csv",
            select_table(
                src_global,
                [
                    ("experiment", "Experiment"),
                    ("status", "Status"),
                    ("f1", "F1"),
                    ("f1_error", "F1 Error"),
                    ("average_precision", "PR-AUC"),
                    ("average_precision_error", "PR-AUC Error"),
                ],
                sort_by=["Experiment"],
            ),
        )
        self.write_table(
            d,
            "input_source_status_summary.csv",
            select_table(
                src_global.assign(
                    metric_available=src_global.get("average_precision", pd.Series(index=src_global.index)).notna()
                ),
                [
                    ("experiment", "Experiment"),
                    ("status", "Status"),
                    ("region_display", "Region"),
                    ("metric_available", "Has Metrics"),
                    ("notes", "Notes"),
                ],
                sort_by=["Experiment"],
            ),
        )
        self.write_table(
            d,
            "input_source_notes.csv",
            select_table(
                src_global,
                [
                    ("experiment", "Experiment"),
                    ("status", "Status"),
                    ("interpretation", "Interpretation"),
                    ("roc_auc", "ROC-AUC"),
                    ("notes", "Notes"),
                ],
                sort_by=["Experiment"],
            ),
        )
        if not by.empty:
            self.write_table(
                d,
                "input_source_by_year_available.csv",
                select_table(
                    by,
                    [
                        ("experiment", "Experiment"),
                        ("status", "Status"),
                        ("year", "Year"),
                        ("region_display", "Region"),
                        ("f1", "F1"),
                        ("f1_error", "F1 Error"),
                        ("average_precision", "PR-AUC"),
                        ("average_precision_error", "PR-AUC Error"),
                    ],
                    sort_by=["Experiment", "Year", "Region"],
                ),
            )
        schema = self.dfs.get("era5_feature_schema.csv", pd.DataFrame())
        if not schema.empty:
            self.write_table(
                d,
                "era5_source_availability.csv",
                select_table(
                    schema,
                    [
                        ("variable", "Variable"),
                        ("era5_zarr_files", "ERA5 Files"),
                        ("era5_years", "ERA5 Years"),
                        ("available_domain", "Available Domain"),
                        ("ecmwf_zarr_files", "ECMWF Files"),
                        ("status", "Status"),
                    ],
                    sort_by=["Variable"],
                ),
            )
        self.write_analysis(
            d,
            [
                f"Completed source-comparison rows: {len(completed)}; blocked/schema rows: {len(blocked)}.",
                "ERA5 -> ERA5 is a retrospective upper bound, not an operational forecast.",
                "SEAS5/ECMWF -> SEAS5/ECMWF is the clean operational source setting.",
                "ERA5 -> SEAS5/ECMWF measures input-source domain shift.",
                "Mixed ERA5 + SEAS5/ECMWF training tests operational robustness from source augmentation.",
            ],
        )

    def build_representative_model_ablation(self) -> None:
        src = self.dfs.get("representative_model_ablation_era5_to_ecmwf_no_tp.csv", pd.DataFrame())
        by = self.dfs.get("representative_model_ablation_era5_to_ecmwf_no_tp_by_year.csv", pd.DataFrame())
        if src.empty:
            return
        exp = Experiment(
            "31_representative_model_ablation_era5_to_ecmwf_no_tp",
            "Representative Model Ablation ERA5 To ECMWF No Tp",
            "Compares a compact set of distinct model families on the ERA5-trained, SEAS5/ECMWF-tested no-tp source-shift setting.",
            [
                "representative_model_ablation_era5_to_ecmwf_no_tp.csv",
                "representative_model_ablation_era5_to_ecmwf_no_tp_by_year.csv",
            ],
            [
                "representative_model_ablation_era5_to_ecmwf_no_tp_pr_auc",
                "representative_model_ablation_era5_to_ecmwf_no_tp_f1",
            ],
            artifact_globs=[
                "representative_model_ablation/**/*",
                "logs/representative_model_ablation.log",
            ],
            notes=[
                "The model set is intentionally limited to representative families: boosted trees, bagged trees, linear classification, point-process GLM, and the best neural model.",
                "All rows use ERA5 validation thresholds and SEAS5/ECMWF test rows with `tp` removed.",
            ],
        )
        d = self.start(exp)
        global_rows = filter_global(src)
        if global_rows.empty:
            global_rows = src
        self.write_table(
            d,
            "representative_model_metrics.csv",
            select_table(
                global_rows,
                [
                    ("model_group", "Model Group"),
                    ("model", "Model"),
                    ("f1", "F1"),
                    ("f1_error", "F1 Error"),
                    ("average_precision", "PR-AUC"),
                    ("average_precision_error", "PR-AUC Error"),
                    ("roc_auc", "ROC-AUC"),
                ],
                sort_by=["Model Group", "Model"],
            ),
        )
        self.write_table(
            d,
            "representative_model_protocol.csv",
            select_table(
                global_rows,
                [
                    ("model_group", "Model Group"),
                    ("model", "Model"),
                    ("training_source", "Train"),
                    ("validation_source", "Validation"),
                    ("test_source", "Test"),
                    ("notes", "Notes"),
                ],
                sort_by=["Model Group", "Model"],
            ),
        )
        if not by.empty:
            by_global = filter_global(by)
            if by_global.empty:
                by_global = by
            self.write_table(
                d,
                "representative_model_by_year.csv",
                select_table(
                    by_global,
                    [
                        ("model", "Model"),
                        ("year", "Year"),
                        ("f1", "F1"),
                        ("average_precision", "PR-AUC"),
                        ("roc_auc", "ROC-AUC"),
                    ],
                    sort_by=["Model", "Year"],
                ),
            )
        self.write_analysis(
            d,
            [
                top_metric_sentence(global_rows, "model", "average_precision", prefix="Best representative model by PR-AUC"),
                top_metric_sentence(global_rows, "model", "f1", prefix="Best representative model by F1"),
                "This compact experiment is meant to replace overcrowded all-model views when discussing source-shift model-family behavior.",
            ],
        )

    def build_importance(self) -> None:
        native = self.dfs.get("feature_importance_native.csv", pd.DataFrame())
        grouped = self.dfs.get("grouped_permutation_importance.csv", pd.DataFrame())
        grouped_y = self.dfs.get("grouped_permutation_importance_by_year.csv", pd.DataFrame())
        climate_window = self.dfs.get("climate_window_permutation_importance.csv", pd.DataFrame())
        shap = self.dfs.get("shap_importance.csv", pd.DataFrame())
        neural = self.dfs.get("neural_feature_importance.csv", pd.DataFrame())
        if not native.empty:
            exp = Experiment(
                "25_feature_importance_native",
                "Native CatBoost Feature Importance",
                "Ranks individual features by CatBoost native importance for the best full model.",
                ["feature_importance_native.csv"],
                ["native_feature_importance_top30"],
                artifact_globs=["models/catboost_full*"],
            )
            d = self.start(exp)
            self.write_table(
                d,
                "native_feature_importance_top30.csv",
                select_table(
                    native,
                    [
                        ("rank", "Rank"),
                        ("feature", "Feature"),
                        ("group", "Group"),
                        ("importance", "Importance"),
                        ("normalized_importance", "Normalized"),
                    ],
                    sort_by=["Rank"],
                    head=30,
                ),
            )
            self.write_analysis(d, ["Native importance is model attribution, not causal proof."])

        if not grouped.empty:
            exp = Experiment(
                "26_grouped_permutation_importance",
                "Grouped Permutation Importance",
                "Measures group-level performance drops after permuting feature sources.",
                [
                    "grouped_permutation_importance.csv",
                    "grouped_permutation_importance_by_year.csv",
                    "climate_window_permutation_importance.csv",
                ],
                ["grouped_permutation_importance", "climate_window_permutation_importance"],
                artifact_globs=["models/catboost_full*"],
            )
            d = self.start(exp)
            self.write_table(
                d,
                "grouped_permutation_importance.csv",
                select_table(
                    grouped,
                    [
                        ("group", "Group"),
                        ("feature_count", "Features"),
                        ("pr_auc_drop", "PR-AUC Drop"),
                        ("pr_auc_drop_error", "Drop Error"),
                        ("f1_drop", "F1 Drop"),
                    ],
                    sort_by=["Group"],
                ),
            )
            if not climate_window.empty:
                self.write_table(
                    d,
                    "climate_window_permutation_importance.csv",
                    select_table(
                        climate_window,
                        [
                            ("group", "Window"),
                            ("feature_count", "Features"),
                            ("pr_auc_drop", "PR-AUC Drop"),
                            ("pr_auc_drop_error", "Drop Error"),
                            ("f1_drop", "F1 Drop"),
                        ],
                        sort_by=["Window"],
                    ),
                )
            if grouped_y is not None and not grouped_y.empty:
                self.write_table(
                    d,
                    "grouped_permutation_importance_by_year.csv",
                    select_table(
                        grouped_y,
                        [
                            ("period", "Year"),
                            ("group", "Group"),
                            ("feature_count", "Features"),
                            ("pr_auc_drop", "PR-AUC Drop"),
                            ("f1_drop", "F1 Drop"),
                            ("roc_auc_drop", "ROC-AUC Drop"),
                        ],
                        sort_by=["Year", "Group"],
                    ),
                )
            self.write_analysis(
                d,
                [
                    top_metric_sentence(grouped, "group", "pr_auc_drop", prefix="Largest grouped PR-AUC drop"),
                    "Permutation importance quantifies model reliance under feature shuffling, not a causal effect.",
                ],
            )

        if not shap.empty:
            exp = Experiment(
                "27_shap_importance",
                "SHAP Importance",
                "Summarizes TreeSHAP mean absolute contributions for the best CatBoost model where feasible.",
                ["shap_importance.csv"],
                ["shap_summary"],
                artifact_globs=["models/catboost_full*"],
            )
            d = self.start(exp)
            self.write_table(
                d,
                "shap_importance_top30.csv",
                select_table(
                    shap,
                    [
                        ("rank", "Rank"),
                        ("feature", "Feature"),
                        ("group", "Group"),
                        ("mean_abs_shap", "Mean Abs SHAP"),
                    ],
                    sort_by=["Rank"],
                    head=30,
                ),
            )
            self.write_analysis(d, ["SHAP values are model attribution summaries and should not be framed as causal evidence."])

        if not neural.empty:
            exp = Experiment(
                "28_neural_feature_importance",
                "Neural Feature Importance",
                "Measures PR-AUC drops after permuting each input of the best global neural architecture.",
                ["neural_feature_importance.csv"],
                ["neural_feature_importance_top30"],
                artifact_globs=[
                    "models/nn_global_full_*",
                    "neural_model_metrics/nn_global_full_*",
                    "predictions/nn_global_full_*",
                ],
                notes=[
                    "The selected neural architecture is the highest global PR-AUC row in the main model comparison.",
                    "Permutation importance is model reliance on the sampled test tensor, not causal attribution.",
                ],
            )
            d = self.start(exp)
            self.write_table(
                d,
                "neural_feature_importance_top30.csv",
                select_table(
                    neural,
                    [
                        ("rank", "Rank"),
                        ("model", "Model"),
                        ("feature", "Feature"),
                        ("feature_type", "Type"),
                        ("delta_average_precision", "PR-AUC Drop"),
                        ("delta_f1", "F1 Drop"),
                    ],
                    sort_by=["Rank"],
                    head=30,
                ),
            )
            self.write_analysis(
                d,
                [
                    top_metric_sentence(neural, "feature", "delta_average_precision", prefix="Largest neural PR-AUC drop"),
                    "Dynamic, static, and categorical neural inputs are permuted with the trained network fixed.",
                ],
            )

    def build_failures(self) -> None:
        exp = Experiment(
            "30_failures_and_limitations",
            "Failures And Limitations",
            "Contains only true remaining blockers and limitations after attempted rebuilds and reruns.",
            [],
        )
        d = self.start(exp)
        failures = "\n\n".join(
            text
            for text in [
                source_text("failures.md"),
                source_text("failures_root.md"),
                source_text("sensitivity_failures.md"),
            ]
            if text
        )
        if failures:
            write_text(d / "artifacts" / "failures_original.md", failures)
            rows = []
            current = None
            for line in failures.splitlines():
                if line.startswith("## "):
                    current = line[3:].strip()
                if line.strip().startswith("- Reason:"):
                    rows.append({"Item": current or "Failure", "Note": line.strip().lstrip("- ")})
            if rows:
                self.write_table(d, "failure_notes.csv", pd.DataFrame(rows[:50]))
            elif "No failed" in failures:
                self.write_table(d, "failure_notes.csv", pd.DataFrame([{"Item": "Failures", "Note": "No failed sensitivity or ERA5 source-comparison experiments remain."}]))
            else:
                self.write_table(d, "failure_notes.csv", pd.DataFrame([{"Item": "Failures", "Note": "See artifacts/failures_original.md"}]))
        else:
            self.write_table(d, "failure_notes.csv", pd.DataFrame([{"Item": "Failures", "Note": "No failures.md source was available."}]))
        if failures and "No failed ERA5 input-source experiment remains" in failures:
            self.write_analysis(
                d,
                [
                    "ERA5 source-comparison rows are completed on the available processed ERA5 domain.",
                    "The remaining caveat is coverage: these are sampled/case-control diagnostics through 2024, not calibrated full-grid deployment metrics.",
                ],
            )
        else:
            self.write_analysis(d, ["Review the original failure artifact before making claims about unavailable experiments."])

    def build_probability_overlays(self) -> None:
        overlay = SHARED / "probability_overlays"
        selected_path = overlay / "tables" / "selected_probability_periods.csv"
        metrics_path = overlay / "tables" / "period_probability_metrics.csv"
        if not overlay.exists():
            return
        if selected_path.exists():
            selected = pd.read_csv(selected_path)
            metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        else:
            selected_frames: list[pd.DataFrame] = []
            metric_frames: list[pd.DataFrame] = []
            for path in sorted(overlay.glob("*/tables/selected_probability_periods.csv")):
                frame = pd.read_csv(path)
                frame.insert(0, "feature_source", path.parents[1].name)
                selected_frames.append(frame)
            for path in sorted(overlay.glob("*/tables/period_probability_metrics.csv")):
                frame = pd.read_csv(path)
                frame.insert(0, "feature_source", path.parents[1].name)
                metric_frames.append(frame)
            if not selected_frames:
                return
            selected = pd.concat(selected_frames, ignore_index=True)
            metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
        exp = Experiment(
            "29_full_grid_prediction_overlays",
            "Full-Grid Prediction Overlays",
            "Plots full-grid probability surfaces for the configured best neural prediction periods and overlays observed fire-positive grid-cell days.",
            [],
            [],
            artifact_globs=["probability_overlays/**/*"],
            notes=[
                "Overlay generation is controlled by the suite YAML config, not a standalone plotting CLI.",
                "Dense neural surfaces are cached under shared_artifacts/probability_overlays/artifacts when enabled.",
            ],
        )
        d = self.start(exp)
        self.write_table(
            d,
            "selected_probability_periods.csv",
            select_table(
                selected,
                [
                    ("region_display", "Region"),
                    ("feature_source", "Feature Source"),
                    ("period_rank", "Rank"),
                    ("model_name", "Model"),
                    ("period_start", "Period Start"),
                    ("period_end", "Period End"),
                    ("average_precision", "PR-AUC"),
                    ("weighted_brier_score", "Brier"),
                ],
                sort_by=["Region", "Rank"],
            ),
        )
        if not metrics.empty:
            self.write_table(
                d,
                "period_metric_summary.csv",
                select_table(
                    metrics.sort_values("average_precision", ascending=False).head(50),
                    [
                        ("region_display", "Region"),
                        ("feature_source", "Feature Source"),
                        ("model_name", "Model"),
                        ("period_start", "Period Start"),
                        ("period_end", "Period End"),
                        ("observed_positive_locations", "Fire Locations"),
                        ("average_precision", "PR-AUC"),
                        ("expected_observed_count_ratio", "E/O Count"),
                    ],
                ),
            )
        for suffix, subdir in [("*.png", "png"), ("*.pdf", "pdf")]:
            plot_sources = list((overlay / "plots" / subdir).glob(suffix))
            plot_sources.extend(overlay.glob(f"*/plots/{subdir}/{suffix}"))
            for src in sorted(plot_sources):
                mkdir(d / "plots" / subdir)
                if src.parent.parent.parent.parent == overlay:
                    dst_name = f"{src.parent.parent.parent.name}_{src.name}"
                else:
                    dst_name = src.name
                shutil.copy2(src, d / "plots" / subdir / dst_name)
        plot_count = sum(1 for _ in overlay.rglob("probability_overlay_*"))
        self.write_analysis(
            d,
            [
                f"Generated {plot_count} overlay plot files from {len(selected)} selected prediction period rows.",
                top_metric_sentence(selected, "model_name", "average_precision", prefix="Highest selected-period PR-AUC"),
            ],
        )

    def write_index(self) -> None:
        index = pd.DataFrame(self.index_rows)
        rows = "\n".join(
            f"- [{row['Experiment']}]({row['Folder']}/description.md): {row['Purpose']}"
            for row in self.index_rows
        )
        write_text(
            EXPERIMENTS / "index.md",
            f"""# Organized Revision Experiments

Each experiment folder contains:

- `description.md`: what the experiment studies.
- `analysis.md`: compact automatic interpretation.
- `tables/`: readable compact CSV tables when the experiment has presentation tables.
- `plots/png/` and `plots/pdf/`: copied only when plots exist.
- `artifacts/`: raw JSONL sources, schemas, manifests, and symlinks to reusable outputs when available.

## Experiments
{rows}
""",
        )
        write_text(
            EXPERIMENTS / "table_column_policy.md",
            f"# Table Column Policy\n\nAll CSV files under `experiments/*/tables/` are compact presentation tables with at most {MAX_PRESENTATION_TABLE_COLUMNS} columns. Raw wide outputs are archived as JSONL.GZ in `shared_artifacts/raw_tables_jsonl/` with schemas in `shared_artifacts/raw_table_schemas/`.",
        )
        index.to_csv(EXPERIMENTS / "experiment_index.csv", index=False)


def verify_outputs() -> None:
    violations = []
    for csv in ROOT.rglob("*.csv"):
        try:
            df = pd.read_csv(csv, nrows=1)
        except Exception:
            continue
        if len(df.columns) > MAX_PRESENTATION_TABLE_COLUMNS:
            violations.append((str(csv), len(df.columns)))
    if violations:
        raise RuntimeError(f"CSV column limit violations: {violations[:10]}")


def write_root_readme() -> None:
    write_text(
        ROOT / "README.md",
        """# Revision Experiments Complete

This directory has been reorganized into a clean experiment library.

- `experiments/`: reader-facing experiment folders with descriptions, analysis, narrow CSV tables, plots, and per-experiment artifacts when those outputs exist.
- `shared_artifacts/`: reusable raw sources, configs, logs, models, predictions, target caches, neural data, and original mixed plot files.
- `experiments/04_primary_full_grid_calibrated/`: calibrated full-grid prediction study with prevalence, contrast, risk concentration, count correction, and spatial-scale tables when available.
- `experiments/05_legacy_sampled_case_control/`: old undersampled-negative diagnostics retained for backwards comparison.
All remaining CSV files are compact presentation tables; wide raw tables were archived as JSONL.GZ plus schema JSON files under `shared_artifacts/`.

Start with [`experiments/index.md`](experiments/index.md).
""",
    )


def organize_results(root: Path = ROOT) -> None:
    set_root(root)
    cleanup_stale_artifacts()

    if not ROOT.exists():
        raise SystemExit(f"Missing result directory: {ROOT}")

    dfs = load_all_tables()

    if EXPERIMENTS.exists():
        shutil.rmtree(EXPERIMENTS)

    archive_root_outputs(dfs)

    builder = Builder(dfs)
    builder.build_metadata()
    builder.build_dataset()
    builder.build_primary_full_grid()
    builder.build_legacy_sampled()
    builder.build_main()
    builder.build_feature_ablation()
    builder.build_neural()
    builder.build_neural_feature_ablation()
    builder.build_label()
    builder.build_lead()
    builder.build_input_source()
    builder.build_representative_model_ablation()
    builder.build_importance()
    builder.build_failures()
    builder.build_probability_overlays()
    builder.write_index()

    write_root_readme()
    prune_empty_dirs(ROOT)
    verify_outputs()


def main(root: Path = ROOT) -> None:
    organize_results(root)


if __name__ == "__main__":
    main()
