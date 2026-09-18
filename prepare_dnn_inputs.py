#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal.*")
import uproot # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
import json
import argparse
import re

import classes
import inferencers
import utils

INPUT_METADATA_COLUMNS = {"source_run", "source_is_pedestal"}
FEATURE_VERSION_LEGACY = "legacy_v1"
FEATURE_VERSION_ALL_CHANNELS = "all_channels_multimodule_v1"
SUPPORTED_FEATURE_VERSIONS = {
    FEATURE_VERSION_LEGACY,
    FEATURE_VERSION_ALL_CHANNELS,
}
DNN_INPUT_MANIFEST_FILENAME = "dnn_input_manifest.json"
PER_CHANNEL_INPUT_COLUMNS = [
    "channel_indices",
    "erx_indices",
    "cell_area_fraction",
    "adc_unconnected_00",
    "adc_unconnected_01",
    "adc_unconnected_02",
    "adc_unconnected_03",
]

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare analysis inputs from a beam run using pedestals from a pedestal run."
    )
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        default=112050,
        help="Beam run number to prepare DNN inputs for (e.g. 112050).",
    )
    parser.add_argument(
        "-p",
        "--pedestal-run",
        type=int,
        default=112044,
        help="Run number from which pedestals (means/stds) were computed (e.g. 112044).",
    )
    parser.add_argument(
        "-m",
        "--modules",
        nargs="+",
        metavar="MOD",
        default=[
            # Electron runs Sep2025 TB
            # "ML_F3WC_IH0182", "ML_F3WC_IH0190", "ML_F3WC_IH0191", "ML_F3WC_IH0192",
            # "ML_F3WC_IH0194", "ML_F3WC_IH0196", "ML_F3WC_IH0197", "ML_F3WC_IH0198",
            "ML_F3WC_IH0182",
        ],
        help="List of module names to process (e.g. ML_F3WC_IH0182 ML_F3WC_IH0190 ...).",
    )
    parser.add_argument(
        "--module-for-correction",
        type=str,
        required=True,
        help="Module from which the correction artifacts should be loaded.",
    )
    parser.add_argument(
        "-s",
        "--selection",
        type=str,
        default="full",
        metavar="SEL",
        help="Only use events for which the column 'selection_{SEL}' is true. These need to be constructed and added to the df before, of course.",
    )
    parser.add_argument(
        "-c",
        "--column-tag",
        type=str,
        default="",
        help="Column tag to be appended at the end of 'adc_ch{i:03d}_pedsub'.",
    )
    parser.add_argument(
        "--plot-inputs",
        action="store_true",
        help="Write one full-distribution histogram per DNN input after preparing the input chunks.",
    )
    parser.add_argument(
        "--feature-version",
        choices=sorted(SUPPORTED_FEATURE_VERSIONS),
        default=FEATURE_VERSION_LEGACY,
        help="DNN feature schema. The default preserves existing prepared inputs.",
    )
    parser.add_argument(
        "--module-vocabulary",
        nargs="+",
        default=None,
        help=(
            "Ordered training-module vocabulary for the multi-module schema. "
            "One additional UNKNOWN one-hot entry is always reserved."
        ),
    )

    args = parser.parse_args()





    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            run_for_correction=args.run,
            module_for_correction=args.module_for_correction,
            derive_correction=True,
            selection_for_correction=args.selection,
            standardize_std=False,
            inputfoldertag="",
        ) 
        for x in args.modules
    ]
    feature_spec = make_feature_spec(
        feature_version=args.feature_version,
        module_vocabulary=args.module_vocabulary or args.modules,
    )
    for cfg in cfgs:
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=args.selection)
        prepare_dnn_inputs(
            cfg=cfg,
            column_tag=args.column_tag,
            inferencer=inferencer,
            plot_inputs=args.plot_inputs,
            feature_spec=feature_spec,
        )



def _load_cell_area_fractions(cfg, adc_channel_indices):
    module_type = cfg.modulename[:4].replace("-", "_")
    cellareas_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data", "cellareas.json")
    )

    with open(cellareas_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if module_type not in payload:
        raise KeyError(
            f"Module type '{module_type}' not found in cell area file {cellareas_path}."
        )

    sfs = np.asarray(payload[module_type]["SF"], dtype=np.float32)
    if sfs.shape[0] != cfg.nch:
        raise ValueError(
            f"Cell area count mismatch for module type '{module_type}': "
            f"expected {cfg.nch}, found {sfs.shape[0]}."
        )

    return sfs[np.asarray(adc_channel_indices, dtype=np.int64)]


def make_feature_spec(feature_version=FEATURE_VERSION_LEGACY, module_vocabulary=None):
    if feature_version not in SUPPORTED_FEATURE_VERSIONS:
        raise ValueError(
            f"Unsupported DNN feature version {feature_version!r}; "
            f"choose one of {sorted(SUPPORTED_FEATURE_VERSIONS)}."
        )
    modules = [] if module_vocabulary is None else [str(x) for x in module_vocabulary]
    if len(set(modules)) != len(modules):
        raise ValueError(f"Duplicate module names in DNN module vocabulary: {modules}")
    if feature_version == FEATURE_VERSION_LEGACY:
        modules = []
    if feature_version == FEATURE_VERSION_ALL_CHANNELS and not modules:
        raise ValueError("The all-channel multi-module schema requires a non-empty module vocabulary.")
    return {
        "feature_version": feature_version,
        "module_vocabulary": modules,
        "unknown_module_index": len(modules) if feature_version == FEATURE_VERSION_ALL_CHANNELS else None,
        "unknown_module_label": "UNKNOWN" if feature_version == FEATURE_VERSION_ALL_CHANNELS else None,
    }


def dnn_input_schema_id(feature_spec):
    """Return a stable folder name for one fully materialized input schema."""
    normalized_spec = make_feature_spec(
        feature_version=feature_spec["feature_version"],
        module_vocabulary=feature_spec.get("module_vocabulary"),
    )
    version = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized_spec["feature_version"])
    modules = compact_module_vocabulary(normalized_spec["module_vocabulary"])
    return f"{version}__{modules}"


def compact_module_vocabulary(module_vocabulary):
    groups = []
    for modulename in module_vocabulary:
        if not re.fullmatch(r"[A-Za-z0-9_]+", modulename):
            raise ValueError(
                f"Module name {modulename!r} cannot be used in a DNN input folder. "
                "Expected only letters, digits, and underscores."
            )
        match = re.match(r"^(.*\D)(\d+)$", modulename)
        if match is None:
            raise ValueError(
                f"Module name {modulename!r} cannot be abbreviated in a DNN input folder. "
                "Expected a trailing numeric identifier."
            )
        prefix_and_digits = match.groups()
        if groups and groups[-1][0] == prefix_and_digits[0]:
            groups[-1][1].append((modulename, prefix_and_digits[1]))
        else:
            groups.append((prefix_and_digits[0], [(modulename, prefix_and_digits[1])]))

    compact_groups = []
    for _, entries in groups:
        first_name, _ = entries[0]
        compact = first_name
        numeric_values = [digits for _, digits in entries]
        common_numeric_prefix = os.path.commonprefix(numeric_values) if len(numeric_values) > 1 else ""
        for _, digits in entries[1:]:
            suffix = digits[len(common_numeric_prefix):] or digits
            compact += f"-{suffix}"
        compact_groups.append(compact)
    return "_".join(compact_groups) or "no_modules"


def _manifest_matches_feature_spec(folder, feature_spec):
    path = os.path.join(folder, DNN_INPUT_MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return False
    return (
        payload.get("feature_version") == feature_spec["feature_version"]
        and list(payload.get("module_vocabulary", [])) == list(feature_spec["module_vocabulary"])
        and payload.get("unknown_module_index") == feature_spec["unknown_module_index"]
    )


def configure_dnn_input_folder(cfg, feature_spec, allow_legacy_fallback=False):
    """Point a config at inputs materialized for exactly this feature schema."""
    base = getattr(cfg, "dnn_training_input_base_folder", None)
    if base is None:
        base = cfg.dnn_training_input_folder
        cfg.dnn_training_input_base_folder = base
    folder = os.path.join(base, dnn_input_schema_id(feature_spec))
    if (
        allow_legacy_fallback
        and not os.path.isfile(os.path.join(folder, DNN_INPUT_MANIFEST_FILENAME))
        and _manifest_matches_feature_spec(base, feature_spec)
    ):
        folder = base
        print(f"Using compatible pre-schema-folder DNN inputs from {folder}")
    cfg.dnn_training_input_folder = folder
    return folder


def module_onehot_columns(feature_spec):
    if feature_spec["feature_version"] != FEATURE_VERSION_ALL_CHANNELS:
        return []
    return [f"module_onehot_{index:03d}" for index in range(len(feature_spec["module_vocabulary"]) + 1)]


def module_onehot_index(modulename, feature_spec):
    if feature_spec["feature_version"] != FEATURE_VERSION_ALL_CHANNELS:
        return None
    modules = feature_spec["module_vocabulary"]
    try:
        return modules.index(modulename)
    except ValueError:
        return int(feature_spec["unknown_module_index"])


def make_input_df(cfg, df, adc_channel_indices, column_tag, feature_spec=None):

    feature_spec = feature_spec or make_feature_spec()
    feature_version = feature_spec["feature_version"]

    cm_columns = [f"cm_erx{idx:02}_pedsub" for idx in range(cfg.ncmchannels)]
    df_inputs = df[cm_columns].copy().astype("float32")
    if "source_run" in df.columns:
        df_inputs["source_run"] = df["source_run"]
    else:
        df_inputs["source_run"] = cfg.run
    if "source_is_pedestal" in df.columns:
        df_inputs["source_is_pedestal"] = df["source_is_pedestal"]
# 
    adc_channel_indices_mean = sum(adc_channel_indices) / len(adc_channel_indices)

    erx_indices = [i // cfg.nch_per_erx for i in adc_channel_indices]
    erx_indices_mean = sum(erx_indices) / len(erx_indices)

    # channel indices as list, one value per channel
    df_inputs["channel_indices"] = [[x - adc_channel_indices_mean for x in adc_channel_indices]] * len(df_inputs)

    # ERX indices as list, one value per channel
    df_inputs["erx_indices"] = [[x - erx_indices_mean for x in erx_indices]] * len(df_inputs)

    # Relative cell area, using the same SF normalization as the summary plots.
    cell_area_fractions = _load_cell_area_fractions(cfg=cfg, adc_channel_indices=adc_channel_indices)
    df_inputs["cell_area_fraction"] = [cell_area_fractions.tolist()] * len(df_inputs)

    # unconnected channels on the same e-Rx as the channel (legacy schema)
    def _build_unconnected_feature(offset: int, out_col: str) -> np.ndarray:
        src_cols = [f"adc_ch{(x * cfg.nch_per_erx + offset):03d}_pedsub{column_tag}" for x in erx_indices]
        arr = df[src_cols].to_numpy(dtype=np.float32, copy=True)
        bad = ~np.isfinite(arr)
        n_bad = int(np.count_nonzero(bad))
        if n_bad > 0:
            n_rows_bad = int(np.count_nonzero(np.any(bad, axis=1)))
            print(
                f"[warning] Replacing {n_bad} non-finite value(s) in '{out_col}' "
                f"(rows affected: {n_rows_bad}) with 0."
            )
            arr[bad] = np.float32(0.)
        df_inputs[out_col] = arr.tolist()
        return arr

    if feature_version == FEATURE_VERSION_LEGACY:
        _build_unconnected_feature(offset=8, out_col="adc_unconnected_00")
        _build_unconnected_feature(offset=17, out_col="adc_unconnected_01")
        _build_unconnected_feature(offset=19, out_col="adc_unconnected_02")
        _build_unconnected_feature(offset=28, out_col="adc_unconnected_03")

    # # number of channels with toa and with tot
    df_inputs[f"nchtoa"] = df["nchtoa"]
    df_inputs[f"nchtot"] = df["nchtot"]
    if feature_version == FEATURE_VERSION_ALL_CHANNELS:
        for column in ["nchadcgt10", "nchadcgt50", "nchadcgt200", "nchadcgt500"]:
            df_inputs[column] = df[column].astype("float32")

        onehot_index = module_onehot_index(cfg.modulename, feature_spec)
        for index, column in enumerate(module_onehot_columns(feature_spec)):
            df_inputs[column] = np.float32(index == onehot_index)

        source_columns = [f"adc_ch{channel:03d}_pedsub_nocut" for channel in adc_channel_indices]
        missing = [column for column in source_columns if column not in df.columns]
        if missing:
            raise KeyError(
                "The all-channel DNN schema requires unmasked pedestal-subtracted ADC "
                f"columns. Missing {len(missing)} column(s), beginning with {missing[:3]}. "
                "Regenerate analysis inputs with convert_to_df.py."
            )
        all_channel_values = df[source_columns].to_numpy(dtype=np.float32, copy=True)
        nonfinite = ~np.isfinite(all_channel_values)
        if np.any(nonfinite):
            print(
                f"[warning] Replacing {int(np.count_nonzero(nonfinite))} non-finite "
                "all-channel ADC input value(s) with 0."
            )
            all_channel_values[nonfinite] = np.float32(0.0)
        all_channel_columns = [f"adc_allch_{channel:03d}" for channel in adc_channel_indices]
        df_all_channels = pd.DataFrame(
            all_channel_values,
            columns=all_channel_columns,
            index=df_inputs.index,
            dtype="float32",
        )
        df_inputs = pd.concat([df_inputs, df_all_channels], axis=1)

    return df_inputs


def prepare_dnn_inputs(
    cfg,
    column_tag,
    inferencer,
    nch_to_use=None,
    plot_inputs: bool = False,
    feature_spec=None,
):
    print("Hello from prepare_dnn_inputs()!")

    # Open file and load tree
    print(f"Preparing DNN inputs from Run{cfg.run} for module {cfg.modulename}...")
    feature_spec = feature_spec or make_feature_spec()
    configure_dnn_input_folder(cfg=cfg, feature_spec=feature_spec)
    # Each complete feature schema has an independent materialized input set.
    os.makedirs(name=cfg.dnn_training_input_folder, exist_ok=True)

    if nch_to_use is None:
        nch_to_use = cfg.nch
    if (
        feature_spec["feature_version"] == FEATURE_VERSION_ALL_CHANNELS
        and nch_to_use != cfg.nch
    ):
        raise ValueError(
            f"The all-channel DNN schema requires all {cfg.nch} channels; "
            f"received nch_to_use={nch_to_use}."
        )

    adc_channel_indices = [x for x in range(nch_to_use)]
    target_columns = [f"adc_ch{idx:03}_pedsub{column_tag}" for idx in adc_channel_indices]
    event_ids_all = []
    chunk_indices = []

    def write_df(df: pd.DataFrame, filename: str, index: bool = True) -> None:
        utils.write_via_tmpdir(
            outfilename=os.path.join(cfg.dnn_training_input_folder, filename),
            suffix=".parquet",
            writer_fn=lambda tmp, data=df, use_index=index: data.to_parquet(tmp, engine="pyarrow", index=use_index, compression="zstd"),
        )

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):
        df_targets = df_chunk[target_columns].copy().astype("float32")
        df_inputs = make_input_df(
            cfg=cfg,
            df=df_chunk,
            adc_channel_indices=adc_channel_indices,
            column_tag=column_tag,
            feature_spec=feature_spec,
        )

        print(df_targets)
        print(df_inputs)

        # write input and target chunks
        write_df(df_targets, f"targets_chunk{idx:03d}.parquet")
        write_df(df_inputs, f"inputs_chunk{idx:03d}.parquet")
        event_ids_all.append(df_chunk.index.to_numpy(np.int64))
        chunk_indices.append(idx)

    event_ids = np.unique(np.concatenate(event_ids_all))
    rng_split = np.random.default_rng(6789)
    perm = rng_split.permutation(len(event_ids))
    test_frac = 0.2
    n_test = int(round(test_frac * len(event_ids)))
    test_ids = event_ids[perm[:n_test]]

    df_split = pd.DataFrame(
        {
            "event_id_global": event_ids,
            "split": np.where(np.isin(event_ids, test_ids), "test", "train"),
        }
    )
    write_df(df_split, "event_split_train_test.parquet", index=False)
    print(df_split)
    write_source_run_channel_weights(
        cfg=cfg,
        chunk_indices=chunk_indices,
        target_columns=target_columns,
        split_map=dict(zip(df_split["event_id_global"].to_numpy(np.int64), df_split["split"].astype(str).to_numpy())),
    )
    input_manifest = dict(feature_spec)
    input_manifest.update(
        {
            "module": cfg.modulename,
            "nch": cfg.nch,
            "column_tag": column_tag,
            "per_channel_columns": (
                ["channel_indices", "erx_indices", "cell_area_fraction"]
                if feature_spec["feature_version"] == FEATURE_VERSION_ALL_CHANNELS
                else list(PER_CHANNEL_INPUT_COLUMNS)
            ),
        }
    )
    utils.write_via_tmpdir(
        outfilename=os.path.join(cfg.dnn_training_input_folder, DNN_INPUT_MANIFEST_FILENAME),
        suffix=".json",
        writer_fn=lambda tmp, payload=input_manifest: _write_json(tmp, payload),
    )
    if plot_inputs:
        plot_dnn_input_distributions(cfg=cfg, chunk_indices=chunk_indices)

    print(f"--> Wrote input, target, and split DFs to: {cfg.dnn_training_input_folder}")


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _iter_input_column_arrays(series: pd.Series, block_rows: int = 2048):
    """Yield flat numeric blocks from either a scalar or list-valued input column."""
    first_value = next((value for value in series if value is not None), None)
    is_per_channel = isinstance(first_value, (list, tuple, np.ndarray))
    if not is_per_channel:
        yield series.to_numpy(dtype=np.float64, copy=False).reshape(-1)
        return

    values = series.to_numpy(copy=False)
    for start in range(0, len(values), block_rows):
        arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values[start:start + block_rows] if value is not None]
        if arrays:
            yield np.concatenate(arrays)


def _input_plot_edges(minimum: float, maximum: float, bins: int) -> np.ndarray:
    if minimum == maximum:
        padding = max(0.5, abs(minimum) * 0.05)
        return np.asarray([minimum - padding, maximum + padding], dtype=np.float64)
    return np.linspace(minimum, maximum, bins + 1, dtype=np.float64)


def plot_dnn_input_distributions(cfg, chunk_indices, bins: int = 100) -> None:
    """Plot the complete distribution of every feature written to inputs_chunk*.parquet."""
    import matplotlib.pyplot as plt  # type: ignore

    if bins <= 0:
        raise ValueError(f"Input histogram bin count must be positive, got {bins}.")
    if not chunk_indices:
        raise RuntimeError("Cannot plot DNN inputs because no input chunks were written.")

    first_path = os.path.join(cfg.dnn_training_input_folder, f"inputs_chunk{chunk_indices[0]:03d}.parquet")
    first_df = pd.read_parquet(first_path)
    first_columns = list(first_df.columns)
    per_event_names = [
        column
        for column in first_columns
        if column not in INPUT_METADATA_COLUMNS and column not in PER_CHANNEL_INPUT_COLUMNS
    ]
    per_channel_names = [column for column in PER_CHANNEL_INPUT_COLUMNS if column in first_columns]
    feature_names = per_event_names + per_channel_names
    if not feature_names:
        raise RuntimeError(f"No DNN feature columns found in {first_path}.")

    stats = {
        name: {
            "minimum": np.inf,
            "maximum": -np.inf,
            "count": 0,
            "nonfinite": 0,
            "sum": 0.0,
            "sum2": 0.0,
        }
        for name in feature_names
    }

    # First streaming pass: establish exact ranges and summary statistics.
    for idx in chunk_indices:
        input_path = os.path.join(cfg.dnn_training_input_folder, f"inputs_chunk{idx:03d}.parquet")
        df_inputs = first_df[feature_names] if idx == chunk_indices[0] else pd.read_parquet(input_path, columns=feature_names)
        for name in feature_names:
            for values in _iter_input_column_arrays(df_inputs[name]):
                finite = np.isfinite(values)
                finite_values = values[finite]
                stats[name]["nonfinite"] += int(values.size - finite_values.size)
                if finite_values.size == 0:
                    continue
                stats[name]["minimum"] = min(stats[name]["minimum"], float(finite_values.min()))
                stats[name]["maximum"] = max(stats[name]["maximum"], float(finite_values.max()))
                stats[name]["count"] += int(finite_values.size)
                stats[name]["sum"] += float(finite_values.sum(dtype=np.float64))
                stats[name]["sum2"] += float(np.square(finite_values).sum(dtype=np.float64))
    del df_inputs
    del first_df

    edges = {}
    counts = {}
    for name in feature_names:
        if stats[name]["count"] == 0:
            continue
        edges[name] = _input_plot_edges(stats[name]["minimum"], stats[name]["maximum"], bins=bins)
        counts[name] = np.zeros(len(edges[name]) - 1, dtype=np.int64)

    # Second streaming pass: fill fixed-bin histograms without retaining all samples.
    for idx in chunk_indices:
        input_path = os.path.join(cfg.dnn_training_input_folder, f"inputs_chunk{idx:03d}.parquet")
        df_inputs = pd.read_parquet(input_path, columns=feature_names)
        for name in feature_names:
            if name not in edges:
                continue
            for values in _iter_input_column_arrays(df_inputs[name]):
                finite_values = values[np.isfinite(values)]
                if finite_values.size:
                    counts[name] += np.histogram(finite_values, bins=edges[name])[0]

    plot_dir = os.path.join(cfg.plotfolder_base, "dnn_inputs")
    os.makedirs(plot_dir, exist_ok=True)
    for input_index, name in enumerate(feature_names):
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
        output_path = os.path.join(plot_dir, f"{input_index:02d}_{safe_name}.pdf")
        fig, ax = plt.subplots(figsize=(7.0, 5.0))

        if name in edges:
            widths = np.diff(edges[name])
            ax.bar(edges[name][:-1], counts[name], width=widths, align="edge", color="tab:blue", alpha=0.8)
            count = stats[name]["count"]
            mean = stats[name]["sum"] / count
            variance = max(stats[name]["sum2"] / count - mean * mean, 0.0)
            summary = (
                f"finite entries: {count:,}\n"
                f"non-finite entries: {stats[name]['nonfinite']:,}\n"
                f"mean: {mean:.6g}\n"
                f"std: {np.sqrt(variance):.6g}\n"
                f"range: [{stats[name]['minimum']:.6g}, {stats[name]['maximum']:.6g}]"
            )
        else:
            summary = f"finite entries: 0\nnon-finite entries: {stats[name]['nonfinite']:,}"
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)

        ax.text(
            0.98,
            0.98,
            summary,
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
        ax.set_title(f"DNN input {input_index}: {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Event-channel entries" if name in PER_CHANNEL_INPUT_COLUMNS else "Events")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Wrote DNN input plot: {output_path}")


def write_source_run_channel_weights(cfg, chunk_indices, target_columns, split_map):
    counts = {"train": {}, "test": {}}

    for idx in chunk_indices:
        inputs_path = os.path.join(cfg.dnn_training_input_folder, f"inputs_chunk{idx:03d}.parquet")
        targets_path = os.path.join(cfg.dnn_training_input_folder, f"targets_chunk{idx:03d}.parquet")
        df_inputs = pd.read_parquet(inputs_path)
        df_targets = pd.read_parquet(targets_path)
        if "source_run" not in df_inputs.columns:
            raise KeyError(f"Missing source_run in {inputs_path}; rerun DNN input preparation.")
        if list(df_targets.columns) != list(target_columns):
            raise ValueError(f"Unexpected target columns in {targets_path}.")

        source_runs = df_inputs["source_run"].to_numpy()
        splits = pd.Index(df_inputs.index.to_numpy(np.int64, copy=False)).map(split_map).to_numpy()
        if not np.all(np.isin(splits, ["train", "test"])):
            bad = df_inputs.index.to_numpy(np.int64, copy=False)[~np.isin(splits, ["train", "test"])]
            raise KeyError(f"Unknown split label while computing weights for events (showing up to 10): {bad[:10]}")
        targets = df_targets.to_numpy(np.float32, copy=False)
        valid = np.isfinite(targets)

        for split in ("train", "test"):
            rows_split = np.flatnonzero(splits == split)
            if rows_split.size == 0:
                continue
            for source_run in np.unique(source_runs[rows_split]):
                rows = rows_split[source_runs[rows_split] == source_run]
                count = valid[rows].sum(axis=0, dtype=np.int64)
                counts[split].setdefault(source_run, np.zeros(len(target_columns), dtype=np.int64))
                counts[split][source_run] += count

    norm_by_split = {}
    for split in ("train", "test"):
        n_valid = 0
        raw_sum = 0.0
        for count in counts[split].values():
            positive = count > 0
            n_valid += int(count[positive].sum())
            raw_sum += float(np.count_nonzero(positive))
        if n_valid == 0:
            raise RuntimeError(f"No valid targets found for split '{split}' while computing DNN weights.")
        norm_by_split[split] = raw_sum / float(n_valid)

    stats = {
        "train": {"n_valid": 0, "n_positive_weight": 0, "sum_weight": 0.0, "min": np.inf, "max": 0.0},
        "test": {"n_valid": 0, "n_positive_weight": 0, "sum_weight": 0.0, "min": np.inf, "max": 0.0},
    }

    for idx in chunk_indices:
        inputs_path = os.path.join(cfg.dnn_training_input_folder, f"inputs_chunk{idx:03d}.parquet")
        targets_path = os.path.join(cfg.dnn_training_input_folder, f"targets_chunk{idx:03d}.parquet")
        df_inputs = pd.read_parquet(inputs_path)
        df_targets = pd.read_parquet(targets_path)

        source_runs = df_inputs["source_run"].to_numpy()
        splits = pd.Index(df_inputs.index.to_numpy(np.int64, copy=False)).map(split_map).to_numpy()
        if not np.all(np.isin(splits, ["train", "test"])):
            bad = df_inputs.index.to_numpy(np.int64, copy=False)[~np.isin(splits, ["train", "test"])]
            raise KeyError(f"Unknown split label while writing weights for events (showing up to 10): {bad[:10]}")
        targets = df_targets.to_numpy(np.float32, copy=False)
        valid = np.isfinite(targets)
        weights = np.zeros(targets.shape, dtype=np.float32)

        for split in ("train", "test"):
            rows_split = np.flatnonzero(splits == split)
            if rows_split.size == 0:
                continue
            norm = norm_by_split[split]
            for source_run in np.unique(source_runs[rows_split]):
                rows = rows_split[source_runs[rows_split] == source_run]
                count = counts[split].get(source_run)
                if count is None:
                    raise RuntimeError(f"Missing count table for split={split}, source_run={source_run}.")
                valid_rows = valid[rows]
                missing = valid_rows & (count[None, :] <= 0)
                if np.any(missing):
                    raise RuntimeError(f"Valid target mapped to zero count for split={split}, source_run={source_run}.")
                row_weights = np.zeros(valid_rows.shape, dtype=np.float32)
                count_rows = np.broadcast_to(count[None, :], valid_rows.shape)
                row_weights[valid_rows] = (1.0 / count_rows[valid_rows]) / norm
                weights[rows] = row_weights

            valid_split = valid[rows_split]
            positive_split = weights[rows_split] > 0.0
            stats[split]["n_valid"] += int(np.count_nonzero(valid_split))
            stats[split]["n_positive_weight"] += int(np.count_nonzero(positive_split))
            if np.any(positive_split):
                positive_weights = weights[rows_split][positive_split]
                stats[split]["sum_weight"] += float(positive_weights.sum(dtype=np.float64))
                stats[split]["min"] = min(stats[split]["min"], float(positive_weights.min()))
                stats[split]["max"] = max(stats[split]["max"], float(positive_weights.max()))

        df_weights = pd.DataFrame(weights, index=df_targets.index, columns=df_targets.columns)
        utils.write_via_tmpdir(
            outfilename=os.path.join(cfg.dnn_training_input_folder, f"weights_chunk{idx:03d}.parquet"),
            suffix=".parquet",
            writer_fn=lambda tmp, data=df_weights: data.to_parquet(tmp, engine="pyarrow", index=True, compression="zstd"),
        )

    for split in ("train", "test"):
        n_valid = stats[split]["n_valid"]
        n_positive = stats[split]["n_positive_weight"]
        mean_weight = stats[split]["sum_weight"] / max(1, n_positive)
        print(
            f"DNN source_run_channel weights ({split}): "
            f"valid_targets={n_valid}, positive_weights={n_positive}, "
            f"mean={mean_weight:.6g}, min={stats[split]['min'] if np.isfinite(stats[split]['min']) else 0.0:.6g}, max={stats[split]['max']:.6g}"
        )



if __name__ == '__main__':
  main()
