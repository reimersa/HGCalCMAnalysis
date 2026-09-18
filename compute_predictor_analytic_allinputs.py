#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import classes
import inferencers
import prepare_dnn_inputs
import utils


WEIGHTS_FILENAME = "analytic_allinputs_weights.parquet"
INTERCEPTS_FILENAME = "analytic_allinputs_intercepts.parquet"
MANIFEST_FILENAME = "analytic_allinputs_manifest.json"
DEFAULT_RCOND = 1.0e-10


def parse_run_arg(value: str):
    return int(value) if str(value).isdigit() else value


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def event_feature_names(df_inputs: pd.DataFrame, per_channel_columns) -> list[str]:
    excluded = set(per_channel_columns) | set(prepare_dnn_inputs.INPUT_METADATA_COLUMNS)
    return [column for column in df_inputs.columns if column not in excluded]


def solve_analytic_allinputs(
    *,
    feature_names,
    target_columns,
    n_events,
    sum_x,
    sum_xx,
    target_counts,
    sum_y,
    sum_xy,
    sum_x_target_valid,
    rcond=DEFAULT_RCOND,
):
    """Solve one centered linear predictor per target from streaming moments."""
    if n_events <= 0:
        raise RuntimeError("Cannot derive analytic-all-inputs predictor from zero events.")
    if not 0.0 < float(rcond) < 1.0:
        raise ValueError(f"rcond must be in (0, 1), got {rcond}.")

    feature_names = list(feature_names)
    target_columns = list(target_columns)
    target_counts = np.asarray(target_counts, dtype=np.float64)
    if np.any(target_counts <= 0):
        missing = [target_columns[index] for index in np.flatnonzero(target_counts <= 0)[:10]]
        raise RuntimeError(f"No finite training targets for channel(s): {missing}")

    mean_x = np.asarray(sum_x, dtype=np.float64) / float(n_events)
    covariance_xx = np.asarray(sum_xx, dtype=np.float64) / float(n_events)
    covariance_xx -= np.outer(mean_x, mean_x)
    covariance_xx = 0.5 * (covariance_xx + covariance_xx.T)

    mean_y = np.asarray(sum_y, dtype=np.float64) / target_counts
    mean_x_target_valid = np.asarray(sum_x_target_valid, dtype=np.float64) / target_counts[None, :]
    covariance_yx = np.asarray(sum_xy, dtype=np.float64).T / target_counts[:, None]
    covariance_yx -= mean_y[:, None] * mean_x_target_valid.T

    variances = np.maximum(np.diag(covariance_xx), 0.0)
    std = np.sqrt(variances)
    variance_tolerance = max(float(np.max(variances)), 1.0) * 1.0e-14
    nonconstant = variances > variance_tolerance

    weights = np.zeros((len(target_columns), len(feature_names)), dtype=np.float64)
    intercepts = np.zeros(len(target_columns), dtype=np.float64)
    excluded_target_features = []

    for channel_index, target_column in enumerate(target_columns):
        target_feature = f"adc_allch_{channel_index:03d}"
        if target_feature not in feature_names:
            raise KeyError(
                f"Target {target_column!r} requires feature {target_feature!r}, "
                "but it is absent from the prepared all-channel inputs."
            )
        target_feature_index = feature_names.index(target_feature)
        active = nonconstant.copy()
        active[target_feature_index] = False
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            raise RuntimeError(f"No usable predictors remain for target {target_column!r}.")

        active_std = std[active_indices]
        covariance_scaled = covariance_xx[np.ix_(active_indices, active_indices)]
        covariance_scaled = covariance_scaled / np.outer(active_std, active_std)
        cross_scaled = covariance_yx[channel_index, active_indices] / active_std
        scaled_weights = cross_scaled @ np.linalg.pinv(covariance_scaled, rcond=rcond)
        weights[channel_index, active_indices] = scaled_weights / active_std
        intercepts[channel_index] = mean_y[channel_index] - (
            weights[channel_index] @ mean_x_target_valid[:, channel_index]
        )
        excluded_target_features.append(target_feature)

        if channel_index == 0 or (channel_index + 1) % 25 == 0 or channel_index + 1 == len(target_columns):
            print(f"Solved analytic-all-inputs target {channel_index + 1}/{len(target_columns)}")

    return {
        "weights": weights,
        "intercepts": intercepts,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "target_counts": target_counts,
        "constant_features": [feature_names[index] for index in np.flatnonzero(~nonconstant)],
        "excluded_target_features": excluded_target_features,
    }


def _validate_input_manifest(cfg, feature_spec) -> dict:
    path = os.path.join(
        cfg.dnn_training_input_folder,
        prepare_dnn_inputs.DNN_INPUT_MANIFEST_FILENAME,
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing prepared-input manifest {path}. Run ./derive.py -i first."
        )
    payload = _read_json(path)
    if payload.get("feature_version") != feature_spec["feature_version"]:
        raise ValueError(
            f"Feature-version mismatch in {path}: found {payload.get('feature_version')!r}, "
            f"expected {feature_spec['feature_version']!r}."
        )
    if list(payload.get("module_vocabulary", [])) != list(feature_spec["module_vocabulary"]):
        raise ValueError(
            f"Module-vocabulary mismatch in {path}: found {payload.get('module_vocabulary')}, "
            f"expected {feature_spec['module_vocabulary']}."
        )
    if int(payload.get("nch", cfg.nch)) != cfg.nch:
        raise ValueError(f"Channel-count mismatch in {path}.")
    return payload


def compute_predictor_analytic_allinputs(
    cfgs,
    cfg_out,
    feature_spec,
    train_event_fractions,
    rcond=DEFAULT_RCOND,
) -> None:
    print("Hello from compute_predictor_analytic_allinputs()!")
    cfgs = list(cfgs)
    if not cfgs:
        raise ValueError("At least one module configuration is required.")
    if feature_spec["feature_version"] != prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS:
        raise ValueError("analytic_allinputs requires the all-channel DNN feature schema.")

    module_names = [cfg.modulename for cfg in cfgs]
    if module_names != list(feature_spec["module_vocabulary"]):
        raise ValueError(
            f"Configuration modules {module_names} do not match feature vocabulary "
            f"{feature_spec['module_vocabulary']}."
        )
    unknown_fraction_modules = sorted(set(train_event_fractions) - set(module_names))
    if unknown_fraction_modules:
        raise KeyError(f"Event fractions supplied for unknown modules: {unknown_fraction_modules}")

    sources = []
    per_channel_columns = None
    for cfg in cfgs:
        prepare_dnn_inputs.configure_dnn_input_folder(
            cfg=cfg,
            feature_spec=feature_spec,
            allow_legacy_fallback=True,
        )
        input_manifest = _validate_input_manifest(cfg=cfg, feature_spec=feature_spec)
        columns = list(input_manifest.get("per_channel_columns", []))
        if per_channel_columns is None:
            per_channel_columns = columns
        elif columns != per_channel_columns:
            raise ValueError(
                f"Prepared per-channel columns differ between modules: "
                f"expected {per_channel_columns}, got {columns}."
            )
        fraction = inferencers.validate_event_fraction(
            train_event_fractions.get(cfg.modulename, 1.0)
        )
        sources.append(
            inferencers.AnalysisDNNInferencer(
                cfg=cfg,
                split="train",
                per_channel_cols=per_channel_columns,
                event_fraction=fraction,
            )
        )

    feature_names = None
    target_columns = None
    n_events = 0
    sum_x = None
    sum_xx = None
    target_counts = None
    sum_y = None
    sum_xy = None
    sum_x_target_valid = None

    for source in sources:
        print(
            f"Accumulating module {source.cfg.modulename} with "
            f"train_event_fraction={source.event_fraction:g}"
        )
        for batch in source.batches:
            selected = source.apply_split(batch)
            df_inputs = selected.full_inputs_df
            df_targets = selected.full_targets_df
            if source.event_fraction < 1.0 and len(df_inputs):
                keep = inferencers.event_keep_mask(
                    df_inputs.index.to_numpy(np.int64), source.event_fraction
                )
                df_inputs = df_inputs.iloc[keep]
                df_targets = df_targets.iloc[keep]
            if len(df_inputs) == 0:
                continue
            if not df_inputs.index.equals(df_targets.index):
                raise ValueError("Prepared analytic inputs and targets have different event indices.")

            chunk_feature_names = event_feature_names(df_inputs, per_channel_columns)
            chunk_target_columns = list(df_targets.columns)
            if feature_names is None:
                feature_names = chunk_feature_names
                target_columns = chunk_target_columns
                n_features = len(feature_names)
                n_targets = len(target_columns)
                sum_x = np.zeros(n_features, dtype=np.float64)
                sum_xx = np.zeros((n_features, n_features), dtype=np.float64)
                target_counts = np.zeros(n_targets, dtype=np.float64)
                sum_y = np.zeros(n_targets, dtype=np.float64)
                sum_xy = np.zeros((n_features, n_targets), dtype=np.float64)
                sum_x_target_valid = np.zeros((n_features, n_targets), dtype=np.float64)
            elif chunk_feature_names != feature_names or chunk_target_columns != target_columns:
                raise ValueError("Prepared feature or target order changed between input chunks.")

            x = df_inputs[feature_names].to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(x).all():
                raise ValueError(
                    f"Non-finite analytic input found for module {source.cfg.modulename}."
                )
            y = df_targets[target_columns].to_numpy(dtype=np.float64, copy=False)
            valid_y = np.isfinite(y)
            y_clean = np.where(valid_y, y, 0.0)
            valid_float = valid_y.astype(np.float64, copy=False)

            n_events += x.shape[0]
            sum_x += x.sum(axis=0)
            sum_xx += x.T @ x
            target_counts += valid_y.sum(axis=0)
            sum_y += y_clean.sum(axis=0)
            sum_xy += x.T @ y_clean
            sum_x_target_valid += x.T @ valid_float

    if feature_names is None or target_columns is None:
        raise RuntimeError("No retained training events were found in the prepared inputs.")

    result = solve_analytic_allinputs(
        feature_names=feature_names,
        target_columns=target_columns,
        n_events=n_events,
        sum_x=sum_x,
        sum_xx=sum_xx,
        target_counts=target_counts,
        sum_y=sum_y,
        sum_xy=sum_xy,
        sum_x_target_valid=sum_x_target_valid,
        rcond=rcond,
    )

    weights = pd.DataFrame(
        result["weights"], index=target_columns, columns=feature_names, dtype="float64"
    )
    intercepts = pd.DataFrame(
        {"intercept": result["intercepts"]}, index=target_columns, dtype="float64"
    )
    output_folder = cfg_out.analytic_predictor_folder
    os.makedirs(output_folder, exist_ok=True)
    utils.write_via_tmpdir(
        outfilename=os.path.join(output_folder, WEIGHTS_FILENAME),
        suffix=".parquet",
        writer_fn=lambda tmp: weights.to_parquet(tmp, index=True, compression="zstd"),
    )
    utils.write_via_tmpdir(
        outfilename=os.path.join(output_folder, INTERCEPTS_FILENAME),
        suffix=".parquet",
        writer_fn=lambda tmp: intercepts.to_parquet(tmp, index=True, compression="zstd"),
    )

    manifest = dict(feature_spec)
    manifest.update(
        {
            "method": "analytic_allinputs",
            "training_modules": module_names,
            "train_event_fractions": {
                module: float(train_event_fractions.get(module, 1.0))
                for module in module_names
            },
            "split": "train",
            "n_train_events": int(n_events),
            "feature_names": feature_names,
            "target_columns": target_columns,
            "per_channel_columns_omitted": list(per_channel_columns),
            "target_adc_policy": "own adc_allch feature coefficient fixed to zero",
            "excluded_target_features": result["excluded_target_features"],
            "constant_features": result["constant_features"],
            "unknown_module_policy": "reserved one-hot coefficient fixed to zero when constant in training",
            "solver": "centered covariance pseudoinverse after feature scaling",
            "predictor_covariance_sample_policy": "all retained training events",
            "cross_covariance_sample_policy": "finite events for each target channel",
            "rcond": float(rcond),
            "weights_file": WEIGHTS_FILENAME,
            "intercepts_file": INTERCEPTS_FILENAME,
        }
    )
    utils.write_via_tmpdir(
        outfilename=os.path.join(output_folder, MANIFEST_FILENAME),
        suffix=".json",
        writer_fn=lambda tmp: _write_json(tmp, manifest),
    )
    print(
        f"Wrote analytic-all-inputs predictor for {len(target_columns)} targets, "
        f"{len(feature_names)} stored features, and {n_events} training events to {output_folder}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive one analytic all-input linear predictor per target channel."
    )
    parser.add_argument("--run", type=parse_run_arg, default="112044_112050_112060_112073_adcmax10")
    parser.add_argument("--pedestal-run", type=int, default=112044)
    parser.add_argument("--selection-for-correction", default="selection_trigtime")
    parser.add_argument("--modules", nargs="+", required=True)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--rcond", type=float, default=DEFAULT_RCOND)
    args = parser.parse_args()

    feature_spec = prepare_dnn_inputs.make_feature_spec(
        feature_version=prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS,
        module_vocabulary=args.modules,
    )
    cfgs = [
        classes.AnalysisConfig(
            modulename=module,
            run=args.run,
            derive_correction=True,
            selection_for_correction=args.selection_for_correction,
            run_for_pedestal=args.pedestal_run,
            run_for_correction=args.run,
            module_for_correction=module,
        )
        for module in args.modules
    ]
    model_group = args.modules[0] if len(args.modules) == 1 else f"MULTI_{'_'.join(args.modules)}"
    cfg_out = cfgs[0] if len(cfgs) == 1 else classes.AnalysisConfig(
        modulename=args.modules[0],
        run=args.run,
        run_for_pedestal=args.pedestal_run,
        run_for_correction=args.run,
        module_for_correction=model_group,
        selection_for_correction=args.selection_for_correction,
    )
    fractions = {
        module: inferencers.validate_event_fraction(args.train_fraction)
        for module in args.modules
    }
    compute_predictor_analytic_allinputs(
        cfgs=cfgs,
        cfg_out=cfg_out,
        feature_spec=feature_spec,
        train_event_fractions=fractions,
        rcond=args.rcond,
    )


if __name__ == "__main__":
    main()
