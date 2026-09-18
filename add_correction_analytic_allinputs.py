#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import classes
import compute_predictor_analytic_allinputs
import inferencers
import prepare_dnn_inputs
import utils


PREDICTION_SUFFIX = "_pred_analytic_allinputs"
RESIDUAL_SUFFIX = "_resid_analytic_allinputs"


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_analytic_allinputs_artifacts(cfg):
    folder = cfg.analytic_predictor_folder
    manifest_path = os.path.join(
        folder, compute_predictor_analytic_allinputs.MANIFEST_FILENAME
    )
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Missing analytic-all-inputs manifest {manifest_path}. "
            "Derive it first with ./derive.py --analytic-allinputs."
        )
    manifest = _read_json(manifest_path)
    if manifest.get("method") != "analytic_allinputs":
        raise ValueError(f"Unexpected analytic-all-inputs method in {manifest_path}.")

    weights_path = os.path.join(
        folder,
        manifest.get(
            "weights_file", compute_predictor_analytic_allinputs.WEIGHTS_FILENAME
        ),
    )
    intercepts_path = os.path.join(
        folder,
        manifest.get(
            "intercepts_file", compute_predictor_analytic_allinputs.INTERCEPTS_FILENAME
        ),
    )
    weights = pd.read_parquet(weights_path)
    intercepts = pd.read_parquet(intercepts_path)
    feature_names = list(manifest.get("feature_names", []))
    target_columns = list(manifest.get("target_columns", []))
    if list(weights.columns) != feature_names:
        raise ValueError("Analytic-all-inputs weight columns do not match its manifest.")
    if list(weights.index) != target_columns:
        raise ValueError("Analytic-all-inputs weight rows do not match its manifest targets.")
    if list(intercepts.index) != target_columns or list(intercepts.columns) != ["intercept"]:
        raise ValueError("Analytic-all-inputs intercept artifact does not match its manifest.")
    if len(target_columns) != cfg.nch:
        raise ValueError(
            f"Analytic-all-inputs predictor expects {len(target_columns)} channels, "
            f"but module {cfg.modulename!r} has {cfg.nch}."
        )

    for channel_index, feature_name in enumerate(
        manifest.get("excluded_target_features", [f"adc_allch_{i:03d}" for i in range(cfg.nch)])
    ):
        if feature_name not in weights.columns:
            raise KeyError(f"Missing target ADC feature {feature_name!r} in predictor weights.")
        target_column = target_columns[channel_index]
        if not np.isclose(weights.loc[target_column, feature_name], 0.0):
            raise ValueError(
                f"Target leakage guard failed: weight for target {channel_index}, "
                f"feature {feature_name!r} is not zero."
            )
    return manifest, weights, intercepts


def predict_analytic_allinputs(df_inputs, feature_names, weights, intercepts) -> np.ndarray:
    actual_feature_names = list(df_inputs.columns)
    if actual_feature_names != list(feature_names):
        raise ValueError(
            "Applied analytic-all-inputs feature order does not match its manifest.\n"
            f"Saved: {list(feature_names)}\nApply: {actual_feature_names}"
        )
    x = df_inputs.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite analytic-all-inputs feature encountered during application.")
    predictions = x @ weights.to_numpy(dtype=np.float64, copy=False).T
    predictions += intercepts["intercept"].to_numpy(dtype=np.float64, copy=False)[None, :]
    return predictions.astype(np.float32, copy=False)


def add_correction_analytic_allinputs(cfg, inferencer, column_tag="") -> None:
    print("Hello from add_correction_analytic_allinputs()!")
    print(f"Loading analytic-all-inputs predictor from {cfg.analytic_predictor_folder}")
    manifest, weights, intercepts = load_analytic_allinputs_artifacts(cfg=cfg)
    feature_spec = prepare_dnn_inputs.make_feature_spec(
        feature_version=manifest["feature_version"],
        module_vocabulary=manifest.get("module_vocabulary", []),
    )
    if feature_spec["feature_version"] != prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS:
        raise ValueError("The analytic-all-inputs predictor requires an all-channel feature schema.")

    module_index = prepare_dnn_inputs.module_onehot_index(cfg.modulename, feature_spec)
    if module_index == feature_spec["unknown_module_index"]:
        print(
            f"Module {cfg.modulename!r} was not used for derivation; activating reserved "
            f"UNKNOWN one-hot index {module_index}."
        )

    feature_names = list(manifest["feature_names"])
    target_columns = list(manifest["target_columns"])
    expected_target_columns = [
        f"adc_ch{channel:03d}_pedsub{column_tag}" for channel in range(cfg.nch)
    ]
    if target_columns != expected_target_columns:
        raise ValueError(
            "Target columns requested during application do not match the derived predictor.\n"
            f"Saved: {target_columns}\nApply: {expected_target_columns}"
        )
    per_channel_columns = list(manifest.get("per_channel_columns_omitted", []))
    adc_channel_indices = list(range(cfg.nch))

    for chunk_index, df_chunk in enumerate(inferencer.full_df_iter()):
        prepared = prepare_dnn_inputs.make_input_df(
            cfg=cfg,
            df=df_chunk,
            adc_channel_indices=adc_channel_indices,
            column_tag=column_tag,
            feature_spec=feature_spec,
        )
        actual_feature_names = compute_predictor_analytic_allinputs.event_feature_names(
            prepared, per_channel_columns
        )
        predictions = predict_analytic_allinputs(
            df_inputs=prepared[actual_feature_names],
            feature_names=feature_names,
            weights=weights,
            intercepts=intercepts,
        )
        measurements = df_chunk[target_columns].to_numpy(dtype=np.float32, copy=False)
        residuals = (measurements - predictions).astype(np.float32, copy=False)

        prediction_columns = [f"{column}{PREDICTION_SUFFIX}" for column in target_columns]
        residual_columns = [f"{column}{RESIDUAL_SUFFIX}" for column in target_columns]
        predictions_df = pd.DataFrame(
            predictions,
            index=df_chunk.index,
            columns=prediction_columns,
            dtype="float32",
        )
        residuals_df = pd.DataFrame(
            residuals,
            index=df_chunk.index,
            columns=residual_columns,
            dtype="float32",
        )

        existing = [
            column
            for column in prediction_columns + residual_columns
            if column in df_chunk.columns
        ]
        if existing:
            df_chunk = df_chunk.drop(columns=existing)
        df_chunk = pd.concat([df_chunk, predictions_df, residuals_df], axis=1)
        df_chunk[f"adc_sum_pedsub{column_tag}{PREDICTION_SUFFIX}"] = predictions_df.sum(
            axis=1, skipna=True
        )
        df_chunk[f"adc_sum_pedsub{column_tag}{RESIDUAL_SUFFIX}"] = residuals_df.sum(
            axis=1, skipna=True
        )

        outfilename = os.path.join(
            cfg.analysis_inputs_folder, f"df_batch{chunk_index:03d}.parquet"
        )
        utils.write_via_tmpdir(
            outfilename=outfilename,
            suffix=".parquet",
            writer_fn=lambda tmp, chunk=df_chunk: chunk.to_parquet(
                tmp, engine="pyarrow", index=True, compression="zstd"
            ),
        )
        print(
            f"Wrote analytic-all-inputs predictions and residuals to {outfilename}, "
            "overwriting existing columns with the same names."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply an analytic all-inputs predictor.")
    parser.add_argument("--run", type=compute_predictor_analytic_allinputs.parse_run_arg, required=True)
    parser.add_argument("--pedestal-run", type=int, default=112044)
    parser.add_argument("--correction-run", type=compute_predictor_analytic_allinputs.parse_run_arg, required=True)
    parser.add_argument("--selection-for-correction", default="selection_trigtime")
    parser.add_argument("--modules", nargs="+", required=True)
    parser.add_argument("--module-for-correction", required=True)
    args = parser.parse_args()

    for module in args.modules:
        cfg = classes.AnalysisConfig(
            modulename=module,
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            run_for_correction=args.correction_run,
            module_for_correction=args.module_for_correction,
            selection_for_correction=args.selection_for_correction,
        )
        add_correction_analytic_allinputs(
            cfg=cfg,
            inferencer=inferencers.AnalysisTruthInferencer(cfg=cfg),
        )


if __name__ == "__main__":
    main()
