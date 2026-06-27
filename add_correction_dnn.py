#!/usr/bin/env python3

import argparse
import os
import json

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore

import classes
import dnn_models
import inferencers
import prepare_dnn_inputs
import functions_plot
import utils

INPUT_PREPROCESSING_TAG = "inputzscore"
INPUT_PREPROCESSING_FILENAME = "input_preprocessing.json"


def main():
    parser = argparse.ArgumentParser(description="Add DNN predictions/residuals to df_batch*.parquet (overwrite).")

    parser.add_argument("-r", "--run", type=int, default=112044)
    parser.add_argument("-p", "--pedestal-run", type=int, default=112044)

    parser.add_argument(
        "-m", "--modules",
        nargs="+",
        metavar="MOD",
        default=[
            "ML_F3WC_IH0180",
            "ML_F3WC_IH0182",
            "ML_F3WC_IH0190",
            "ML_F3WC_IH0191",
            "ML_F3WC_IH0192",
            "ML_F3WC_IH0194",
            "ML_F3WC_IH0196",
            "ML_F3WC_IH0197",
            "ML_F3WC_IH0198",
            "ML_F3WC_IH0199",
        ],
    )
    parser.add_argument(
        "--module-for-correction",
        type=str,
        required=True,
        help="Module from which the DNN correction artifacts should be loaded.",
    )
    parser.add_argument(
        "--selection-for-correction",
        type=str,
        default="",
        help="Optional selection tag encoded in the correction-artifact folder.",
    )

    # --- DNN config / checkpoint ---
    parser.add_argument("-n", "--nodes", nargs="+", type=int, required=True, help="Nodes per hidden layer (must match training).")
    parser.add_argument("-d", "--dropout", type=float, default=0.0, help="Dropout rate (must match training).")
    parser.add_argument("-t", "--tag", type=str, default="", help="Model tag (must match training only for model_string; weights load regardless).")

    # naming
    parser.add_argument(
        "--column-tag",
        type=str,
        default="",
        help="Input ADC column tag appended after 'adc_ch{i:03d}_pedsub'. Outputs are written as *_pred_dnn and *_resid_dnn.",
    )

    # inputs definition
    parser.add_argument(
        "--per-channel-cols",
        nargs="+",
        default=["channel_indices"],
        help="List-like per-channel columns in df_inputs. All other df_inputs columns are treated as per-event.",
    )

    # inference batching
    parser.add_argument(
        "--infer-batch",
        type=int,
        default=65536,
        help="Number of (event,channel) samples per forward pass when predicting (per-channel loop).",
    )
    parser.add_argument(
        "--preprocess-inputs",
        action="store_true",
        help="Apply saved DNN input z-score preprocessing before inference.",
    )


    parser.add_argument(
        "--plotfolder",
        type=str,
        required=True,
        help="Folder to save loss plots to.",
    )

    args = parser.parse_args()

    cfgs = [
        classes.AnalysisConfig(
            modulename=x,
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            run_for_correction=args.run,
            module_for_correction=args.module_for_correction,
            selection_for_correction=args.selection_for_correction,
            standardize_std=False,
            inputfoldertag="",
        )
        for x in args.modules
    ]

    for cfg in cfgs:
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)
        add_correction_dnn(cfg=cfg, inferencer=inferencer, nodes=args.nodes, dropout=args.dropout, tag=args.tag, column_tag=args.column_tag, per_channel_cols=args.per_channel_cols, infer_batch=args.infer_batch, plot_dir_loss=args.plotfolder, preprocess_inputs=args.preprocess_inputs)


def tag_with_input_preprocessing(tag: str, preprocess_inputs: bool) -> str:
    if not preprocess_inputs:
        return tag
    if INPUT_PREPROCESSING_TAG in tag.split("_"):
        return tag
    return f"{tag}_{INPUT_PREPROCESSING_TAG}" if tag else INPUT_PREPROCESSING_TAG


def load_input_preprocessing(modeldir: str, feature_names: list[str]):
    path = os.path.join(modeldir, INPUT_PREPROCESSING_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Requested DNN input preprocessing, but missing stats file: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not payload.get("enabled", False):
        raise ValueError(f"Input preprocessing stats file is not enabled: {path}")
    if payload.get("method") != "zscore":
        raise ValueError(f"Unsupported input preprocessing method in {path}: {payload.get('method')!r}")

    saved_features = list(payload.get("feature_names", []))
    if saved_features != list(feature_names):
        raise ValueError(
            "DNN input preprocessing feature order mismatch.\n"
            f"Saved features: {saved_features}\n"
            f"Apply features: {list(feature_names)}"
        )

    mean = np.asarray(payload.get("mean", []), dtype=np.float32)
    std = np.asarray(payload.get("std", []), dtype=np.float32)
    if mean.shape != std.shape or mean.shape[0] != len(feature_names):
        raise ValueError(
            f"Input preprocessing stats shape mismatch in {path}: "
            f"mean={mean.shape}, std={std.shape}, features={len(feature_names)}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError(f"Invalid input preprocessing mean/std values in {path}.")

    if not payload.get("targets_enabled", False):
        raise ValueError(f"Input preprocessing is enabled, but target preprocessing stats are missing in {path}.")
    if payload.get("target_method") != "per_channel_zscore":
        raise ValueError(f"Unsupported target preprocessing method in {path}: {payload.get('target_method')!r}")

    target_mean = np.asarray(payload.get("target_mean", []), dtype=np.float32)
    target_std = np.asarray(payload.get("target_std", []), dtype=np.float32)
    target_channels = list(payload.get("target_channels", []))
    if target_mean.shape != target_std.shape or target_mean.shape[0] == 0:
        raise ValueError(f"Target preprocessing stats shape mismatch in {path}: mean={target_mean.shape}, std={target_std.shape}")
    if target_channels != list(range(target_mean.shape[0])):
        raise ValueError(f"Unexpected target channel list in {path}: {target_channels[:10]}...")
    if not np.all(np.isfinite(target_mean)) or not np.all(np.isfinite(target_std)) or np.any(target_std <= 0.0):
        raise ValueError(f"Invalid target preprocessing mean/std values in {path}.")

    print(f"Loaded DNN input preprocessing stats from {path}")
    return {"mean": mean, "std": std, "target_mean": target_mean, "target_std": target_std}


def apply_input_preprocessing(x_np: np.ndarray, input_preprocessing) -> np.ndarray:
    if input_preprocessing is None:
        return x_np
    mean = input_preprocessing["mean"]
    std = input_preprocessing["std"]
    if x_np.shape[1] != mean.shape[0]:
        raise ValueError(f"Input preprocessing shape mismatch: x has {x_np.shape[1]} columns, stats have {mean.shape[0]}.")
    return ((x_np - mean[None, :]) / std[None, :]).astype(np.float32, copy=False)


def inverse_target_preprocessing(y_np: np.ndarray, channel_idx: int, input_preprocessing) -> np.ndarray:
    if input_preprocessing is None:
        return y_np
    target_mean = input_preprocessing["target_mean"]
    target_std = input_preprocessing["target_std"]
    if channel_idx < 0 or channel_idx >= target_mean.shape[0]:
        raise ValueError(f"Target inverse preprocessing got channel {channel_idx}, but stats have {target_mean.shape[0]} channel(s).")
    return (y_np * target_std[channel_idx] + target_mean[channel_idx]).astype(np.float32, copy=False)


def add_correction_dnn(cfg, inferencer, nodes: list[int], dropout: float, tag: str, column_tag: str, per_channel_cols: list[str], infer_batch: int, plot_dir_loss: str, preprocess_inputs: bool = False) -> None:
    print("Hello from add_correction_dnn()!")
    print(f"Loading checkpoint: {cfg.dnn_models_folder}")
    tag = tag_with_input_preprocessing(tag, preprocess_inputs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"DNN apply preprocess_inputs={preprocess_inputs}")

    columns_to_predict = [f"adc_ch{i:03d}_pedsub{column_tag}" for i in range(cfg.nch)]
    adc_channel_indices = [x for x in range(cfg.nch)]

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):
        print(f"Probing input/target files from chunk {idx:03d}...")
        df_inputs = prepare_dnn_inputs.make_input_df(cfg=cfg, df=df_chunk, adc_channel_indices=adc_channel_indices, column_tag=column_tag)
        metadata_cols = ["source_run", "source_is_pedestal"]
        per_event_cols = [c for c in df_inputs.columns if c not in per_channel_cols and c not in metadata_cols]
        feature_names = list(per_event_cols) + list(per_channel_cols)
        input_dim = len(per_event_cols) + len(per_channel_cols)
        break



    # infer C and the base adc column names from targets
    C = cfg.nch
    print(f"Per-event cols: {input_dim-len(per_channel_cols)} | per-channel cols: {len(per_channel_cols)} | input_dim={input_dim} | C={C}")

    # --- model ---
    model = dnn_models.build_per_channel_model(
        input_dim=input_dim,
        nodes_per_layer=nodes,
        dropout_rate=dropout,
        tag=tag,
    ).to(device)
    modeldir = os.path.join(cfg.dnn_models_folder, model.get_model_string())
    input_preprocessing = load_input_preprocessing(modeldir=modeldir, feature_names=feature_names) if preprocess_inputs else None

    state = torch.load(os.path.join(modeldir, "dnn_best.pth"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()



    print(f"Now plotting loss")
    functions_plot.plot_loss(modeldir=modeldir, plot_dir=plot_dir_loss)
    print(f"Plotted loss")
    # return

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):

        df_inputs = prepare_dnn_inputs.make_input_df(cfg=cfg, df=df_chunk, adc_channel_indices=adc_channel_indices, column_tag=column_tag)
        E = df_inputs.shape[0]

        x_evt = df_inputs[per_event_cols].to_numpy(np.float32, copy=False)
        ch_mats = inferencers.matrices_from_per_channel_cols(per_channel_cols=per_channel_cols, df=df_inputs, nch=cfg.nch)

        # predictions [E, C]
        preds = np.full((E, C), np.nan, dtype=np.float32)

        # predict channel-by-channel (keeps memory bounded)
        with torch.no_grad():
            for ch in range(C):
                # build feature matrix for all events at this channel: [E, Fevt + Fch]
                ch_feats = [ch_mats[ccol][:, ch][:, None] for ccol in per_channel_cols]  # each [E,1]
                X = np.concatenate([x_evt] + ch_feats, axis=1).astype(np.float32, copy=False)  # [E, F]
                X = apply_input_preprocessing(X, input_preprocessing)

                # torch inference in batches
                out = np.empty((E,), dtype=np.float32)
                for start in range(0, E, infer_batch):
                    stop = min(start + infer_batch, E)
                    xb = torch.from_numpy(X[start:stop]).to(device=device, dtype=torch.float32)
                    model_out = model(xb)
                    out[start:stop] = model_out.detach().float().cpu().numpy()

                preds[:, ch] = inverse_target_preprocessing(out, ch, input_preprocessing)

        meas = df_chunk[columns_to_predict].to_numpy(np.float32, copy=False)
        resids = (meas - preds).astype(np.float32, copy=False)

        preds_df = pd.DataFrame(preds, index=df_chunk.index, columns=columns_to_predict).add_suffix(f"_pred_dnn")
        resids_df = pd.DataFrame(resids, index=df_chunk.index, columns=columns_to_predict).add_suffix(f"_resid_dnn")

        # drop old columns if present
        existing_cols = list(preds_df.columns) + list(resids_df.columns)
        existing = [c for c in existing_cols if c in df_chunk.columns]
        if existing:
            df_chunk = df_chunk.drop(columns=existing)

        frames = [df_chunk, preds_df, resids_df]
        df_chunk = pd.concat(frames, axis=1)
        df_chunk[f"adc_sum_pedsub{column_tag}_pred_dnn"] = df_chunk[[f"{x}_pred_dnn" for x in columns_to_predict]].sum(axis=1, skipna=True)
        df_chunk[f"adc_sum_pedsub{column_tag}_resid_dnn"] = df_chunk[[f"{x}_resid_dnn" for x in columns_to_predict]].sum(axis=1, skipna=True)

        outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{idx:03d}.parquet")
        utils.write_via_tmpdir(
            outfilename=outfilename,
            suffix=".parquet",
            writer_fn=lambda tmp, chunk=df_chunk: chunk.to_parquet(tmp, engine="pyarrow", index=True, compression="zstd"),
        )
        print(f"Wrote updated df with DNN predictions and residuals to {outfilename}, overwriting possibly existing columns in existing file.")


    print("Done.")


if __name__ == "__main__":
    main()
