#!/usr/bin/env python3

import argparse
import os
import json
import re
from glob import glob

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
MODEL_MANIFEST_FILENAME = "model_manifest.json"


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
        help="Input ADC column tag appended after 'adc_ch{i:03d}_pedsub'. DNN output columns use the resolved model tag.",
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
    parser.add_argument(
        "--plot-inputs",
        action="store_true",
        help="Plot the full distribution of every feature actually passed to the DNN.",
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
        add_correction_dnn(
            cfg=cfg,
            inferencer=inferencer,
            nodes=args.nodes,
            dropout=args.dropout,
            tag=args.tag,
            column_tag=args.column_tag,
            per_channel_cols=args.per_channel_cols,
            infer_batch=args.infer_batch,
            plot_dir_loss=args.plotfolder,
            preprocess_inputs=args.preprocess_inputs,
            plot_inputs=args.plot_inputs,
            plot_dir_inputs=os.path.join(args.plotfolder, "dnn_inputs_apply"),
        )


def tag_with_input_preprocessing(tag: str, preprocess_inputs: bool) -> str:
    if not preprocess_inputs:
        return tag
    if INPUT_PREPROCESSING_TAG in tag.split("_"):
        return tag
    return f"{tag}_{INPUT_PREPROCESSING_TAG}" if tag else INPUT_PREPROCESSING_TAG


def dnn_output_tag_from_model_tag(tag: str) -> str:
    return "_dnn" if tag == "" else f"_dnn_{tag}"


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


class StreamingDNNInputPlotter:
    def __init__(self, feature_names: list[str], plot_dir: str, bins: int = 100):
        if bins <= 0:
            raise ValueError(f"Input histogram bin count must be positive, got {bins}.")
        self.feature_names = list(feature_names)
        self.plot_dir = plot_dir
        self.bins = bins
        n_features = len(self.feature_names)
        self.minimum = np.full(n_features, np.inf, dtype=np.float64)
        self.maximum = np.full(n_features, -np.inf, dtype=np.float64)
        self.count = np.zeros(n_features, dtype=np.int64)
        self.nonfinite = np.zeros(n_features, dtype=np.int64)
        self.sum = np.zeros(n_features, dtype=np.float64)
        self.sum2 = np.zeros(n_features, dtype=np.float64)
        self.edges = None
        self.histograms = None

    def _validate(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected DNN input matrix [N,{len(self.feature_names)}], got {values.shape}."
            )
        return values

    def observe_range(self, values: np.ndarray) -> None:
        values = self._validate(values)
        for feature_index in range(values.shape[1]):
            column = values[:, feature_index]
            finite_values = column[np.isfinite(column)]
            self.nonfinite[feature_index] += column.size - finite_values.size
            if finite_values.size == 0:
                continue
            self.minimum[feature_index] = min(self.minimum[feature_index], float(finite_values.min()))
            self.maximum[feature_index] = max(self.maximum[feature_index], float(finite_values.max()))
            self.count[feature_index] += finite_values.size
            self.sum[feature_index] += finite_values.sum(dtype=np.float64)
            self.sum2[feature_index] += np.square(finite_values).sum(dtype=np.float64)

    def finalize_ranges(self) -> None:
        self.edges = []
        self.histograms = []
        for feature_index in range(len(self.feature_names)):
            if self.count[feature_index] == 0:
                self.edges.append(None)
                self.histograms.append(None)
                continue
            minimum = self.minimum[feature_index]
            maximum = self.maximum[feature_index]
            if minimum == maximum:
                padding = max(0.5, abs(minimum) * 0.05)
                feature_edges = np.asarray([minimum - padding, maximum + padding], dtype=np.float64)
            else:
                feature_edges = np.linspace(minimum, maximum, self.bins + 1, dtype=np.float64)
            self.edges.append(feature_edges)
            self.histograms.append(np.zeros(len(feature_edges) - 1, dtype=np.int64))

    def fill(self, values: np.ndarray) -> None:
        if self.edges is None or self.histograms is None:
            raise RuntimeError("Call finalize_ranges() before filling DNN input histograms.")
        values = self._validate(values)
        for feature_index, feature_edges in enumerate(self.edges):
            if feature_edges is None:
                continue
            column = values[:, feature_index]
            finite_values = column[np.isfinite(column)]
            if finite_values.size:
                self.histograms[feature_index] += np.histogram(finite_values, bins=feature_edges)[0]

    def write(self) -> None:
        import matplotlib.pyplot as plt  # type: ignore

        if self.edges is None or self.histograms is None:
            raise RuntimeError("Cannot write DNN input plots before histogram filling.")
        os.makedirs(self.plot_dir, exist_ok=True)
        for feature_index, name in enumerate(self.feature_names):
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
            output_path = os.path.join(self.plot_dir, f"{feature_index:02d}_{safe_name}.pdf")
            fig, ax = plt.subplots(figsize=(7.0, 5.0))
            feature_edges = self.edges[feature_index]
            histogram = self.histograms[feature_index]
            if feature_edges is not None:
                ax.bar(
                    feature_edges[:-1],
                    histogram,
                    width=np.diff(feature_edges),
                    align="edge",
                    color="tab:blue",
                    alpha=0.8,
                )
                mean = self.sum[feature_index] / self.count[feature_index]
                variance = max(self.sum2[feature_index] / self.count[feature_index] - mean * mean, 0.0)
                summary = (
                    f"finite entries: {self.count[feature_index]:,}\n"
                    f"non-finite entries: {self.nonfinite[feature_index]:,}\n"
                    f"mean: {mean:.6g}\n"
                    f"std: {np.sqrt(variance):.6g}\n"
                    f"range: [{self.minimum[feature_index]:.6g}, {self.maximum[feature_index]:.6g}]"
                )
            else:
                summary = f"finite entries: 0\nnon-finite entries: {self.nonfinite[feature_index]:,}"
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
            ax.set_title(f"DNN input {feature_index}: {name}")
            ax.set_xlabel(name)
            ax.set_ylabel("Event-channel entries")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
            print(f"Wrote applied DNN input plot: {output_path}")


def iter_feature_matrices(df_inputs, per_event_cols, per_channel_cols, nch):
    x_evt = df_inputs[per_event_cols].to_numpy(np.float32, copy=False)
    allch_col_idx = inferencers.adc_allch_col_indices(per_event_cols, nch)
    ch_mats = inferencers.matrices_from_per_channel_cols(
        per_channel_cols=per_channel_cols,
        df=df_inputs,
        nch=nch,
    )
    for channel_index in range(nch):
        ch_feats = [ch_mats[column][:, channel_index][:, None] for column in per_channel_cols]
        x_evt_target = x_evt
        if allch_col_idx is not None:
            x_evt_target = x_evt.copy()
            x_evt_target[:, allch_col_idx[channel_index]] = np.float32(0.0)
        values = np.concatenate([x_evt_target] + ch_feats, axis=1).astype(np.float32, copy=False)
        yield channel_index, values


def find_model_manifest(cfg, nodes, dropout, tag):
    matches = []
    for path in sorted(glob(os.path.join(cfg.dnn_models_folder, "*", MODEL_MANIFEST_FILENAME))):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if [int(value) for value in payload.get("nodes_per_layer", [])] != [int(value) for value in nodes]:
            continue
        if float(payload.get("dropout_rate", -1.0)) != float(dropout):
            continue
        if payload.get("model_tag", "") != tag:
            continue
        matches.append((os.path.dirname(path), payload))
    if len(matches) > 1:
        raise RuntimeError(
            "More than one DNN model manifest matches the requested architecture/tag: "
            f"{[path for path, _ in matches]}"
        )
    return matches[0] if matches else (None, None)


def add_correction_dnn(
    cfg,
    inferencer,
    nodes: list[int],
    dropout: float,
    tag: str,
    column_tag: str,
    per_channel_cols: list[str],
    infer_batch: int,
    plot_dir_loss: str,
    preprocess_inputs: bool = False,
    plot_inputs: bool = False,
    plot_dir_inputs: str = None,
) -> None:
    print("Hello from add_correction_dnn()!")
    print(f"Loading checkpoint: {cfg.dnn_models_folder}")
    tag = tag_with_input_preprocessing(tag, preprocess_inputs)
    dnn_output_tag = dnn_output_tag_from_model_tag(tag)
    pred_suffix = f"_pred{dnn_output_tag}"
    resid_suffix = f"_resid{dnn_output_tag}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"DNN apply preprocess_inputs={preprocess_inputs}")
    print(f"DNN apply resolved tag={tag!r}, output tag={dnn_output_tag!r}")

    modeldir, model_manifest = find_model_manifest(
        cfg=cfg,
        nodes=nodes,
        dropout=dropout,
        tag=tag,
    )
    if model_manifest is None:
        feature_spec = prepare_dnn_inputs.make_feature_spec()
    else:
        feature_spec = prepare_dnn_inputs.make_feature_spec(
            feature_version=model_manifest["feature_version"],
            module_vocabulary=model_manifest.get("module_vocabulary", []),
        )
        if bool(model_manifest.get("preprocess_inputs", False)) != bool(preprocess_inputs):
            raise ValueError(
                "DNN preprocessing setting does not match the saved model manifest: "
                f"saved={model_manifest.get('preprocess_inputs')}, apply={preprocess_inputs}."
            )
        saved_per_channel_cols = list(model_manifest.get("per_channel_columns", []))
        if saved_per_channel_cols:
            if list(per_channel_cols) != saved_per_channel_cols:
                print(
                    "Using per-channel feature order from the DNN model manifest: "
                    f"{saved_per_channel_cols}"
                )
            per_channel_cols = saved_per_channel_cols
        if int(model_manifest.get("nch", cfg.nch)) != cfg.nch:
            raise ValueError(
                f"DNN model expects {model_manifest.get('nch')} channels, "
                f"but module {cfg.modulename!r} has {cfg.nch}."
            )
        if feature_spec["feature_version"] == prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS:
            module_index = prepare_dnn_inputs.module_onehot_index(cfg.modulename, feature_spec)
            if module_index == feature_spec.get("unknown_module_index"):
                print(
                    f"Module {cfg.modulename!r} was not used for training; activating reserved "
                    f"UNKNOWN one-hot index {module_index}."
                )

    columns_to_predict = [f"adc_ch{i:03d}_pedsub{column_tag}" for i in range(cfg.nch)]
    adc_channel_indices = [x for x in range(cfg.nch)]

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):
        print(f"Probing DNN inputs from chunk {idx:03d}...")
        df_inputs = prepare_dnn_inputs.make_input_df(
            cfg=cfg,
            df=df_chunk,
            adc_channel_indices=adc_channel_indices,
            column_tag=column_tag,
            feature_spec=feature_spec,
        )
        per_event_cols = [
            column
            for column in df_inputs.columns
            if column not in per_channel_cols
            and column not in prepare_dnn_inputs.INPUT_METADATA_COLUMNS
        ]
        feature_names = list(per_event_cols) + list(per_channel_cols)
        input_dim = len(feature_names)
        if model_manifest is not None and list(model_manifest.get("feature_names", [])) != feature_names:
            raise ValueError(
                "Applied DNN feature order does not match the saved model manifest.\n"
                f"Saved: {model_manifest.get('feature_names')}\nApply: {feature_names}"
            )
        break
    else:
        raise RuntimeError("Could not probe DNN inputs: no input chunks were available.")


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
    if modeldir is None:
        modeldir = os.path.join(cfg.dnn_models_folder, model.get_model_string())
    input_preprocessing = load_input_preprocessing(modeldir=modeldir, feature_names=feature_names) if preprocess_inputs else None

    state = torch.load(os.path.join(modeldir, "dnn_best.pth"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print(f"Now plotting loss")
    functions_plot.plot_loss(modeldir=modeldir, plot_dir=plot_dir_loss)
    print(f"Plotted loss")
    # return

    network_input_plotter = None
    raw_input_plotter = None
    if plot_inputs:
        if not plot_dir_inputs:
            raise ValueError("plot_dir_inputs is required when plot_inputs=True.")
        network_input_plotter = StreamingDNNInputPlotter(
            feature_names=feature_names,
            plot_dir=os.path.join(plot_dir_inputs, "network_inputs"),
        )
        if input_preprocessing is not None:
            raw_input_plotter = StreamingDNNInputPlotter(
                feature_names=feature_names,
                plot_dir=os.path.join(plot_dir_inputs, "before_preprocessing"),
            )

        print("Scanning applied DNN inputs to establish plotting ranges...")
        for df_chunk in inferencer.full_df_iter():
            df_inputs = prepare_dnn_inputs.make_input_df(
                cfg=cfg,
                df=df_chunk,
                adc_channel_indices=adc_channel_indices,
                column_tag=column_tag,
                feature_spec=feature_spec,
            )
            for _, raw_values in iter_feature_matrices(
                df_inputs=df_inputs,
                per_event_cols=per_event_cols,
                per_channel_cols=per_channel_cols,
                nch=C,
            ):
                if raw_input_plotter is not None:
                    raw_input_plotter.observe_range(raw_values)
                network_values = apply_input_preprocessing(raw_values, input_preprocessing)
                network_input_plotter.observe_range(network_values)
        network_input_plotter.finalize_ranges()
        if raw_input_plotter is not None:
            raw_input_plotter.finalize_ranges()

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):

        df_inputs = prepare_dnn_inputs.make_input_df(cfg=cfg, df=df_chunk, adc_channel_indices=adc_channel_indices, column_tag=column_tag, feature_spec=feature_spec)
        E = df_inputs.shape[0]

        # predictions [E, C]
        preds = np.full((E, C), np.nan, dtype=np.float32)

        # predict channel-by-channel (keeps memory bounded)
        with torch.no_grad():
            for ch, raw_values in iter_feature_matrices(
                df_inputs=df_inputs,
                per_event_cols=per_event_cols,
                per_channel_cols=per_channel_cols,
                nch=C,
            ):
                if raw_input_plotter is not None:
                    raw_input_plotter.fill(raw_values)
                X = apply_input_preprocessing(raw_values, input_preprocessing)
                if network_input_plotter is not None:
                    network_input_plotter.fill(X)

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

        preds_df = pd.DataFrame(preds, index=df_chunk.index, columns=columns_to_predict).add_suffix(pred_suffix)
        resids_df = pd.DataFrame(resids, index=df_chunk.index, columns=columns_to_predict).add_suffix(resid_suffix)

        # drop old columns if present
        existing_cols = list(preds_df.columns) + list(resids_df.columns)
        existing = [c for c in existing_cols if c in df_chunk.columns]
        if existing:
            df_chunk = df_chunk.drop(columns=existing)

        frames = [df_chunk, preds_df, resids_df]
        df_chunk = pd.concat(frames, axis=1)
        df_chunk[f"adc_sum_pedsub{column_tag}{pred_suffix}"] = df_chunk[[f"{x}{pred_suffix}" for x in columns_to_predict]].sum(axis=1, skipna=True)
        df_chunk[f"adc_sum_pedsub{column_tag}{resid_suffix}"] = df_chunk[[f"{x}{resid_suffix}" for x in columns_to_predict]].sum(axis=1, skipna=True)

        outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{idx:03d}.parquet")
        utils.write_via_tmpdir(
            outfilename=outfilename,
            suffix=".parquet",
            writer_fn=lambda tmp, chunk=df_chunk: chunk.to_parquet(tmp, engine="pyarrow", index=True, compression="zstd"),
        )
        print(f"Wrote updated df with DNN predictions and residuals to {outfilename}, overwriting possibly existing columns in existing file.")

    if network_input_plotter is not None:
        network_input_plotter.write()
    if raw_input_plotter is not None:
        raw_input_plotter.write()

    print("Done.")


if __name__ == "__main__":
    main()
