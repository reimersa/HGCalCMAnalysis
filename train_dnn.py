#!/usr/bin/env python3

import os
import time
import argparse
import json
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau  # type: ignore
from tqdm import tqdm  # type: ignore

import classes
import dnn_models
import inferencers
import utils


defaults = {
    "modules_for_training": ["ML_F3WC_IH0182"],
    "nodes_per_layer": [512, 512, 512, 512, 64],
    "dropout_rate": 0.0,
    "max_epochs": 1000,
    "modeltag": "",
    "batch_samples": 8192,   # number of (ev,ch) samples per optimizer step
    "max_steps_per_epoch": None,  # optionally cap steps/epoch for quick tests
    "shuffle_mode": "chunk_events",
    "shuffle_buffer_samples": 200_000,
    "shuffle_buffer_chunks": 4,
    "exclude_unconnected_targets": False,
    "weight_decay": 0.0,
    "sample_weighting": "none",
    "preprocess_inputs": False,
}

INPUT_PREPROCESSING_TAG = "inputzscore"
INPUT_PREPROCESSING_FILENAME = "input_preprocessing.json"


def parse_run_arg(value: str):
    return int(value) if str(value).isdigit() else value


def format_weight_decay_tag(weight_decay: float) -> str:
    value = f"{weight_decay:g}"
    if "e" in value:
        mantissa, exponent = value.split("e")
        exp = int(exponent)
        exp_tag = f"{'p' if exp >= 0 else 'm'}{abs(exp)}"
        return f"weightdecay{mantissa.replace('.', 'p')}e{exp_tag}"
    return f"weightdecay{value.replace('.', 'p')}"


def tag_with_weight_decay(tag: str, weight_decay: float) -> str:
    if weight_decay == 0.0:
        return tag
    if "weightdecay" in tag:
        return tag
    weight_decay_tag = format_weight_decay_tag(weight_decay)
    return f"{tag}_{weight_decay_tag}" if tag else weight_decay_tag


def tag_with_input_preprocessing(tag: str, preprocess_inputs: bool) -> str:
    if not preprocess_inputs:
        return tag
    if INPUT_PREPROCESSING_TAG in tag.split("_"):
        return tag
    return f"{tag}_{INPUT_PREPROCESSING_TAG}" if tag else INPUT_PREPROCESSING_TAG


def main():
    p = argparse.ArgumentParser(description="Train PerChannelDNN on HGCal parquet DNN inputs (event x channel shuffle).")

    p.add_argument("-n", "--nodes", nargs="+", type=int, default=defaults["nodes_per_layer"])
    p.add_argument("-d", "--dropout", type=float, default=defaults["dropout_rate"])
    p.add_argument("-e", "--epochs", type=int, default=defaults["max_epochs"])
    p.add_argument("-t", "--tag", type=str, default=defaults["modeltag"])
    p.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    p.add_argument(
        "--sample-weighting",
        choices=["none", "source_run_channel"],
        default=defaults["sample_weighting"],
        help="Optional per-sample DNN loss weighting.",
    )

    p.add_argument("-m", "--modules", nargs="+", metavar="MOD", default=defaults["modules_for_training"])
    p.add_argument("--run", type=parse_run_arg, default=112044)
    p.add_argument("--pedestal-run", type=int, default=112044)

    p.add_argument("--batch-samples", type=int, default=defaults["batch_samples"])
    p.add_argument(
        "--shuffle-mode",
        choices=["chunk_events", "buffered_chunk_events", "global_samples"],
        default=defaults["shuffle_mode"],
        help="DNN training sample shuffling: chunk-local events, buffered multi-chunk events, or streaming global (event,channel) samples.",
    )
    p.add_argument(
        "--shuffle-buffer-samples",
        type=int,
        default=defaults["shuffle_buffer_samples"],
        help="Maximum number of flattened (event,channel) samples to mix for --shuffle-mode global_samples.",
    )
    p.add_argument(
        "--shuffle-buffer-chunks",
        type=int,
        default=defaults["shuffle_buffer_chunks"],
        help="Number of parquet chunks to load and event-shuffle together for --shuffle-mode buffered_chunk_events.",
    )
    p.add_argument(
        "--exclude-unconnected-targets",
        action="store_true",
        help="Exclude unconnected channels from supervised DNN train/test targets.",
    )
    p.add_argument(
        "--selection-for-correction",
        type=str,
        default="",
        help="Optional selection tag encoded in the correction-artifact folder.",
    )

    p.add_argument("--noprogbar", action="store_true")
    p.add_argument("--override-name", action="store_true")
    p.add_argument("--new-name", type=str, default="TESTTEST")
    p.add_argument(
        "--preprocess-inputs",
        action="store_true",
        default=defaults["preprocess_inputs"],
        help="Z-score DNN input features and per-channel targets using train-split statistics.",
    )

    # tell inferencer which columns are per-channel list-columns
    p.add_argument(
        "--per-channel-cols",
        nargs="+",
        default=["channel_indices"],
        help="List of list-like per-channel columns in df_inputs. All other df_inputs columns are treated as per-event.",
    )
    args = p.parse_args()

    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            run_for_correction=args.run,
            module_for_correction=x,
            selection_for_correction=args.selection_for_correction,
            standardize_std = False,
            inputfoldertag = "",
        )
        for x in args.modules
    ]

    for cfg in cfgs:
        train_dnn(cfg=cfg, noprogbar=args.noprogbar, per_channel_cols=args.per_channel_cols, nodes=args.nodes, dropout=args.dropout, tag=args.tag, override_name=args.override_name, new_name=args.new_name, batch_samples=args.batch_samples, epochs=args.epochs, shuffle_mode=args.shuffle_mode, shuffle_buffer_samples=args.shuffle_buffer_samples, shuffle_buffer_chunks=args.shuffle_buffer_chunks, exclude_unconnected_targets=args.exclude_unconnected_targets, weight_decay=args.weight_decay, sample_weighting=args.sample_weighting, preprocess_inputs=args.preprocess_inputs)




def train_dnn(cfg, noprogbar, per_channel_cols, nodes, dropout, tag, batch_samples, epochs, override_name=False, new_name="TESTTEST", shuffle_mode: str = defaults["shuffle_mode"], shuffle_buffer_samples: int = defaults["shuffle_buffer_samples"], shuffle_buffer_chunks: int = defaults["shuffle_buffer_chunks"], exclude_unconnected_targets: bool = defaults["exclude_unconnected_targets"], weight_decay: float = defaults["weight_decay"], sample_weighting: str = defaults["sample_weighting"], preprocess_inputs: bool = defaults["preprocess_inputs"]) -> None:
    if sample_weighting not in ("none", "source_run_channel"):
        raise ValueError("sample_weighting must be 'none' or 'source_run_channel'.")
    show_progbar = not noprogbar
    tag = tag_with_input_preprocessing(tag, preprocess_inputs)
    tag = tag_with_weight_decay(tag, weight_decay)
    use_sample_weights = sample_weighting != "none"

    # Device and split-specific data streams.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"DNN training shuffle_mode={shuffle_mode}, shuffle_buffer_samples={shuffle_buffer_samples}, shuffle_buffer_chunks={shuffle_buffer_chunks}")
    print(f"DNN training weight_decay={weight_decay}, tag={tag}")
    print(f"DNN training sample_weighting={sample_weighting}")
    print(f"DNN training preprocess_inputs={preprocess_inputs}")
    exclude_target_channels = cfg.unconnected_channels if exclude_unconnected_targets else None
    if exclude_target_channels:
        print(f"Excluding {len(exclude_target_channels)} unconnected channel(s) from DNN train/test targets.")

    train_inferencer = inferencers.AnalysisDNNInferencer(cfg=cfg, split="train", per_channel_cols=per_channel_cols, require_weights=use_sample_weights)
    test_inferencer  = inferencers.AnalysisDNNInferencer(cfg=cfg, split="test", per_channel_cols=per_channel_cols, require_weights=use_sample_weights)

    # Probe one batch to determine the model input shape.
    input_dim, feature_names = infer_input_dim_and_feature_names(
        train_inferencer=train_inferencer,
        batch_samples=batch_samples,
        per_channel_cols=per_channel_cols,
    )
    print(f"Detected input_dim = {input_dim}")

    input_preprocessing = None
    if preprocess_inputs:
        input_preprocessing = compute_input_preprocessing_stats(
            train_inferencer=train_inferencer,
            batch_samples=batch_samples,
            feature_names=feature_names,
            shuffle_mode=shuffle_mode,
            shuffle_buffer_samples=shuffle_buffer_samples,
            shuffle_buffer_chunks=shuffle_buffer_chunks,
        )
    
    model = build_model(
        input_dim=input_dim,
        nodes=nodes,
        dropout=dropout,
        tag=tag,
        override_name=override_name,
        new_name=new_name,
    ).to(device)

    # Resolve output location from the final model name.
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.get_model_string()} ({n_params:,} trainable params)")

    modelfolder = os.path.join(cfg.dnn_models_folder, model.get_model_string())
    os.makedirs(modelfolder, exist_ok=False)
    print(f"Writing outputs to: {modelfolder}")
    if input_preprocessing is not None:
        save_input_preprocessing(modelfolder=modelfolder, input_preprocessing=input_preprocessing)

    # Optimizer and LR schedule.
    # optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(1e-3), weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # Sample counts are only used for progress bars.
    if show_progbar:
        if exclude_target_channels:
            nch_supervised = cfg.nch - len(set(exclude_target_channels))
            n_train = count_samples(train_inferencer, nch_per_event=nch_supervised)
            n_test = count_samples(test_inferencer, nch_per_event=nch_supervised)
        else:
            n_train = count_samples(train_inferencer, nch_per_event=cfg.nch)
            n_test = count_samples(test_inferencer, nch_per_event=cfg.nch)
        print(f"Total samples: train={n_train}, test={n_test}")

    train_losses: list[float] = []
    test_losses: list[float] = []
    best_test = float("inf")
    patience = 0
    early_stop_patience = 15

    for epoch in range(epochs):
        print(f"\n--- epoch {epoch+1}/{epochs} ---")
        t0 = time.time()

        # Training pass.
        train_loss = run_train_epoch(
            model=model,
            inferencer=train_inferencer,
            optimizer=optimizer,
            device=device,
            batch_samples=batch_samples,
            epoch=epoch,
            feature_names=feature_names,
            show_progbar=show_progbar,
            n_total=n_train if show_progbar else None,
            shuffle_mode=shuffle_mode,
            shuffle_buffer_samples=shuffle_buffer_samples,
            shuffle_buffer_chunks=shuffle_buffer_chunks,
            exclude_target_channels=exclude_target_channels,
            use_sample_weights=use_sample_weights,
            input_preprocessing=input_preprocessing,
        )
        train_losses.append(train_loss)

        # Validation pass.
        test_loss = run_eval_epoch(
            model=model,
            inferencer=test_inferencer,
            device=device,
            batch_samples=batch_samples,
            show_progbar=show_progbar,
            n_total=n_test if show_progbar else None,
            shuffle_mode=shuffle_mode,
            shuffle_buffer_samples=shuffle_buffer_samples,
            shuffle_buffer_chunks=shuffle_buffer_chunks,
            exclude_target_channels=exclude_target_channels,
            use_sample_weights=use_sample_weights,
            input_preprocessing=input_preprocessing,
        )
        test_losses.append(test_loss)

        # Update LR from validation loss.
        scheduler.step(test_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch+1}] lr = {lr:.2e}")

        # Save checkpoints and apply early stopping on validation loss.
        if test_loss < best_test:
            best_test = test_loss
            patience = 0
            save_training_state(
                modelfolder=modelfolder,
                model=model,
                train_losses=train_losses,
                test_losses=test_losses,
                is_best=True,
            )
            print(f"New best model saved (test={test_loss:.6f}).")
        else:
            patience += 1
            print(f"No improvement ({patience}/{early_stop_patience}).")
            if patience >= early_stop_patience:
                print("Early stopping.")
                break

        save_training_state(
            modelfolder=modelfolder,
            model=model,
            train_losses=train_losses,
            test_losses=test_losses,
            is_best=False,
        )

        dt = time.time() - t0
        print(f"Epoch {epoch+1} | train {train_loss:.4f} | test {test_loss:.4f} | time {dt:.1f}s")

    print(f"Training completed. Outputs in: {modelfolder}")





def build_model(input_dim: int, nodes, dropout, tag, override_name=False, new_name="TESTTEST"):
    model = dnn_models.build_per_channel_model(
        input_dim=input_dim,
        nodes_per_layer=nodes,
        dropout_rate=dropout,
        tag=tag,
    )
    if override_name:
        model.override_model_string(new_name)
    return model


def infer_input_dim_and_feature_names(train_inferencer, batch_samples, per_channel_cols):
    probe_x = None
    for x, y in train_inferencer.sample_iter(batch_samples=batch_samples, include_targets=True):
        probe_x = x
        break
    if probe_x is None:
        raise RuntimeError("Could not probe input_dim: no samples yielded from train split.")

    input_dim = int(probe_x.shape[1])
    feature_names = None
    per_event_cols = getattr(train_inferencer, "per_event_cols", None)
    if per_event_cols is not None:
        feature_names = list(per_event_cols) + list(per_channel_cols)
        if len(feature_names) != input_dim:
            print(
                f"[warning] Feature-name count mismatch: len(feature_names)={len(feature_names)} vs input_dim={input_dim}; "
                "using positional labels in non-finite diagnostics."
            )
            feature_names = None
    return input_dim, feature_names


def compute_input_preprocessing_stats(
    train_inferencer,
    batch_samples: int,
    feature_names,
    shuffle_mode: str,
    shuffle_buffer_samples: int,
    shuffle_buffer_chunks: int,
):
    if feature_names is None:
        raise RuntimeError("DNN input preprocessing requires stable feature names, but feature_names is None.")

    n_features = len(feature_names)
    count = np.zeros(n_features, dtype=np.int64)
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)
    n_target_channels = train_inferencer.cfg.nch
    target_count = np.zeros(n_target_channels, dtype=np.int64)
    target_sum = np.zeros(n_target_channels, dtype=np.float64)
    target_sum2 = np.zeros(n_target_channels, dtype=np.float64)

    print("Computing DNN input and target preprocessing statistics from train split...")
    for x_np, y_np, ch_np in train_inferencer.sample_iter(
        batch_samples=batch_samples,
        include_targets=True,
        epoch_seed=None,
        shuffle_mode=shuffle_mode,
        shuffle_buffer_samples=shuffle_buffer_samples,
        shuffle_buffer_chunks=shuffle_buffer_chunks,
        exclude_target_channels=None,
        include_channel_indices=True,
    ):
        if x_np.shape[1] != n_features:
            raise ValueError(f"Input preprocessing feature mismatch: got {x_np.shape[1]} columns, expected {n_features}.")
        finite = np.isfinite(x_np)
        x_clean = np.where(finite, x_np, 0.0).astype(np.float64, copy=False)
        count += finite.sum(axis=0, dtype=np.int64)
        sum_x += x_clean.sum(axis=0, dtype=np.float64)
        sum_x2 += (x_clean * x_clean).sum(axis=0, dtype=np.float64)

        finite_y = np.isfinite(y_np)
        if np.any(finite_y):
            ch_valid = ch_np[finite_y].astype(np.int64, copy=False)
            y_valid = y_np[finite_y].astype(np.float64, copy=False)
            target_count += np.bincount(ch_valid, minlength=n_target_channels)
            target_sum += np.bincount(ch_valid, weights=y_valid, minlength=n_target_channels)
            target_sum2 += np.bincount(ch_valid, weights=y_valid * y_valid, minlength=n_target_channels)

    if np.any(count == 0):
        missing = [feature_names[i] for i in np.flatnonzero(count == 0)[:10]]
        raise RuntimeError(f"Cannot preprocess DNN inputs: no finite values for feature(s): {missing}")
    if np.any(target_count == 0):
        missing = np.flatnonzero(target_count == 0)[:10].astype(int).tolist()
        raise RuntimeError(f"Cannot preprocess DNN targets: no finite values for target channel(s): {missing}")

    mean = sum_x / count
    var = (sum_x2 / count) - (mean * mean)
    var = np.maximum(var, 0.0)
    std_raw = np.sqrt(var)
    bad_std = (~np.isfinite(std_raw)) | (std_raw <= 0.0)
    std = std_raw.copy()
    std[bad_std] = 1.0

    target_mean = target_sum / target_count
    target_var = (target_sum2 / target_count) - (target_mean * target_mean)
    target_var = np.maximum(target_var, 0.0)
    target_std_raw = np.sqrt(target_var)
    target_bad_std = (~np.isfinite(target_std_raw)) | (target_std_raw <= 0.0)
    target_std = target_std_raw.copy()
    target_std[target_bad_std] = 1.0

    print(
        f"DNN input preprocessing: computed stats for {n_features} feature(s); "
        f"zero/non-finite std replaced for {int(np.count_nonzero(bad_std))} feature(s)."
    )
    print(
        f"DNN target preprocessing: computed stats for {n_target_channels} channel(s); "
        f"zero/non-finite std replaced for {int(np.count_nonzero(target_bad_std))} channel(s)."
    )

    return {
        "enabled": True,
        "method": "zscore",
        "feature_names": list(feature_names),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "zero_std_policy": "replace non-finite or non-positive std with 1.0",
        "targets_enabled": True,
        "target_method": "per_channel_zscore",
        "target_channels": list(range(n_target_channels)),
        "target_mean": target_mean.astype(float).tolist(),
        "target_std": target_std.astype(float).tolist(),
        "target_zero_std_policy": "replace non-finite or non-positive std with 1.0",
    }


def save_input_preprocessing(modelfolder: str, input_preprocessing) -> None:
    def _write_json(path: str, payload) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    utils.write_via_tmpdir(
        outfilename=os.path.join(modelfolder, INPUT_PREPROCESSING_FILENAME),
        suffix=".json",
        writer_fn=lambda tmp, payload=input_preprocessing: _write_json(tmp, payload),
    )
    print(f"Wrote DNN input preprocessing stats to {os.path.join(modelfolder, INPUT_PREPROCESSING_FILENAME)}")


def apply_input_preprocessing(x_np: np.ndarray, input_preprocessing) -> np.ndarray:
    if input_preprocessing is None:
        return x_np
    mean = np.asarray(input_preprocessing["mean"], dtype=np.float32)
    std = np.asarray(input_preprocessing["std"], dtype=np.float32)
    if x_np.shape[1] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError(
            f"Input preprocessing shape mismatch: x has {x_np.shape[1]} columns, "
            f"mean has {mean.shape[0]}, std has {std.shape[0]}."
        )
    return ((x_np - mean[None, :]) / std[None, :]).astype(np.float32, copy=False)


def apply_target_preprocessing(y_np: np.ndarray, ch_np: np.ndarray, input_preprocessing) -> np.ndarray:
    if input_preprocessing is None:
        return y_np
    if not input_preprocessing.get("targets_enabled", False):
        raise RuntimeError("Input preprocessing is enabled, but target preprocessing stats are missing.")
    target_mean = np.asarray(input_preprocessing["target_mean"], dtype=np.float32)
    target_std = np.asarray(input_preprocessing["target_std"], dtype=np.float32)
    ch_np = ch_np.astype(np.int64, copy=False)
    if np.any((ch_np < 0) | (ch_np >= target_mean.shape[0])):
        raise ValueError("Target preprocessing received channel index outside saved target stats range.")
    return ((y_np - target_mean[ch_np]) / target_std[ch_np]).astype(np.float32, copy=False)


def run_train_epoch(
    model,
    inferencer,
    optimizer,
    device,
    batch_samples,
    epoch,
    feature_names,
    show_progbar=False,
    n_total=None,
    shuffle_mode: str = defaults["shuffle_mode"],
    shuffle_buffer_samples: int = defaults["shuffle_buffer_samples"],
    shuffle_buffer_chunks: int = defaults["shuffle_buffer_chunks"],
    exclude_target_channels=None,
    use_sample_weights: bool = False,
    input_preprocessing=None,
):
    model.train()
    sum_loss = 0.0
    sum_count = 0

    pbar = None
    if show_progbar:
        pbar = tqdm(total=n_total, desc="Training", unit="samples", leave=False, dynamic_ncols=True)

    for batch in inferencer.sample_iter(
        batch_samples=batch_samples,
        include_targets=True,
        epoch_seed=epoch,
        shuffle_mode=shuffle_mode,
        shuffle_buffer_samples=shuffle_buffer_samples,
        shuffle_buffer_chunks=shuffle_buffer_chunks,
        exclude_target_channels=exclude_target_channels,
        include_weights=use_sample_weights,
        include_channel_indices=input_preprocessing is not None,
    ):
        if use_sample_weights and input_preprocessing is not None:
            x_np, y_np, w_np, ch_np = batch
        elif use_sample_weights:
            x_np, y_np, w_np = batch
            ch_np = None
        elif input_preprocessing is not None:
            x_np, y_np, ch_np = batch
            w_np = None
        else:
            x_np, y_np = batch
            w_np = None
            ch_np = None
        x_np = apply_input_preprocessing(x_np, input_preprocessing)
        if input_preprocessing is not None:
            y_np = apply_target_preprocessing(y_np, ch_np, input_preprocessing)
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)
        w = None if w_np is None else torch.from_numpy(w_np).to(device=device, dtype=torch.float32)
        if use_sample_weights and sum_count == 0:
            print_sample_weight_stats(w=w, y=y, split="train")

        if not torch.isfinite(x).all():
            detail = summarize_nonfinite_tensor_2d(x, tensor_name="x", feature_names=feature_names)
            raise RuntimeError(f"Non-finite inputs detected before making predictions.\n{detail}")

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        if not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite prediction detected before loss.")
        loss = masked_weighted_mse(pred, y, w) if use_sample_weights else masked_mse(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected.")

        if loss.item() == 0.0 and not torch.any(torch.isfinite(pred) & torch.isfinite(y)):
            if pbar is not None:
                pbar.update(int(x.shape[0]))
            continue

        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(f"Non-finite grad in {n}")

        optimizer.step()
        for n, p in model.named_parameters():
            if not torch.isfinite(p).all():
                print(f"[after step] non-finite parameter in {n}")
                print("sample values:", p.detach().flatten()[:20].cpu())
                raise RuntimeError(f"Non-finite parameter after optimizer.step(): {n}")

        sum_loss += float(loss.item())
        sum_count += 1

        if pbar is not None:
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "steps": sum_count})
            pbar.update(int(x.shape[0]))

    if pbar is not None:
        pbar.close()

    return sum_loss / max(1, sum_count)


def run_eval_epoch(
    model,
    inferencer,
    device,
    batch_samples,
    show_progbar=False,
    n_total=None,
    shuffle_mode: str = defaults["shuffle_mode"],
    shuffle_buffer_samples: int = defaults["shuffle_buffer_samples"],
    shuffle_buffer_chunks: int = defaults["shuffle_buffer_chunks"],
    exclude_target_channels=None,
    use_sample_weights: bool = False,
    input_preprocessing=None,
):
    model.eval()
    sum_vloss = 0.0
    sum_vcount = 0

    pbar = None
    if show_progbar:
        pbar = tqdm(total=n_total, desc="Validation", unit="samples", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for batch in inferencer.sample_iter(batch_samples=batch_samples, include_targets=True, shuffle_mode=shuffle_mode, shuffle_buffer_samples=shuffle_buffer_samples, shuffle_buffer_chunks=shuffle_buffer_chunks, exclude_target_channels=exclude_target_channels, include_weights=use_sample_weights, include_channel_indices=input_preprocessing is not None):
            if use_sample_weights and input_preprocessing is not None:
                x_np, y_np, w_np, ch_np = batch
            elif use_sample_weights:
                x_np, y_np, w_np = batch
                ch_np = None
            elif input_preprocessing is not None:
                x_np, y_np, ch_np = batch
                w_np = None
            else:
                x_np, y_np = batch
                w_np = None
                ch_np = None
            x_np = apply_input_preprocessing(x_np, input_preprocessing)
            if input_preprocessing is not None:
                y_np = apply_target_preprocessing(y_np, ch_np, input_preprocessing)
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            y = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)
            w = None if w_np is None else torch.from_numpy(w_np).to(device=device, dtype=torch.float32)
            if use_sample_weights and sum_vcount == 0:
                print_sample_weight_stats(w=w, y=y, split="test")

            pred = model(x)
            loss = masked_weighted_mse(pred, y, w) if use_sample_weights else masked_mse(pred, y)

            if loss.item() == 0.0 and not torch.any(torch.isfinite(pred) & torch.isfinite(y)):
                if pbar is not None:
                    pbar.update(int(x.shape[0]))
                continue

            sum_vloss += float(loss.item())
            sum_vcount += 1

            if pbar is not None:
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "steps": sum_vcount})
                pbar.update(int(x.shape[0]))

    if pbar is not None:
        pbar.close()

    return sum_vloss / max(1, sum_vcount)


def save_training_state(modelfolder, model, train_losses, test_losses, is_best=False):
    state_dict = model.state_dict()
    train_arr = np.asarray(train_losses, dtype=np.float64)
    test_arr = np.asarray(test_losses, dtype=np.float64)

    if is_best:
        utils.write_via_tmpdir(
            outfilename=os.path.join(modelfolder, "dnn_best.pth"),
            suffix=".pth",
            writer_fn=lambda tmp, state=state_dict: torch.save(state, tmp),
        )
    utils.write_via_tmpdir(
        outfilename=os.path.join(modelfolder, "dnn_last.pth"),
        suffix=".pth",
        writer_fn=lambda tmp, state=state_dict: torch.save(state, tmp),
    )
    utils.write_via_tmpdir(
        outfilename=os.path.join(modelfolder, "train_losses.npy"),
        suffix=".npy",
        writer_fn=lambda tmp, arr=train_arr: np.save(tmp, arr),
    )
    utils.write_via_tmpdir(
        outfilename=os.path.join(modelfolder, "test_losses.npy"),
        suffix=".npy",
        writer_fn=lambda tmp, arr=test_arr: np.save(tmp, arr),
    )


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred/target: [N]
    Mask out NaNs in either pred or target. (NaN inputs typically produce NaN preds.)
    Returns mean over valid entries. If none valid, returns 0.
    """
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if not torch.any(valid):
        # no valid samples in this batch
        return pred.new_tensor(0.0)
    diff = pred[valid] - target[valid]
    return (diff * diff).mean()


def masked_weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(pred) & torch.isfinite(target) & torch.isfinite(weight) & (weight > 0.0)
    if not torch.any(valid):
        return pred.new_tensor(0.0)
    diff = pred[valid] - target[valid]
    w = weight[valid]
    return torch.sum(w * diff * diff) / torch.sum(w)


def print_sample_weight_stats(w: torch.Tensor, y: torch.Tensor, split: str) -> None:
    if w is None:
        return
    valid_target = torch.isfinite(y)
    positive_weight = torch.isfinite(w) & (w > 0.0)
    n_valid = int(valid_target.sum().item())
    n_positive = int((valid_target & positive_weight).sum().item())
    if n_positive == 0:
        print(f"DNN sample weights ({split}): valid_targets={n_valid}, positive_weights=0")
        return
    w_valid = w[valid_target & positive_weight]
    print(
        f"DNN sample weights ({split}, first batch): "
        f"valid_targets={n_valid}, positive_weights={n_positive}, "
        f"mean={float(w_valid.mean().item()):.6g}, min={float(w_valid.min().item()):.6g}, max={float(w_valid.max().item()):.6g}"
    )
def summarize_nonfinite_tensor_2d(
    tensor: torch.Tensor,
    tensor_name: str = "tensor",
    feature_names = None,
    max_cols: int = 12,
    max_rows: int = 5,
    max_bad_values_per_row: int = 8,
) -> str:
    if tensor.ndim != 2:
        return f"{tensor_name}: expected 2D tensor, got shape {tuple(tensor.shape)}."

    t = tensor.detach().to(device="cpu")
    bad = ~torch.isfinite(t)
    total_bad = int(bad.sum().item())
    if total_bad == 0:
        return f"{tensor_name}: all finite."

    bad_rows = torch.nonzero(bad.any(dim=1), as_tuple=False).flatten()
    bad_per_col = bad.sum(dim=0).to(dtype=torch.int64)

    def col_label(col_idx: int) -> str:
        if feature_names is None:
            return f"col_{col_idx}"
        if 0 <= col_idx < len(feature_names):
            return str(feature_names[col_idx])
        return f"col_{col_idx}"

    lines = [
        f"{tensor_name}: shape={tuple(t.shape)}, non_finite={total_bad}, rows_with_non_finite={int(bad_rows.numel())}, cols_with_non_finite={int((bad_per_col > 0).sum().item())}"
    ]

    k = min(max_cols, t.shape[1])
    if k > 0:
        top_counts, top_idx = torch.topk(bad_per_col, k=k, largest=True, sorted=True)
        top_items = []
        for c_count, c_idx in zip(top_counts.tolist(), top_idx.tolist()):
            if c_count <= 0:
                continue
            top_items.append(f"{col_label(int(c_idx))}:{int(c_count)}")
        if top_items:
            lines.append("Top bad features (count): " + ", ".join(top_items))

    n_show_rows = min(max_rows, int(bad_rows.numel()))
    for i in range(n_show_rows):
        r = int(bad_rows[i].item())
        row_bad_cols = torch.nonzero(bad[r], as_tuple=False).flatten()
        shown_cols = row_bad_cols[:max_bad_values_per_row].tolist()
        row_parts = []
        for c in shown_cols:
            v = float(t[r, c].item())
            if np.isnan(v):
                vtxt = "nan"
            elif np.isposinf(v):
                vtxt = "+inf"
            elif np.isneginf(v):
                vtxt = "-inf"
            else:
                vtxt = f"{v:.6g}"
            row_parts.append(f"{col_label(int(c))}={vtxt}")
        n_extra = int(row_bad_cols.numel()) - len(shown_cols)
        suffix = "" if n_extra <= 0 else f", ... (+{n_extra} more)"
        lines.append(f"row {r}: " + ", ".join(row_parts) + suffix)

    return "\n".join(lines)


def count_samples(inf: "inferencers.AnalysisDNNInferencer", nch_per_event: int) -> int:
    """
    Count total (ev,ch) samples in the selected split by summing lengths of filtered input dfs.
    This is used only for progress bars / reporting.
    """
    total = 0
    for df_shuf in inf.full_inputs_iter():
        total += len(df_shuf)
    return int(total)*nch_per_event


def cov_like_from_residuals_torch(r: torch.Tensor, m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Torch analogue of your CovAccumulator for i=j=residuals, per batch.

    r: [B, C] masked residuals
    m: [B, C] float mask (0/1)
    returns: [C, C] matrix C = (r^T r) / (m^T m)
    """
    S = r.transpose(0, 1) @ r
    N = m.transpose(0, 1) @ m
    return S / N.clamp_min(eps)

def f_corr_from_cov_torch(cov: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Same as your f_corr_from_cov but safe for total_sum2==0:
      f_corr = offdiag_sum2 / total_sum2
    so perfect residual=0 -> loss ~ 0.
    """
    cov2 = cov * cov
    total_sum2 = cov2.sum()
    diag_sum2 = torch.diagonal(cov2, 0).sum()
    offdiag_sum2 = total_sum2 - diag_sum2
    return offdiag_sum2 / (total_sum2 + eps)

class MSEPlusOffDiagFracLossFlatOrdered(nn.Module):
    """
    Assumes flattened samples ordered as:
      event0 ch0..chC-1, event1 ch0..chC-1, ...

    loss = masked_mse + lam * f_corr(cov(residuals))
    """
    def __init__(self, lam: float = 0.01, eps: float = 1e-12):
        super().__init__()
        self.lam = float(lam)
        self.eps = float(eps)

    def forward(self, y_pred_flat: torch.Tensor, y_true_flat: torch.Tensor, n_ch: int):
        if y_pred_flat.ndim != 1 or y_true_flat.ndim != 1:
            raise RuntimeError("Expected 1D tensors: y_pred_flat, y_true_flat.")
        if y_pred_flat.shape[0] != y_true_flat.shape[0]:
            raise RuntimeError("y_pred_flat and y_true_flat must have same length.")
        if y_pred_flat.shape[0] % n_ch != 0:
            raise RuntimeError(f"Batch length N={y_pred_flat.shape[0]} not divisible by n_ch={n_ch}.")

        # --- masked MSE on the flat vectors (simple & correct) ---
        mse = masked_mse(y_pred_flat, y_true_flat)

        # --- f_corr term needs [m_ev, n_ch] residual matrix ---
        N = y_pred_flat.shape[0]
        m_ev = N // n_ch
        yp = y_pred_flat.reshape(m_ev, n_ch)
        yt = y_true_flat.reshape(m_ev, n_ch)

        m = torch.isfinite(yt).to(dtype=yp.dtype)
        yt0 = torch.nan_to_num(yt, nan=0.0, posinf=0.0, neginf=0.0)
        r = (yp - yt0) * m

        cov = cov_like_from_residuals_torch(r, m, eps=self.eps)
        f_corr = f_corr_from_cov_torch(cov, eps=self.eps)

        loss = mse*0.1 + self.lam * f_corr

        metrics = {
            "loss": loss.detach(),
            "mse": mse.detach(),
            "f_corr": f_corr.detach(),
            "valid_frac": m.mean().detach(),
        }
        return loss, metrics



if __name__ == "__main__":
    main()
