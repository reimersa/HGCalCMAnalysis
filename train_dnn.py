#! /eos/user/a/areimers/torch-env/bin/python

import os
import time
import argparse
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
}


def main():
    p = argparse.ArgumentParser(description="Train PerChannelDNN on HGCal parquet DNN inputs (event x channel shuffle).")

    p.add_argument("-n", "--nodes", nargs="+", type=int, default=defaults["nodes_per_layer"])
    p.add_argument("-d", "--dropout", type=float, default=defaults["dropout_rate"])
    p.add_argument("-e", "--epochs", type=int, default=defaults["max_epochs"])
    p.add_argument("-t", "--tag", type=str, default=defaults["modeltag"])

    p.add_argument("-m", "--modules", nargs="+", metavar="MOD", default=defaults["modules_for_training"])
    p.add_argument("--run", type=int, default=112044)
    p.add_argument("--pedestal-run", type=int, default=112044)

    p.add_argument("--batch-samples", type=int, default=defaults["batch_samples"])
    p.add_argument(
        "--selection-for-correction",
        type=str,
        default="",
        help="Optional selection tag encoded in the correction-artifact folder.",
    )

    p.add_argument("--noprogbar", action="store_true")
    p.add_argument("--override-name", action="store_true")
    p.add_argument("--new-name", type=str, default="TESTTEST")

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
        train_dnn(cfg=cfg, noprogbar=args.noprogbar, per_channel_cols=args.per_channel_cols, nodes=args.nodes, dropout=args.dropout, tag=args.tag, override_name=args.override_name, new_name=args.new_name, batch_samples=args.batch_samples, epochs=args.epochs)




def train_dnn(cfg, noprogbar, per_channel_cols, nodes, dropout, tag, batch_samples, epochs, override_name=False, new_name="TESTTEST") -> None:
    show_progbar = not noprogbar

    # Device and split-specific data streams.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_inferencer = inferencers.AnalysisDNNInferencer(cfg=cfg, split="train", per_channel_cols=per_channel_cols)
    test_inferencer  = inferencers.AnalysisDNNInferencer(cfg=cfg, split="test", per_channel_cols=per_channel_cols)

    # Probe one batch to determine the model input shape.
    input_dim, feature_names = infer_input_dim_and_feature_names(
        train_inferencer=train_inferencer,
        batch_samples=batch_samples,
        per_channel_cols=per_channel_cols,
    )
    print(f"Detected input_dim = {input_dim}")
    
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

    # Optimizer and LR schedule.
    optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-3))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # Sample counts are only used for progress bars.
    if show_progbar:
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
    model = dnn_models.PerChannelDNN(
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
):
    model.train()
    sum_loss = 0.0
    sum_count = 0

    pbar = None
    if show_progbar:
        pbar = tqdm(total=n_total, desc="Training", unit="samples", leave=False, dynamic_ncols=True)

    for x_np, y_np in inferencer.sample_iter(batch_samples=batch_samples, include_targets=True, epoch_seed=epoch):
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)

        if not torch.isfinite(x).all():
            detail = summarize_nonfinite_tensor_2d(x, tensor_name="x", feature_names=feature_names)
            raise RuntimeError(f"Non-finite inputs detected before making predictions.\n{detail}")

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)

        if not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite prediction detected before loss.")

        loss = masked_mse(pred, y)
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
            pbar.set_postfix({"batch_mse": f"{loss.item():.4f}", "steps": sum_count})
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
):
    model.eval()
    sum_vloss = 0.0
    sum_vcount = 0

    pbar = None
    if show_progbar:
        pbar = tqdm(total=n_total, desc="Validation", unit="samples", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for x_np, y_np in inferencer.sample_iter(batch_samples=batch_samples, include_targets=True):
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            y = torch.from_numpy(y_np).to(device=device, dtype=torch.float32)

            pred = model(x)
            loss = masked_mse(pred, y)

            if loss.item() == 0.0 and not torch.any(torch.isfinite(pred) & torch.isfinite(y)):
                if pbar is not None:
                    pbar.update(int(x.shape[0]))
                continue

            sum_vloss += float(loss.item())
            sum_vcount += 1

            if pbar is not None:
                pbar.set_postfix({"batch_mse": f"{loss.item():.4f}", "steps": sum_vcount})
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



class GlobalCoherentNoiseLossFlatOrdered(nn.Module):
    """
    Coherent-noise loss computed over ALL channels (single number per batch),
    using your dir/alt RMS method, applied to residuals.

    Assumes flattened samples ordered as:
      event0 ch0..chC-1, event1 ch0..chC-1, ...
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, y_pred_flat: torch.Tensor, y_true_flat: torch.Tensor, n_ch: int):
        if y_pred_flat.ndim != 1 or y_true_flat.ndim != 1:
            raise RuntimeError("Expected 1D tensors: y_pred_flat, y_true_flat.")
        if y_pred_flat.shape[0] != y_true_flat.shape[0]:
            raise RuntimeError("y_pred_flat and y_true_flat must have same length.")
        if y_pred_flat.shape[0] % n_ch != 0:
            raise RuntimeError(f"Batch length N={y_pred_flat.shape[0]} not divisible by n_ch={n_ch}.")

        N = y_pred_flat.shape[0]
        n_ev = N // n_ch

        yp = y_pred_flat.reshape(n_ev, n_ch)
        yt = y_true_flat.reshape(n_ev, n_ch)

        m = torch.isfinite(yt).to(dtype=yp.dtype)  # [n_ev, n_ch]
        yt0 = torch.nan_to_num(yt, nan=0.0, posinf=0.0, neginf=0.0)

        r = (yp - yt0) * m                          # residuals [n_ev, n_ch]

        # If an event has <2 valid channels, drop it from the RMS (like your >=2 guard)
        nvalid = m.sum(dim=1)                       # [n_ev]
        m_ev = (nvalid >= 2).to(dtype=yp.dtype)     # [n_ev]

        # per-event sums
        d = r.sum(dim=1)                            # [n_ev]
        a = r[:, ::2].sum(dim=1) - r[:, 1::2].sum(dim=1)  # [n_ev]

        # RMS over events (masked)
        rms_d = torch.sqrt(((d * d) * m_ev).sum() / m_ev.sum().clamp_min(self.eps))
        rms_a = torch.sqrt(((a * a) * m_ev).sum() / m_ev.sum().clamp_min(self.eps))

        delta = rms_d * rms_d - rms_a * rms_a

        C = torch.tensor(float(n_ch), device=yp.device, dtype=yp.dtype)
        inc = rms_a / torch.sqrt(C)
        coh = torch.sign(delta) * torch.sqrt(delta.abs().clamp_min(self.eps)) / C

        mse = masked_mse(y_pred_flat, y_true_flat)

        # loss = coh * coh  # scalar
        loss = coh + 0.5*mse  # scalar

        # optional monitoring metric (coh vs inc power)
        f_coh = (coh * coh) / ((coh * coh) + (inc * inc) + self.eps)

        metrics = {
            "coh_loss": loss.detach(),
            "coh": coh.detach(),
            "inc": inc.detach(),
            "f_coh": f_coh.detach(),
            "mse": mse.detach(),
            "valid_event_frac": m_ev.mean().detach(),
        }
        return loss, metrics





if __name__ == "__main__":
    main()
