#! /eos/user/a/areimers/torch-env/bin/python

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal.*")
import uproot # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
import json
import argparse

from copy import deepcopy

import classes

### This script can only be used for ANALYSIS, not to prepare inputs for training a DNN.


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare analysis inputs from a beam run using pedestals from a pedestal run."
    )
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        default=112050,
        help="Beam run number to prepare analysis inputs for (e.g. 112050).",
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
        "--standardize-std",
        action="store_true",
        help="If set, also divide pedestal-subtracted columns by their std (unit variance).",
    )

    args = parser.parse_args()





    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            standardize_std=args.standardize_std,
            inputfoldertag="",
        ) 
        for x in args.modules
    ]
    for cfg in cfgs:
        convert_to_df(cfg=cfg)






def convert_to_df(cfg, nevt_per_batch: int = 100000, adcmax=None):
    print("Hello from convert_to_df()!")

    # make output folder
    os.makedirs(name=cfg.analysis_inputs_folder, exist_ok=True)

    print(f"Processing in batches of {nevt_per_batch}...")
    keepall = cfg.run not in cfg.runs_to_select_rings_for
    for batch_idx, df_chunk in enumerate(iter_analysis_df_chunks(cfg, nevt_per_batch, keepall=keepall, adcmax=adcmax)):
        
        # Save to file
        outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{batch_idx:03d}.parquet")
        df_chunk.to_parquet(outfilename, engine="pyarrow", index=True, compression="zstd")
        print(f"[INFO]: Wrote analysis df with {len(df_chunk)} events to {outfilename}")

    print(f"--> Wrote all analysis dfs {cfg.analysis_inputs_folder}")

def convert_to_df_synthetic(cfg, nevt_per_batch: int=100000, adcmax=None):
    os.makedirs(cfg.analysis_inputs_folder, exist_ok=True)

    source_runs = cfg.runs_per_synthetic_run[cfg.run]
    keepall = cfg.run not in cfg.runs_to_select_rings_for

    batch_idx = 0
    global_event_id = 0
    for r in source_runs:
        cfg_r = deepcopy(cfg)
        cfg_r.run = r
        
        for df_chunk in iter_analysis_df_chunks(cfg_r, nevt_per_batch, keepall=keepall, adcmax=adcmax):
            n = len(df_chunk)
            # continuous synthetic index
            df_chunk.index = np.arange(global_event_id, global_event_id + n, dtype=np.int64)
            global_event_id += n

            outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{batch_idx:03d}.parquet")
            df_chunk.to_parquet(outfilename, engine="pyarrow", index=True, compression="zstd")
            batch_idx += 1

    print(f"--> Wrote all analysis dfs {cfg.analysis_inputs_folder}")


# Preprocesses chunks of input data and converts to pd.df
def iter_analysis_df_chunks(cfg, nevt_per_batch=100000, keepall=True, adcmax=None):
    print(f"Preparing analysis inputs from Run{cfg.run} for module {cfg.modulename}...")
    infilename = os.path.join(cfg.get_histofiller_folder(), f"input_features_{cfg.modulename}.root")
    if not os.path.exists(infilename):
        raise ValueError(f"[ERROR]: Input file {infilename} does not exist, skipping module {cfg.modulename}.")

    scalar_means, scalar_stds, per_channel_means, per_channel_stds = load_means_stds_from_folder(cfg.pedestal_mean_std_folder)

    for df_chunk in uproot.iterate(f"{infilename}:InputFeatures", None, library="pd", step_size=int(nevt_per_batch)):
        zero_extra_cms(df=df_chunk, ncmchannels=cfg.ncmchannels)

        df_chunk = expand_per_channel_cols(df=df_chunk, colnames_to_expand=["adc", "adcm1", "chtypeadc", "erx", "ch_ucoord", "ch_vcoord"], colname_indices="chadc", nch=cfg.nch)
        df_chunk = expand_per_channel_cols(df=df_chunk, colnames_to_expand=["toa"], colname_indices="chtoa", nch=cfg.nch)
        df_chunk = expand_per_channel_cols(df=df_chunk, colnames_to_expand=["tot", "adc_tctp3"], colname_indices="chtot", nch=cfg.nch)
        df_chunk.drop(columns=["nerx", "chadc", "chtot", "chtoa", "cm2", "cm4", "cmall", "nchadc"], inplace=True)
        df_chunk = apply_mean_std(df=df_chunk, nch=cfg.nch, scalar_means=scalar_means, per_channel_means=per_channel_means, scalar_stds=scalar_stds, per_channel_stds=per_channel_stds, standardize_std=cfg.standardize_std)

        adc_cols = [f"adc_ch{i:03d}_pedsub" for i in range(cfg.nch)]
        df_chunk["adc_sum_allchannels_pedsub"] = df_chunk[adc_cols].sum(axis=1, skipna=True)

        if not keepall:
            rings_to_keep = cfg.channel_rings_to_keep_per_run[cfg.run]
            if rings_to_keep != "all":
                keep_mask_coords = select_rings_uv(df=df_chunk, nch=cfg.nch, rings_to_keep=rings_to_keep)
                mask_channels(df=df_chunk, keep_mask=keep_mask_coords)

        if adcmax:
            mask_channels_above_adc(df=df_chunk, nch=cfg.nch, adcmax=adcmax)
        
        df_chunk = apply_mean_std(df=df_chunk, nch=cfg.nch, scalar_means=scalar_means, per_channel_means=per_channel_means, scalar_stds=scalar_stds, per_channel_stds=per_channel_stds, standardize_std=cfg.standardize_std)

        yield df_chunk


def select_rings_uv(df, nch, rings_to_keep: list[int]) -> np.ndarray:
    ucols = [f"ch_ucoord_ch{i:03d}" for i in range(nch)]
    vcols = [f"ch_vcoord_ch{i:03d}" for i in range(nch)]
    u = df.loc[df.index[0], ucols].to_numpy()
    v = df.loc[df.index[0], vcols].to_numpy()
    w = u - v
    
    # "main" = valid coords (not the special -1 channels)
    mask_main = (u >= 0) & (v >= 0)
    special_mask = (u == -1) & (v == -1)

    # extrema over the main module
    umin, umax = u[mask_main].min(), u[mask_main].max()
    vmin, vmax = v[mask_main].min(), v[mask_main].max()
    wmin, wmax = w[mask_main].min(), w[mask_main].max()

    keep = np.zeros_like(u, dtype=bool)

    # special channels
    if -1 in rings_to_keep:
        keep |= special_mask

    # rings counted from outside: k means within [k] of any boundary (in u/v/w)
    for k in [r for r in rings_to_keep if r >= 0]:
        ring_k = mask_main & (
            (u == umin + k) | (u == umax - k) |
            (v == vmin + k) | (v == vmax - k) |
            (w == wmin + k) | (w == wmax - k)
        )
        keep |= ring_k

    return keep


def mask_channels(df: pd.DataFrame, keep_mask: np.ndarray) -> None:
    # keep_mask: True means keep, False means set NaN
    drop_idx = np.where(~keep_mask)[0]
    if len(drop_idx) == 0:
        return

    cols = [f"adc_ch{i:03d}" for i in drop_idx]
    cols = [c for c in cols if c in df.columns]
    if cols:
        df.loc[:, cols] = np.nan


def mask_channels_above_adc(df: pd.DataFrame, nch, adcmax) -> None:
    print(f"--> Masking channels with adc count above {adcmax}")
    adc_cols_to_apply_mask = [f"adc_ch{i:03d}" for i in range(nch)]
    adc_cols_pedsub = [f"{x}_pedsub" for x in adc_cols_to_apply_mask]
    
    adc = df.loc[:, adc_cols_pedsub]
    drop_mask = adc.gt(adcmax)  # shape (n_events, n_channels)
    raw_vals = df.loc[:, adc_cols_to_apply_mask]
    df.loc[:, adc_cols_to_apply_mask] = raw_vals.mask(drop_mask.to_numpy())



def zero_extra_cms(df, ncmchannels):
    for idx_erx in range(ncmchannels):
        df.loc[df["nerx"] <= idx_erx, f"cm_erx{idx_erx:02}"] = 0


def load_means_stds_from_folder(foldername: str):

    print(f"Loading means and stds from '{foldername}'")
    with open(f"{foldername}/means_scalar.json", "r") as f:
        scalar_means = json.load(f)
    with open(f"{foldername}/stds_scalar.json", "r") as f:
        scalar_stds = json.load(f)
    with open(f"{foldername}/means_vector.json", "r") as f:
        per_channel_means = json.load(f)
        per_channel_means = {k: np.array(v) for k, v in per_channel_means.items()}
    with open(f"{foldername}/stds_vector.json", "r") as f:
        per_channel_stds = json.load(f)
        per_channel_stds = {k: np.array(v) for k, v in per_channel_stds.items()}

    return scalar_means, scalar_stds, per_channel_means, per_channel_stds


def apply_mean_std(df, nch, scalar_means, per_channel_means, scalar_stds=None, per_channel_stds=None, standardize_std: bool = False):
    # 1) scalars: simple subtraction
    for col, mu in scalar_means.items():
        if col in df.columns:
            # print(f"[INFO]: Using means to center column {col}.")
            df[f"{col}_pedsub"] = (df[col] - mu).astype("float32")
            if standardize_std:
                std_val = scalar_stds[col]
                if std_val == 0.0:
                    std_val = 1.0
                df[f"{col}_pedsub_unitstd"] = (df[f"{col}_pedsub"] / std_val).astype("float32")
        else:
            print(f"[WARNING]: Expected scalar column {col} not found in DataFrame during centering.")

    # 2) vectors: subtract per channel using chadc as an index map
    new_cols_vector = []
    for col, mu in per_channel_means.items():

        mu_arr = np.asarray(mu)  # shape (nch,)
        if len(mu_arr) != nch:
            raise ValueError(f"Length mismatch for per-channel means of column '{col}': expected {nch}, got {len(mu_arr)}")
        cols_expanded = [f"{col}_ch{ch:03d}" for ch in range(nch)]

        if [c for c in cols_expanded if c not in df.columns]:
            print(f"[WARNING]: Expected vector column {col} not found in DataFrame during centering.")
            continue

        base = df[cols_expanded].to_numpy(dtype=np.float32)

        pedsub_vals = (base - mu_arr[np.newaxis, :]).astype("float32")
        pedsub_cols = [f"{x}_pedsub" for x in cols_expanded]
        new_cols_vector.append(pd.DataFrame(pedsub_vals, columns=pedsub_cols, index=df.index))

        if standardize_std:
            std_arr = np.asarray(per_channel_stds[col], dtype=np.float32)
            std_arr = np.where(std_arr > 1e-12, std_arr, 1.0)
            unit_vals = (pedsub_vals / std_arr[np.newaxis, :]).astype("float32")
            unit_cols = [f"{x}_pedsub_unitstd" for x in cols_expanded]
            new_cols_vector.append(pd.DataFrame(unit_vals, columns=unit_cols, index=df.index))
    if new_cols_vector:

        cols_to_add = []
        for d in new_cols_vector:
            cols_to_add.extend(d.columns)

        df = df.drop(columns=cols_to_add, errors="ignore")
        df = pd.concat([df] + new_cols_vector, axis=1)
    return df



def expand_per_channel_cols(df, colnames_to_expand, colname_indices, nch):
    n_events = len(df)
    idx_series = df[colname_indices].to_numpy()

    dfs_newcols = []
    for col in colnames_to_expand:
        full_arr = np.full((n_events, nch), np.nan, dtype=np.float32)
        col_series = df[col].to_numpy()

        for i, (idx, vals) in enumerate(zip(idx_series, col_series)):
            idx = np.asarray(idx, dtype=int)
            vals = np.asarray(vals, dtype=np.float32)

            if len(idx) != len(vals):
                raise ValueError(f"Length mismatch in row {i} for column '{col}': len(idx)={len(idx)}, len(vals)={len(vals)}")

            # at event row 'i' and channel columns 'idx' -> nonexistent values will not get overwritten, remain nan
            full_arr[i, idx] = vals

        # Create one col per channel, later fill only where there was an actual measurement in a given event
        ch_cols = [f"{col}_ch{ch:03d}" for ch in range(nch)]
        dfs_newcols.append(pd.DataFrame(full_arr, columns=ch_cols, index=df.index))
        df.drop(columns=[col], inplace=True)

    if dfs_newcols:
        result = pd.concat([df]+dfs_newcols, axis=1)
    else:
        result = df
    return result


if __name__ == '__main__':
  main()
