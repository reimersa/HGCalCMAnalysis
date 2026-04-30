#! /eos/user/a/areimers/torch-env/bin/python
import os
import numpy as np # type: ignore
import uproot # type: ignore
import math
import json
import argparse

from tqdm import tqdm # type: ignore

import classes
import utils



def main():

    parser = argparse.ArgumentParser(description="Compute global means/stds for HGCal  modules.")
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        default=112044,
        # default=110398,
        help="Run number (e.g. 112044).",
    )
    parser.add_argument(
        "-m",
        "--modules",
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
        help="List of module names to process.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the means and stds (default is False to reduce verbosity).",
    )
    args = parser.parse_args()





    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            run=args.run,
            run_for_pedestal=args.run, # this determines where the means will be written, so must be the same as the run we're reading in
            run_for_correction=args.run,
            module_for_correction=x,
            standardize_std = False,
            inputfoldertag = "",
        ) 
        for x in args.modules
    ]
    for cfg in cfgs:
        calculate_means_stds(cfg=cfg, print_vals=args.print)










# def calculate_means_stds(run: int, modulename: list[str], print_vals: bool=False) -> None:
def calculate_means_stds(cfg, print_vals: bool=False) -> None:

    # columns for which the mean and std are needed (for others we just don't need these)
    scalar_cols = [f"cm_erx{idxerx:02}" for idxerx in range(cfg.ncmchannels)] + ["nchadc", "nerx"]
    vector_cols = ["erx", "chadc", "adc"]
    vector_cols_to_center_per_channel = ["adc"]

    os.makedirs(name=cfg.pedestal_mean_std_folder, exist_ok=True)

    # Open file and load tree
    print(f"Computing means and stds for module {cfg.modulename}...")
    infilename = os.path.join(cfg.histofiller_folder, f"input_features_{cfg.modulename}.root")

    print("Computing global means for inputs...")
    scalar_means, per_channel_means, scalar_stds, per_channel_stds = compute_global_means(infilename=infilename, treename="InputFeatures", scalar_cols=scalar_cols, vector_cols=vector_cols, nch_expected=cfg.nch, ncmchannels=cfg.ncmchannels, vector_cols_to_center_per_channel=vector_cols_to_center_per_channel, nevt_per_batch=100000)
    if print_vals:
        for k, v in scalar_means.items():
            print(f"  scalar means {k}: {v:.4f}")
        for k, v in per_channel_means.items():
            print(f"   per-channel means {k}: {v.shape}\n{v}")
        for k, v in scalar_stds.items():
            print(f"  scalar stds{k}: {v:.4f}")
        for k, v in per_channel_stds.items():
            print(f"  per-channel stds {k}: {v.shape}\n{v}")

    def write_json(payload, filename: str) -> None:
        utils.write_via_tmpdir(
            outfilename=os.path.join(cfg.pedestal_mean_std_folder, filename),
            suffix=".json",
            writer_fn=lambda tmp, data=payload: _dump_json(tmp, data),
        )

    write_json(scalar_means, "means_scalar.json")
    write_json(scalar_stds, "stds_scalar.json")
    write_json({k: v.tolist() for k, v in per_channel_means.items()}, "means_vector.json")
    write_json({k: v.tolist() for k, v in per_channel_stds.items()}, "stds_vector.json")
    print(f"Means and stds saved in folder: {cfg.pedestal_mean_std_folder}")


def _dump_json(path: str, payload) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle)








def compute_global_means(infilename, treename, scalar_cols, vector_cols, nch_expected, ncmchannels, vector_cols_to_center_per_channel, nevt_per_batch=100000):
    """
    Streaming pass over the tree, filtering bad events on the fly,
    to compute:
      - global means for scalar numeric input columns (cm_erxXX etc),
      - per-channel means (length = nch_expected),

    Returns: (scalar_means_dict, per_channel_means_dict)
    """
    # We only need: cm_erxXX, nerx (for zeroing), chadc/adc/nchadc
    needed = scalar_cols + vector_cols
    if "erx" not in needed:
        needed.append("erx")

    # Accumulators
    # For scalar columns
    sums = {c: 0.0 for c in scalar_cols}
    sumsqs = {c: 0.0 for c in scalar_cols}
    counts = 0  # event count after filtering

    # For per-channel ADC (length = nch_expected)
    sums_arrays = {c: np.zeros(nch_expected, dtype=np.float64) for c in vector_cols}
    sumsqs_arrays = {c: np.zeros(nch_expected, dtype=np.float64) for c in vector_cols}
    counts_arrays = {c: np.zeros(nch_expected, dtype=np.int64) for c in vector_cols}

    with uproot.open(infilename) as f:
        total_entries = f[treename].num_entries
    print("Computing global means...")


    # Stream in modest chunks
    for df_chunk in tqdm(
        uproot.iterate(
            f"{infilename}:{treename}",
            needed,
            library="pd",
            step_size=nevt_per_batch,
        ),
        desc=f"{os.path.basename(infilename)}",
        unit="chunk",
        total=None if total_entries is None else math.ceil(total_entries / nevt_per_batch),  # rough estimate
    ):

        # 2) apply the same zeroing rule as everywhere to avoid computing mean/std for CMs that do not exist
        for idx_erx in range(ncmchannels):
            df_chunk.loc[df_chunk["nerx"] <= idx_erx, f"cm_erx{idx_erx:02}"] = 0

        for c in sums.keys():
            vals = df_chunk[c].to_numpy(dtype=np.float64, copy=False)
            sums[c] += np.sum(vals)
            sumsqs[c] += np.sum(vals ** 2)
        counts += len(df_chunk)

        # 3b) accumulate per-channel means using chadc mapping
        # Each row has arrays; slice and add to the right bins
        ch_lists = df_chunk["chadc"].to_numpy()
        # Flatten chadc for the whole chunk in one go
        ch_all = np.concatenate(ch_lists).astype(np.int64, copy=False)

        for vector_col in sums_arrays.keys():
            # print(f"Processing vector column: {vector_col}")
            v_lists = df_chunk[vector_col].to_numpy()
            v_all   = np.concatenate(v_lists).astype(np.float64, copy=False)

            # Add all contributions in one shot:
            np.add.at(sums_arrays[vector_col],   ch_all, v_all)
            np.add.at(sumsqs_arrays[vector_col],  ch_all, v_all ** 2)
            np.add.at(counts_arrays[vector_col], ch_all, 1)

    scalar_means = {c: (sums[c] / counts) for c in sums.keys()}
    scalar_stds  = {}
    for c in sums.keys():
        var = (sumsqs[c] / counts) - scalar_means[c] ** 2
        scalar_stds[c] = np.sqrt(max(var, 0.0))

    # Per-channel means
    per_channel_means = {}
    per_channel_stds = {}
    for vector_col in sums_arrays.keys():
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.divide(sums_arrays[vector_col], counts_arrays[vector_col],where=(counts_arrays[vector_col] > 0))
            var = np.divide(sumsqs_arrays[vector_col], counts_arrays[vector_col],where=(counts_arrays[vector_col] > 0)) - mean ** 2
            std = np.sqrt(np.maximum(var, 0.0))
            # per_channel_means[vector_col] = np.divide(sums_arrays[vector_col], counts_arrays[vector_col], where=(counts_arrays[vector_col] > 0))

            # come columns shouldn't be centered per channel, but globally across all channels (e.g., 'erx')
            if vector_col not in vector_cols_to_center_per_channel:
                mean[:] = mean.mean()
                std[:]  = std.mean()
                # per_channel_means[vector_col][:] = per_channel_means[vector_col].mean()
        mean[counts_arrays[vector_col] == 0] = 0.0
        std[counts_arrays[vector_col] == 0]  = 1.0
        per_channel_means[vector_col] = mean
        per_channel_stds[vector_col]  = std

    return scalar_means, per_channel_means, scalar_stds, per_channel_stds



if __name__ == "__main__":
    main()
    
