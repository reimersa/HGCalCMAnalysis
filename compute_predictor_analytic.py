#! /eos/user/a/areimers/torch-env/bin/python

import argparse
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore

import inferencers
import classes
import utils

def main():

    parser = argparse.ArgumentParser(description="Compute analytic (k=0) predictor from pedestal covariance matrices.")
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
        help="List of module names to compute covariances for.",
    )
    parser.add_argument(
        "--selection-for-correction",
        type=str,
        default="",
        help="Optional selection tag encoded in the correction-artifact folder.",
    )
    args = parser.parse_args()



    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            run=args.pedestal_run,
            run_for_pedestal=args.pedestal_run,
            run_for_correction=args.pedestal_run,
            module_for_correction=x,
            derive_correction=True,
            selection_for_correction=args.selection_for_correction,
            standardize_std = False,
            inputfoldertag = "",
        )
        for x in args.modules
    ]

    for cfg in cfgs:
        compute_predictor_analytic(cfg=cfg)


def compute_predictor_analytic(cfg) -> None:
    print("Hello from compute_predictor_analytic()!")
    if not cfg.derive_correction:
        raise ValueError(
            "Trying to compute an analytic predictor with cfg.derive_correction=False. "
            "Set derive_correction=True for configs that are meant to produce correction artifacts."
        )

    # Load covs
    sigma_cc_df = cfg.load_from_cov_folder(filename="sigma_cc.parquet")
    sigma_mc_df = cfg.load_from_cov_folder(filename="sigma_mc.parquet")

    # Keep only rows/cols of CM channels up to the number of erx's we have in this module
    keep_labels = [f"cm_erx{i:02d}_pedsub" for i in range(cfg.nerx)]
    sigma_cc_df = sigma_cc_df.loc[keep_labels, keep_labels]
    sigma_mc_df = sigma_mc_df[keep_labels]

    # compute W
    W = pd.DataFrame(sigma_mc_df.values @ np.linalg.inv(sigma_cc_df.values), index=sigma_mc_df.index, columns=sigma_cc_df.index)

    def write_df(df: pd.DataFrame, filename: str) -> None:
        utils.write_via_tmpdir(
            outfilename=os.path.join(cfg.analytic_predictor_folder, filename),
            suffix=".parquet",
            writer_fn=lambda tmp, data=df: data.to_parquet(tmp, index=True, compression="zstd"),
        )

    # Write to file
    os.makedirs(cfg.analytic_predictor_folder, exist_ok=True)
    write_df(W, "analytic_k0.parquet")
    print(f"Wrote analytic (k=0) predictor matrix W to folder: {cfg.analytic_predictor_folder}")


    # now the version that also decorrelates from the unconnected channels

    # Load covs
    sigma_cucu_df = cfg.load_from_cov_folder(filename="sigma_cucu.parquet")
    sigma_mnou_cu_df = cfg.load_from_cov_folder(filename="sigma_mnou_cu.parquet")

    # Keep only rows/cols of CM channels up to the number of erx's we have in this module
    unc_cols = [f"adc_ch{(x*cfg.nch_per_erx + off):03d}_pedsub" for x in range(cfg.nerx) for off in [8, 17, 19, 28]]
    keep_labels = [f"cm_erx{i:02d}_pedsub" for i in range(cfg.nerx)] + unc_cols
    sigma_cucu_df = sigma_cucu_df.loc[keep_labels, keep_labels]
    sigma_mnou_cu_df = sigma_mnou_cu_df[keep_labels]

    # compute W
    W_cmunc = pd.DataFrame(sigma_mnou_cu_df.values @ np.linalg.inv(sigma_cucu_df.values), index=sigma_mnou_cu_df.index, columns=sigma_cucu_df.index)
    # output-padded matrix: 222 targets (rows), inputs = keep_labels (cols)
    adc_cols = [f"adc_ch{idx:03d}_pedsub" for idx in range(cfg.nch)]
    W_outfull = pd.DataFrame(0.0, index=adc_cols, columns=W_cmunc.columns, dtype="float32")
    W_outfull.loc[W_cmunc.index, :] = W_cmunc.astype("float32").values

    # Write to file
    os.makedirs(cfg.analytic_predictor_folder, exist_ok=True)
    write_df(W_outfull, "analytic_with_unconnected_k0.parquet")
    print(f"Wrote analytic (k=0) predictor matrix W using CMs and unconnected channels to folder: {cfg.analytic_predictor_folder}")







if __name__ == "__main__":
    main()
    
