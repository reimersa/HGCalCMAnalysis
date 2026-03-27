#! /eos/user/a/areimers/torch-env/bin/python

import argparse
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore

import inferencers
import classes
import utils

def main():

    parser = argparse.ArgumentParser(description="Compute all possible variants of covariance matrices.")
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        default=112044,
        # default=110398,
        help="Run number to compute covariances for (e.g. 112044).",
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
        "--module-for-correction",
        type=str,
        required=True,
        help="Module from which the correction artifacts should be loaded.",
    )
    args = parser.parse_args()



    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            run_for_correction=args.run,
            module_for_correction=args.module_for_correction,
            standardize_std = False,
            inputfoldertag = "",
        )
        for x in args.modules
    ]

    for cfg in cfgs:
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)
        add_correction_analytic(cfg=cfg, inferencer=inferencer)


def add_correction_analytic(cfg, inferencer) -> None:
    print("Hello from add_correction_analytic()!")

    # Load W
    print(f"Loading analytic predictor W matrix from {cfg.analytic_predictor_folder}")
    W = pd.read_parquet(os.path.join(cfg.analytic_predictor_folder, "analytic_k0.parquet"))
    W_cmunc = pd.read_parquet(os.path.join(cfg.analytic_predictor_folder, "analytic_with_unconnected_k0.parquet"))
    cm_cols = [f"cm_erx{i:02d}_pedsub" for i in range(cfg.nerx)]
    unc_cols = [f"adc_ch{(x*cfg.nch_per_erx + off):03d}_pedsub" for x in range(cfg.nerx) for off in [8, 17, 19, 28]]

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):

        # make predictions and residuals
        cms = df_chunk[cm_cols]
        preds = (cms @ W.T).astype("float32")
        resids = (df_chunk[preds.columns] - preds).astype("float32")
        preds = preds.add_suffix("_pred_analytic_k0")
        resids = resids.add_suffix("_resid_analytic_k0")

        # for the version with CMs and unconnected channels
        cms_unc = df_chunk[cm_cols+unc_cols]
        preds_cmunc = (cms_unc @ W_cmunc.T).astype("float32")
        resids_cmunc = (df_chunk[preds_cmunc.columns] - preds_cmunc).astype("float32")
        preds_cmunc = preds_cmunc.add_suffix("_pred_analytic_with_unconnected_k0")
        resids_cmunc = resids_cmunc.add_suffix("_resid_analytic_with_unconnected_k0")

        # if rerunning on files that already had these columns, drop and recreate them.
        existing = [c for c in list(preds.columns) + list(resids.columns) + list(preds_cmunc.columns) + list(resids_cmunc.columns) if c in df_chunk.columns]
        if existing:
            df_chunk = df_chunk.drop(columns=existing)

        # merge preds and resids with main df
        df_chunk = pd.concat([df_chunk, preds, resids, preds_cmunc, resids_cmunc], axis=1)

        # overwrite file
        outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{idx:03d}.parquet")
        df_chunk.to_parquet(outfilename, engine="pyarrow", index=True, compression="zstd")

        print(f"Wrote updated df with analytic predictions and residuals, including a version using unconnected channes as well, to {outfilename}, overwriting existing file.")












if __name__ == "__main__":
    main()
    
