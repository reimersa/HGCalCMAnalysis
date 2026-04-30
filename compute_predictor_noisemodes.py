#! /eos/user/a/areimers/torch-env/bin/python

import argparse
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore

import inferencers
import classes
import utils

def main():

    parser = argparse.ArgumentParser(description="Compute analytic (k>0) predictor from pedestal covariance matrices.")
    parser.add_argument(
        "-p",
        "--pedestal-run",
        type=int,
        default=112044,
        help="Run number from which pedestals (means/stds) were computed (e.g. 112044).",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=1,
        help="How many modes to compute predictor for.",
    )
    parser.add_argument(
        "-c",
        "--column-tag",
        type=str,
        default="",
        help="Column tag to be appended at the end of 'adc_ch{i:03d}_pedsub'. E.g.: '_pred_analytic_k0'",
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
        help="List of module names to compute for.",
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
        compute_predictor_noisemodes(cfg=cfg, column_tag=args.column_tag, k=args.k)


def compute_predictor_noisemodes(cfg, column_tag: str, k: int) -> None:
    print("Hello from compute_predictor_noisemodes()!")
    if not cfg.derive_correction:
        raise ValueError(
            "Trying to compute a noise-mode subtraction predictor with cfg.derive_correction=False. "
            "Set derive_correction=True for configs that are meant to produce correction artifacts."
        )
    # normally, one would compute this for analytic (k=0) residuals (using the corresponding column_tag), but we can do whatever we want :)

    # load eigenvectors/-vals for this column tag
    vecs = cfg.load_from_cov_folder(filename=f"eigenvectors_mm{column_tag}.parquet")
    cols_to_keep = [f"eigvec_{idx}" for idx in range(k)]
    vecs_kept = vecs[cols_to_keep]

    # compute what to multiply with
    predictor = vecs_kept @ vecs_kept.T
    # print(predictor)

    # save this to "cov.predictors_folder" (look up exact name)
    os.makedirs(cfg.analytic_predictor_folder, exist_ok=True)
    predictor.to_parquet(os.path.join(cfg.analytic_predictor_folder, f"modesubtractor{column_tag}_k{k}.parquet"), index=True, compression="zstd")
    print(f"Wrote mode subtractor for {k} modes to folder: {cfg.analytic_predictor_folder}")







if __name__ == "__main__":
    main()
    
