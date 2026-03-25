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
        "-k",
        "--k",
        type=int,
        default=1,
        help="How many modes to subtract.",
    )
    parser.add_argument(
        "-c",
        "--column-tag",
        type=str,
        default="",
        help="Column tag to be appended at the end of 'adc_ch{i:03d}_pedsub'. E.g.: '_pred_analytic_k0' when APPLYING the noisemode subtraction",
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
    args = parser.parse_args()



    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            standardize_std = False,
            inputfoldertag = "",
        )
        for x in args.modules
    ]

    for cfg in cfgs:
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)
        add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer)


def add_projections_onto_noisemode(cfg, inferencer, column_tag, k: int) -> None:
    print("Hello from add_projections_onto_noisemode()!")    
    
    vecs = cfg.load_from_cov_folder(filename=f"eigenvectors_mm.parquet") # always load uncorrected eigenvectors, these are what we want to project onto
    vecs_kept = vecs[[f"eigvec_{k}"]]
    
    columns_to_project = [f"adc_ch{i:03d}_pedsub{column_tag}" for i in range(cfg.nch)] # columns to project, e.g. analytic k0 residuals

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):

        df_to_project = df_chunk[columns_to_project].astype("float32")

        projection = (df_to_project @ vecs_kept.rename(index=lambda s: f"{s}{column_tag}")).astype("float32") # shape (n_events, 1)
        projection.rename(columns={f"eigvec_{k}": f"proj_mode{k}_adc_chall_pedsub{column_tag}"}, inplace=True)

        # if rerunning on files that already had this, drop and recreate it.
        if f"proj_mode{k}_adc_chall_pedsub{column_tag}" in df_chunk.columns:
            df_chunk = df_chunk.drop(columns=f"proj_mode{k}_adc_chall_pedsub{column_tag}")

        # merge projection with main df
        df_chunk = pd.concat([df_chunk, projection], axis=1)

        # overwrite file
        outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{idx:03d}.parquet")
        df_chunk.to_parquet(outfilename, engine="pyarrow", index=True, compression="zstd")

        print(f"Wrote updated df with adc columns (tagged '{column_tag}') projected onto uncorrected noise mode {k} to {outfilename}, overwriting possibly existing file.")












if __name__ == "__main__":
    main()
    
