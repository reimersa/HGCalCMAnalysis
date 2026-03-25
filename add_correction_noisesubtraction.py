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
        add_correction_noisesubtraction(cfg=cfg, inferencer=inferencer)


def add_correction_noisesubtraction(cfg, inferencer, column_tag, k: int) -> None:
    print("Hello from add_correction_noisesubtraction()!")

    predictor = pd.read_parquet(os.path.join(cfg.analytic_predictor_folder, f"modesubtractor{column_tag}_k{k}.parquet"))
    cols_to_correct_on_top_of = [f"adc_ch{i:03d}_pedsub{column_tag}" for i in range(cfg.nch)] # columns to correct on top of (i.e., subtract noise modes that are left over in these columns)
    # cols_input_is_made_from = [x.replace("_resid", "_pred") for x in cols_to_correct_on_top_of]
    
    # decide whether a baseline prediction exists
    has_baseline_pred = "_resid" in column_tag
    cols_input_is_made_from = [c.replace("_resid", "_pred") for c in cols_to_correct_on_top_of] if has_baseline_pred else None


    for idx, df_chunk in enumerate(inferencer.full_df_iter()):

        # make predictions and residuals
        # input = df_chunk[cols_to_correct_on_top_of]
        input = df_chunk[cols_to_correct_on_top_of].fillna(0.0).astype("float32")
        rows_with_nan = df_chunk[cols_to_correct_on_top_of].isna().any(axis=1)
        if rows_with_nan.any():
            print(f"[INFO] {rows_with_nan.sum()} / {len(rows_with_nan)} events had NaNs -> treated as 0 in projection")

        preds_wrt_input = (input @ predictor).astype("float32")

        if cols_input_is_made_from is None:
            preds_wrt_raw = preds_wrt_input
        else:
            # base = df_chunk[cols_input_is_made_from].astype("float32")
            base = df_chunk[cols_input_is_made_from].fillna(0.0).astype("float32")
            assert(preds_wrt_input.rename(columns=lambda c: c.replace("_resid", "_pred")).columns.equals(base.columns))
            preds_wrt_raw = base + preds_wrt_input.rename(columns=lambda c: c.replace("_resid", "_pred"))

        # print("raw:")
        # print(df_chunk[[f"adc_ch{i:03d}_pedsub" for i in range(cfg.nch)]])
        # print(f"k={k} predictions of this:")
        # print(preds_wrt_raw)

        preds_wrt_raw = preds_wrt_raw.add_suffix(f"_pred_submodes{k}")
        resids_of_input = (input - preds_wrt_input).astype("float32") # these are the same as if computing the pred wrt raw and then subtracting the total pred from raw
        resids_of_input = resids_of_input.add_suffix(f"_resid_submodes{k}")

        # if rerunning on files that already had these columns, drop and recreate them.
        existing = [c for c in list(resids_of_input.columns) if c in df_chunk.columns] + [c for c in list(preds_wrt_raw.columns) if c in df_chunk.columns]
        if existing:
            df_chunk = df_chunk.drop(columns=existing)

        # merge preds and resids_of_input with main df
        df_chunk = pd.concat([df_chunk, preds_wrt_raw, resids_of_input], axis=1)

        # overwrite file
        outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{idx:03d}.parquet")
        df_chunk.to_parquet(outfilename, engine="pyarrow", index=True, compression="zstd")

        print(f"Wrote updated df with noise mode (k={k}) subtraction predictions and residuals to {outfilename}, overwriting existing file.")












if __name__ == "__main__":
    main()
    
