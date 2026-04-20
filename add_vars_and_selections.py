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
        help="Module whose correction context this output should correspond to.",
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
        add_vars_and_selections(cfg=cfg, inferencer=inferencer)


def add_vars_and_selections(cfg, inferencer) -> None:
    print("Hello from add_vars_and_selections()!")

    selections_per_selstring = {
        "selection_full": 
            "",
        "selection_notot":
            "nchtot == 0",
        "selection_notot_notoa":
            "nchtot == 0 and nchtoa == 0",
        "selection_notot_notoa_roc0":
            "nchtot == 0 and nchtoa == 0 and erx00_hastot == 0 and erx01_hastot == 0 and erx00_hastoa == 0 and erx01_hastoa == 0",
        "selection_notot_notoa_roc2":
            "nchtot == 0 and nchtoa == 0 and erx04_hastot == 0 and erx05_hastot == 0 and erx04_hastoa == 0 and erx05_hastoa == 0",
        "selection_notot_roc0":
            "erx00_hastot == 0 and erx01_hastot == 0",
        "selection_notot_roc2":
            "erx04_hastot == 0 and erx05_hastot == 0",
        "selection_notot_notoa_nosaturatedadc":
            "nchtot == 0 and nchtoa == 0 and adc_max_pedsub < 900",
        "selection_notot_notoa_nosaturatedadc_adcsumlt50":
            "nchtot == 0 and nchtoa == 0 and adc_max_pedsub < 900 and adc_sum_pedsub < 50",
        "selection_notot_notoa_nosaturatedadc_adcsumgt50":
            "nchtot == 0 and nchtoa == 0 and adc_max_pedsub < 900 and adc_sum_pedsub > 50",
        "selection_notot_notoa_nosaturatedadc_adcsumltm170":
            "nchtot == 0 and nchtoa == 0 and adc_max_pedsub < 900 and adc_sum_pedsub < -170",
        "selection_notot_notoa_nosaturatedadc_adcsumgtm170":
            "nchtot == 0 and nchtoa == 0 and adc_max_pedsub < 900 and adc_sum_pedsub > -170",
        "selection_withtot":
            "nchtot > 0",
        "selection_toa0to10":
            "nchtoa >= 0 and nchtoa < 10",
        "selection_toa10to20":
            "nchtoa >= 10 and nchtoa < 20",
        "selection_toa20to30":
            "nchtoa >= 20 and nchtoa < 30",
        "selection_trigtime":
            "trig_time > 107 and trig_time < 113 ",
        "selection_train":
            "split == 'train'",
        "selection_test":
            "split == 'test'",
        "adcsumm500to500":
            "adc_sum_pedsub > -500 and adc_sum_pedsub < 500",
        "adcsum_residdnn_200to700":
            "adc_sum_pedsub_resid_dnn > 200 and adc_sum_pedsub_resid_dnn < 700",
        "adcsum_residdnn_lt200":
            "adc_sum_pedsub_resid_dnn < 200",
    }


    adc_cols = [f"adc_ch{i:03d}_pedsub" for i in range(cfg.nch)]
    adc_cols_beforeped = [c.replace("_pedsub", "") for c in adc_cols]
    ped_cols = [c.replace("adc_", "ped_") for c in adc_cols_beforeped]
    adcm1_cols = [f"adcm1_ch{i:03d}" for i in range(cfg.nch)]
    for idx, df_chunk in enumerate(inferencer.full_df_iter()):

        df_chunk["adc_sum_pedsub"] = df_chunk[adc_cols].sum(axis=1, skipna=True)
        for erx in range(cfg.nerx):
            df_chunk[f"erx{erx:02d}_hastot"] = df_chunk[[f"tot_ch{chidx:03d}" for chidx in range(cfg.nch_per_erx*erx, cfg.nch_per_erx*(erx+1))]].notna().any(axis=1).astype(int)
        for erx in range(cfg.nerx):
            df_chunk[f"erx{erx:02d}_hastoa"] = df_chunk[[f"toa_ch{chidx:03d}" for chidx in range(cfg.nch_per_erx*erx, cfg.nch_per_erx*(erx+1))]].notna().any(axis=1).astype(int)
        df_chunk["adc_max_pedsub"] = df_chunk[adc_cols].max(axis=1, skipna=True)
        df_chunk["adcm1_max"] = df_chunk[adcm1_cols].max(axis=1, skipna=True)
        df_chunk["event_id"] = df_chunk.index

        peds = df_chunk[adc_cols_beforeped].to_numpy() - df_chunk[adc_cols].to_numpy()
        df_chunk.loc[:, ped_cols] = peds

        split_file = os.path.join(cfg.dnn_training_input_folder, "event_split_train_test.parquet")
        if os.path.isfile(split_file):
            df_split = pd.read_parquet(split_file)
            split_map = df_split.set_index("event_id_global")["split"]
            df_chunk["split"] = df_chunk["event_id"].map(split_map)
        else:
            print(f"[WARNING] Did not define selections into train and test events because the split file ({split_file}) does not exist. Was the split maybe not defined yet? Remember to rerun this function to add those columns.")
            if "selection_train" in selections_per_selstring:
                del selections_per_selstring["selection_train"]
            if "selection_test" in selections_per_selstring:
                del selections_per_selstring["selection_test"]


        for sel_col, expr in selections_per_selstring.items():
            if not expr:
                df_chunk[sel_col] = True
            else:
                try:
                    df_chunk[sel_col] = df_chunk.eval(expr, engine="python")
                except Exception as e:
                    print(f"[WARNING] Skipping selection '{sel_col}' because required columns are not available yet: {e}")

        # overwrite file
        outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{idx:03d}.parquet")
        df_chunk.to_parquet(outfilename, engine="pyarrow", index=True, compression="zstd")

        print(f"Wrote updated df with additional variables and selection decisions to {outfilename}, overwriting existing file.")












if __name__ == "__main__":
    main()
    
