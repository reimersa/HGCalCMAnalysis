#!/usr/bin/env python3

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
    parser.add_argument(
        "--selection-for-correction",
        type=str,
        default="",
        help="Optional selection tag encoded in the correction-artifact folder.",
    )
    args = parser.parse_args()



    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            run_for_correction=args.run,
            module_for_correction=args.module_for_correction,
            selection_for_correction=args.selection_for_correction,
            standardize_std = False,
            inputfoldertag = "",
        )
        for x in args.modules
    ]

    for cfg in cfgs:
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)
        add_vars_and_selections(cfg=cfg, inferencer=inferencer)


def add_vars_and_selections(cfg, inferencer, split_selections_only: bool = False) -> None:
    print("Hello from add_vars_and_selections()!")

    selections_per_selstring = {
        "selection_full": 
            "",
        "selection_trigtime":
            "trig_time > 107 and trig_time < 113 ",
        "selection_test":
            "split == 'test'",
        "selection_test_trigtime":
            "split == 'test' and trig_time > 107 and trig_time < 113",
    }


    adc_cols = [f"adc_ch{i:03d}_pedsub" for i in range(cfg.nch)]
    adc_cols_beforeped = [c.replace("_pedsub", "") for c in adc_cols]
    ped_cols = [c.replace("adc_", "ped_") for c in adc_cols_beforeped]
    adcm1_cols = [f"adcm1_ch{i:03d}" for i in range(cfg.nch)]
    for idx, df_chunk in enumerate(inferencer.full_df_iter()):
        if split_selections_only:
            split_file = os.path.join(cfg.dnn_training_input_folder, "event_split_train_test.parquet")
            if not os.path.isfile(split_file):
                raise FileNotFoundError(
                    f"Cannot update split selections because {split_file} does not exist. "
                    "Run prepare_dnn_inputs first."
                )
            if "selection_trigtime" not in df_chunk.columns:
                raise KeyError(
                    "Column 'selection_trigtime' is required for split_selections_only=True. "
                    "Run add_vars_and_selections with split_selections_only=False first."
                )

            df_split = pd.read_parquet(split_file)
            split_map = df_split.set_index("event_id_global")["split"]
            event_ids = df_chunk["event_id"] if "event_id" in df_chunk.columns else df_chunk.index
            df_chunk["split"] = event_ids.map(split_map)
            df_chunk["selection_test"] = df_chunk["split"] == "test"
            df_chunk["selection_test_trigtime"] = df_chunk["selection_test"] & df_chunk["selection_trigtime"].astype(bool)

            outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{idx:03d}.parquet")
            utils.write_via_tmpdir(
                outfilename=outfilename,
                suffix=".parquet",
                writer_fn=lambda tmp, chunk=df_chunk: chunk.to_parquet(tmp, engine="pyarrow", index=True, compression="zstd"),
            )

            print(f"Wrote updated df with split selections to {outfilename}, overwriting existing file.")
            continue

        df_chunk["adc_sum_pedsub"] = df_chunk[adc_cols].sum(axis=1, skipna=True)
        if "nchadcgt10" not in df_chunk.columns:
            df_chunk["nchadcgt10"] = (df_chunk[adc_cols] > 10).sum(axis=1)
        if "nchadcgt50" not in df_chunk.columns:
            df_chunk["nchadcgt50"] = (df_chunk[adc_cols] > 50).sum(axis=1)
        if "nchadcgt200" not in df_chunk.columns:
            df_chunk["nchadcgt200"] = (df_chunk[adc_cols] > 200).sum(axis=1)
        if "nchadcgt500" not in df_chunk.columns:
            df_chunk["nchadcgt500"] = (df_chunk[adc_cols] > 500).sum(axis=1)
        for erx in range(cfg.nerx):
            df_chunk[f"erx{erx:02d}_hastot"] = df_chunk[[f"tot_ch{chidx:03d}" for chidx in range(cfg.nch_per_erx*erx, cfg.nch_per_erx*(erx+1))]].notna().any(axis=1).astype(int)
        for erx in range(cfg.nerx):
            df_chunk[f"erx{erx:02d}_hastoa"] = df_chunk[[f"toa_ch{chidx:03d}" for chidx in range(cfg.nch_per_erx*erx, cfg.nch_per_erx*(erx+1))]].notna().any(axis=1).astype(int)
        for erx in range(cfg.nerx):
            ch_start = cfg.nch_per_erx * erx
            ch_stop = cfg.nch_per_erx * (erx + 1)
            adc_cols_erx = [f"adc_ch{chidx:03d}_pedsub" for chidx in range(ch_start, ch_stop)]
            toa_cols_erx = [f"toa_ch{chidx:03d}" for chidx in range(ch_start, ch_stop)]
            tot_cols_erx = [f"tot_ch{chidx:03d}" for chidx in range(ch_start, ch_stop)]

            if f"erx{erx:02d}_nchtoa" not in df_chunk.columns:
                df_chunk[f"erx{erx:02d}_nchtoa"] = df_chunk[toa_cols_erx].notna().sum(axis=1).astype(int)
            if f"erx{erx:02d}_nchtot" not in df_chunk.columns:
                df_chunk[f"erx{erx:02d}_nchtot"] = df_chunk[tot_cols_erx].notna().sum(axis=1).astype(int)
            if f"erx{erx:02d}_nchadcgt10" not in df_chunk.columns:
                df_chunk[f"erx{erx:02d}_nchadcgt10"] = (df_chunk[adc_cols_erx] > 10).sum(axis=1).astype(int)
            if f"erx{erx:02d}_nchadcgt50" not in df_chunk.columns:
                df_chunk[f"erx{erx:02d}_nchadcgt50"] = (df_chunk[adc_cols_erx] > 50).sum(axis=1).astype(int)
            if f"erx{erx:02d}_nchadcgt200" not in df_chunk.columns:
                df_chunk[f"erx{erx:02d}_nchadcgt200"] = (df_chunk[adc_cols_erx] > 200).sum(axis=1).astype(int)
            if f"erx{erx:02d}_nchadcgt500" not in df_chunk.columns:
                df_chunk[f"erx{erx:02d}_nchadcgt500"] = (df_chunk[adc_cols_erx] > 500).sum(axis=1).astype(int)
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
            if "selection_test" in selections_per_selstring:
                del selections_per_selstring["selection_test"]
            if "selection_test_trigtime" in selections_per_selstring:
                del selections_per_selstring["selection_test_trigtime"]

        for sel_col, expr in selections_per_selstring.items():
            if sel_col == "selection_trigtime":
                if "source_is_pedestal" not in df_chunk.columns:
                    raise KeyError(
                        "Column 'source_is_pedestal' is required to evaluate "
                        "'selection_trigtime'. Rerun convert_to_df to regenerate "
                        "the parquet inputs with per-event source provenance."
                    )

                trig_mask = pd.Series(False, index=df_chunk.index)
                if "trig_time" in df_chunk.columns:
                    trig_mask = (df_chunk["trig_time"] > 107) & (df_chunk["trig_time"] < 113)

                df_chunk[sel_col] = df_chunk["source_is_pedestal"].astype(bool) | trig_mask
                continue
            if not expr:
                df_chunk[sel_col] = True
            else:
                try:
                    df_chunk[sel_col] = df_chunk.eval(expr, engine="python")
                except Exception as e:
                    print(f"[WARNING] Skipping selection '{sel_col}' because required columns are not available yet: {e}")

        # overwrite file
        outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{idx:03d}.parquet")
        utils.write_via_tmpdir(
            outfilename=outfilename,
            suffix=".parquet",
            writer_fn=lambda tmp, chunk=df_chunk: chunk.to_parquet(tmp, engine="pyarrow", index=True, compression="zstd"),
        )

        print(f"Wrote updated df with additional variables and selection decisions to {outfilename}, overwriting existing file.")












if __name__ == "__main__":
    main()
    
