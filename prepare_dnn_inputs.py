#! /eos/user/a/areimers/torch-env/bin/python

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal.*")
import uproot # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
import json
import argparse

import classes
import inferencers
import utils

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare analysis inputs from a beam run using pedestals from a pedestal run."
    )
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        default=112050,
        help="Beam run number to prepare DNN inputs for (e.g. 112050).",
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
        "--module-for-correction",
        type=str,
        required=True,
        help="Module from which the correction artifacts should be loaded.",
    )
    parser.add_argument(
        "-s",
        "--selection",
        type=str,
        default="full",
        metavar="SEL",
        help="Only use events for which the column 'selection_{SEL}' is true. These need to be constructed and added to the df before, of course.",
    )
    parser.add_argument(
        "-c",
        "--column-tag",
        type=str,
        default="",
        help="Column tag to be appended at the end of 'adc_ch{i:03d}_pedsub'. E.g.: '_pred_analytic_k0' when APPLYING the noisemode subtraction",
    )

    args = parser.parse_args()





    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            run_for_correction=args.run,
            module_for_correction=args.module_for_correction,
            derive_correction=True,
            selection_for_correction=args.selection,
            standardize_std=False,
            inputfoldertag="",
        ) 
        for x in args.modules
    ]
    for cfg in cfgs:
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=args.selection)
        prepare_dnn_inputs(cfg=cfg, column_tag=args.column_tag, inferencer=inferencer)



def _load_cell_area_fractions(cfg, adc_channel_indices):
    module_type = cfg.modulename[:4].replace("-", "_")
    cellareas_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "cellareas.json")
    )

    with open(cellareas_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if module_type not in payload:
        raise KeyError(
            f"Module type '{module_type}' not found in cell area file {cellareas_path}."
        )

    sfs = np.asarray(payload[module_type]["SF"], dtype=np.float32)
    if sfs.shape[0] != cfg.nch:
        raise ValueError(
            f"Cell area count mismatch for module type '{module_type}': "
            f"expected {cfg.nch}, found {sfs.shape[0]}."
        )

    return sfs[np.asarray(adc_channel_indices, dtype=np.int64)]


def make_input_df(cfg, df, adc_channel_indices, column_tag):

    cm_columns = [f"cm_erx{idx:02}_pedsub" for idx in range(cfg.ncmchannels)]
    df_inputs = df[cm_columns].copy().astype("float32")
# 
    adc_channel_indices_mean = sum(adc_channel_indices) / len(adc_channel_indices)

    erx_indices = [i // cfg.nch_per_erx for i in adc_channel_indices]
    erx_indices_mean = sum(erx_indices) / len(erx_indices)

    # channel indices as list, one value per channel
    df_inputs["channel_indices"] = [[x - adc_channel_indices_mean for x in adc_channel_indices]] * len(df_inputs)

    # ERX indices as list, one value per channel
    df_inputs["erx_indices"] = [[x - erx_indices_mean for x in erx_indices]] * len(df_inputs)

    # Relative cell area, using the same SF normalization as the summary plots.
    cell_area_fractions = _load_cell_area_fractions(cfg=cfg, adc_channel_indices=adc_channel_indices)
    df_inputs["cell_area_fraction"] = [cell_area_fractions.tolist()] * len(df_inputs)

    # unconnected channels on the same e-Rx as the channel
    def _build_unconnected_feature(offset: int, out_col: str) -> None:
        src_cols = [f"adc_ch{(x * cfg.nch_per_erx + offset):03d}_pedsub{column_tag}" for x in erx_indices]
        arr = df[src_cols].to_numpy(dtype=np.float32, copy=True)
        bad = ~np.isfinite(arr)
        n_bad = int(np.count_nonzero(bad))
        if n_bad > 0:
            n_rows_bad = int(np.count_nonzero(np.any(bad, axis=1)))
            print(
                f"[warning] Replacing {n_bad} non-finite value(s) in '{out_col}' "
                f"(rows affected: {n_rows_bad}) with 0."
            )
            arr[bad] = np.float32(0.)
        df_inputs[out_col] = arr.tolist()

    _build_unconnected_feature(offset=8, out_col="adc_unconnected_00")
    _build_unconnected_feature(offset=17, out_col="adc_unconnected_01")
    _build_unconnected_feature(offset=19, out_col="adc_unconnected_02")
    _build_unconnected_feature(offset=28, out_col="adc_unconnected_03")

    # # number of channels with toa and with tot
    df_inputs[f"nchtoa"] = df["nchtoa"]
    df_inputs[f"nchtot"] = df["nchtot"]
    
    return df_inputs


def prepare_dnn_inputs(cfg, column_tag, inferencer, nch_to_use=None):
    print("Hello from prepare_dnn_inputs()!")

    # Open file and load tree
    print(f"Preparing DNN inputs from Run{cfg.run} for module {cfg.modulename}...")
    # make output folder
    os.makedirs(name=cfg.dnn_training_input_folder, exist_ok=True)

    if nch_to_use is None:
        nch_to_use = cfg.nch

    adc_channel_indices = [x for x in range(nch_to_use)]
    target_columns = [f"adc_ch{idx:03}_pedsub{column_tag}" for idx in adc_channel_indices]
    event_ids_all = []

    def write_df(df: pd.DataFrame, filename: str, index: bool = True) -> None:
        utils.write_via_tmpdir(
            outfilename=os.path.join(cfg.dnn_training_input_folder, filename),
            suffix=".parquet",
            writer_fn=lambda tmp, data=df, use_index=index: data.to_parquet(tmp, engine="pyarrow", index=use_index, compression="zstd"),
        )

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):
        df_targets = df_chunk[target_columns].copy().astype("float32")
        df_inputs  = make_input_df(cfg=cfg, df=df_chunk, adc_channel_indices=adc_channel_indices, column_tag=column_tag)

        print(df_targets)
        print(df_inputs)

        # write input and target chunks
        write_df(df_targets, f"targets_chunk{idx:03d}.parquet")
        write_df(df_inputs, f"inputs_chunk{idx:03d}.parquet")
        event_ids_all.append(df_chunk.index.to_numpy(np.int64))

    event_ids = np.unique(np.concatenate(event_ids_all))
    rng_split = np.random.default_rng(6789)
    perm = rng_split.permutation(len(event_ids))
    test_frac = 0.2
    n_test = int(round(test_frac * len(event_ids)))
    test_ids = event_ids[perm[:n_test]]

    df_split = pd.DataFrame(
        {
            "event_id_global": event_ids,
            "split": np.where(np.isin(event_ids, test_ids), "test", "train"),
        }
    )
    write_df(df_split, "event_split_train_test.parquet", index=False)
    print(df_split)

    print(f"--> Wrote input, target, and split DFs to: {cfg.dnn_training_input_folder}")




if __name__ == '__main__':
  main()
