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
            standardize_std=False,
            inputfoldertag="",
        ) 
        for x in args.modules
    ]
    for cfg in cfgs:
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=args.selection)
        prepare_dnn_inputs(cfg=cfg, column_tag=args.column_tag, inferencer=inferencer)



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

    # scaled components of top 5 eigenvectors
    vecs = cfg.load_from_corrections_cov_folder(filename=f"eigenvectors_mcmc{column_tag}.parquet")
    vals = cfg.load_from_corrections_cov_folder(filename=f"eigenvalues_mcmc{column_tag}.parquet")
    k = int(min(20, vecs.shape[1]))
    top_vecs = vecs[[f"eigvec_{i}" for i in range(k)]]
    vec_components_cm = top_vecs.loc[cm_columns]
    vec_components_meas = top_vecs.drop(index=cm_columns)
    
    for i in range(k):
        col = f"eigvec_{i}"
        keep_rows = [f"adc_ch{idx:03}_pedsub{column_tag}" for idx in adc_channel_indices]
        vec_meas = vec_components_meas[col].loc[keep_rows].to_numpy()   # shape (nch,)
        val = vals["eigval"].loc[i]
        df_inputs[col] = [vec_meas*np.sqrt(np.maximum(val, 0))] * len(df_inputs)

    # Per-event projections onto CM eigenmodes, shape (nevt, k)
    cm_mat = df[cm_columns].to_numpy(dtype=np.float32)
    U_cm = vec_components_cm[[f"eigvec_{i}" for i in range(k)]].to_numpy(dtype=np.float32)
    proj_cm = cm_mat @ U_cm
    for i in range(k):
        df_inputs[f"cm_proj_eigvec_{i}"] = proj_cm[:, i]

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

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):
        df_targets = df_chunk[target_columns].copy().astype("float32")
        df_inputs  = make_input_df(cfg=cfg, df=df_chunk, adc_channel_indices=adc_channel_indices, column_tag=column_tag)

        print(df_targets)
        print(df_inputs)

        # write input and target chunks
        df_targets.to_parquet(os.path.join(cfg.dnn_training_input_folder, f"targets_chunk{idx:03d}.parquet"), engine="pyarrow", index=True, compression="zstd")
        df_inputs.to_parquet(os.path.join(cfg.dnn_training_input_folder, f"inputs_chunk{idx:03d}.parquet"), engine="pyarrow", index=True, compression="zstd")
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
    df_split.to_parquet(os.path.join(cfg.dnn_training_input_folder, "event_split_train_test.parquet"), index=False, compression="zstd")
    print(df_split)

    print(f"--> Wrote input, target, and split DFs to: {cfg.dnn_training_input_folder}")




if __name__ == '__main__':
  main()
