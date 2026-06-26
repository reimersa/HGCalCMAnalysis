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
        help="List of module names to compute covariances for.",
    )
    parser.add_argument(
        "--module-for-correction",
        type=str,
        required=True,
        help="Module whose correction context this covariance output should correspond to.",
    )
    parser.add_argument(
        "-s",
        "--selection",
        type=str,
        default="",
        help="Only use events for which the given selection column is true when deriving these covariances.",
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
            standardize_std = False,
            inputfoldertag = "",
        )
        for x in args.modules
    ]

    for cfg in cfgs:
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=args.selection)
        compute_covariances_and_eigen(
            cfg=cfg,
            inferencer=inferencer,
            column_tag=args.column_tag,
        )


def compute_covariances_and_eigen(cfg, inferencer, column_tag) -> None:
    print("Hello from compute_covariances_and_eigen()!")

    cm_cols = [f"cm_erx{i:02d}_pedsub" for i in range(cfg.ncmchannels)]
    pred_cols = [f"adc_ch{i:03d}_pedsub{column_tag}" for i in range(cfg.nch)]
    unc_cols = [f"adc_ch{(x*cfg.nch_per_erx + off):03d}_pedsub{column_tag}" for x in range(cfg.nerx) for off in [8, 17, 19, 28]]
    pred_cols_no_unc = [x for x in pred_cols if x not in unc_cols]
    artifact_tag = column_tag

    sigma_cc_df = utils.compute_cov_streaming(inferencer.df_cols_iter(colnames=cm_cols), inferencer.df_cols_iter(colnames=cm_cols))
    sigma_mc_df = utils.compute_cov_streaming(inferencer.df_cols_iter(colnames=pred_cols), inferencer.df_cols_iter(colnames=cm_cols))
    sigma_mm_df = utils.compute_cov_streaming(inferencer.df_cols_iter(colnames=pred_cols), inferencer.df_cols_iter(colnames=pred_cols))
    sigma_cucu_df = utils.compute_cov_streaming(inferencer.df_cols_iter(colnames=cm_cols+unc_cols), inferencer.df_cols_iter(colnames=cm_cols+unc_cols))
    sigma_mnou_cu_df = utils.compute_cov_streaming(inferencer.df_cols_iter(colnames=pred_cols_no_unc), inferencer.df_cols_iter(colnames=cm_cols+unc_cols))
    A = sigma_mm_df.to_numpy(dtype=np.float64)
    A = 0.5*(A + A.T)
    w = np.linalg.eigvalsh(A)
    print("min eig:", w.min(), "max eig:", w.max(), "neg frac:", (w < -1e-8).mean())
    sigma_mcmc_df = utils.compute_cov_streaming(inferencer.df_cols_iter(colnames=pred_cols+cm_cols), inferencer.df_cols_iter(colnames=pred_cols+cm_cols))

    vals_mm, vecs_mm = utils.compute_eig_from_cov(C=sigma_mm_df)
    vals_mm_df = pd.DataFrame({"eigval": vals_mm})
    vecs_mm_df = pd.DataFrame(vecs_mm, index=sigma_mm_df.index, columns=[f"eigvec_{i}" for i in range(vecs_mm.shape[1])])

    vals_mcmc, vecs_mcmc = utils.compute_eig_from_cov(C=sigma_mcmc_df)
    vals_mcmc_df = pd.DataFrame({"eigval": vals_mcmc})
    vecs_mcmc_df = pd.DataFrame(vecs_mcmc, index=sigma_mcmc_df.index, columns=[f"eigvec_{i}" for i in range(vecs_mcmc.shape[1])])

    def write_df(df: pd.DataFrame, filename: str, index: bool = True) -> None:
        utils.write_via_tmpdir(
            outfilename=os.path.join(cfg.own_covs_folder, filename),
            suffix=".parquet",
            writer_fn=lambda tmp, data=df, use_index=index: data.to_parquet(tmp, index=use_index, compression="zstd"),
        )

    os.makedirs(cfg.own_covs_folder, exist_ok=True)
    write_df(sigma_cc_df, "sigma_cc.parquet")
    write_df(sigma_mc_df, f"sigma_mc{artifact_tag}.parquet")
    write_df(sigma_mm_df, f"sigma_mm{artifact_tag}.parquet")
    write_df(sigma_mcmc_df, f"sigma_mcmc{artifact_tag}.parquet")
    write_df(sigma_cucu_df, f"sigma_cucu{artifact_tag}.parquet")
    write_df(sigma_mnou_cu_df, f"sigma_mnou_cu{artifact_tag}.parquet")
    print(f"Wrote covariances to folder: {cfg.own_covs_folder}")

    write_df(vals_mm_df, f"eigenvalues_mm{artifact_tag}.parquet")
    write_df(vecs_mm_df, f"eigenvectors_mm{artifact_tag}.parquet")

    write_df(vals_mcmc_df, f"eigenvalues_mcmc{artifact_tag}.parquet")
    write_df(vecs_mcmc_df, f"eigenvectors_mcmc{artifact_tag}.parquet")
    print(f"Wrote eigenvalues and -vectors to the same folder: {cfg.own_covs_folder}")









if __name__ == "__main__":
    main()
    
