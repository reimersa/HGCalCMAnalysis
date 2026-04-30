#! /eos/user/a/areimers/torch-env/bin/python
import os

import classes
import inferencers

import calculate_means_stds
import convert_to_df
import compute_covariances_and_eigen
import fit_covariance_noise_model
import compute_predictor_analytic
import add_vars_and_selections
import prepare_dnn_inputs
import train_dnn


def main():
    modulenames = ["ML_F3WC_IH0182"]

    # Events used to derive the correction artifacts.
    # selection_for_correction = "selection_full"
    selection_for_correction = "selection_trigtime"

    n_coherent_noise_model = 3
    # per_channel_cols = ["channel_indices", "erx_indices"] + [f"eigvec_{i}" for i in range(20)] + [f"adc_unconnected_{i:02d}" for i in range(4)]
    # per_channel_cols = ["channel_indices", "erx_indices"] + [f"adc_unconnected_{i:02d}" for i in range(4)]
    per_channel_cols = ["channel_indices", "erx_indices", "cell_area_fraction"] + [f"adc_unconnected_{i:02d}" for i in range(4)]

    cfgs = [
        classes.AnalysisConfig(
            modulename=x,
            # Correction-source run. This is where the predictor/model is derived.
            run="112044_112050_112060_112073_adcmax10",
            derive_correction=True,
            selection_for_correction=selection_for_correction,
            run_for_pedestal=112044,
            run_for_correction="112044_112050_112060_112073_adcmax10",
            module_for_correction=x,
            standardize_std=False,
            inputfoldertag="",
        )
        for x in modulenames
    ]

    for cfg in cfgs:
        # if cfg.is_pedestal:
        #     calculate_means_stds.calculate_means_stds(cfg=cfg, print_vals=True)

        # if isinstance(cfg.run, int):
        #     convert_to_df.convert_to_df(cfg=cfg, adcmax=cfg.adcmax)
        # else:
        #     convert_to_df.convert_to_df_synthetic(cfg=cfg, adcmax=cfg.adcmax)

        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)

        # add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer)
        inferencer_sel = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=selection_for_correction)


        # Analytic correction derived from the selected events only.
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer_sel, column_tag="")
        # compute_predictor_analytic.compute_predictor_analytic(cfg=cfg)


        # DNN correction derived from the selected events only.
        # prepare_dnn_inputs.prepare_dnn_inputs(cfg=cfg, column_tag="", inferencer=inferencer_sel, nch_to_use=None)
        # add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer)
        train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=per_channel_cols, nodes=[256, 256, 256, 32], dropout=0.00, tag="", batch_samples=1024, epochs=500)


if __name__ == "__main__":
    main()
