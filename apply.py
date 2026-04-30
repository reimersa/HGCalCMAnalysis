#! /eos/user/a/areimers/torch-env/bin/python
import os

import classes
import inferencers

import convert_to_df
import add_correction_analytic
import add_correction_dnn
import add_projections_onto_noisemode
import add_vars_and_selections
import compute_covariances_and_eigen
import fit_covariance_noise_model
import plot
import plot_summaries


def main():
    modulenames = ["ML_F3WC_IH0182"]

    # Selection used only for plots/evaluation on the target run.
    # selection = "selection_full"
    selection = "selection_trigtime"

    # Selection that was used when the stored correction was derived.
    # selection_for_correction = "selection_full"
    selection_for_correction = "selection_trigtime"

    n_coherent_noise_model = 3
    # per_channel_cols = ["channel_indices", "erx_indices"] + [f"eigvec_{i}" for i in range(20)] + [f"adc_unconnected_{i:02d}" for i in range(4)]
    # per_channel_cols = ["channel_indices", "erx_indices"] + [f"adc_unconnected_{i:02d}" for i in range(4)]
    per_channel_cols = ["channel_indices", "erx_indices", "cell_area_fraction"] + [f"adc_unconnected_{i:02d}" for i in range(4)]

    cfgs = [
        classes.AnalysisConfig(
            modulename=x,
            # Target run to which an already-derived correction is applied.
            # run=112044,
            # run=112051,
            run="112044_112050_112060_112073_adcmax10",
            derive_correction=False,

            selection_for_correction=selection_for_correction,
            run_for_pedestal=112044,
            run_for_correction="112044_112050_112060_112073_adcmax10",
            module_for_correction="ML_F3WC_IH0182",

            standardize_std=False,
            inputfoldertag="",
        )
        for x in modulenames
    ]

    for cfg in cfgs:


        if isinstance(cfg.run, int):
            convert_to_df.convert_to_df(cfg=cfg, adcmax=cfg.adcmax)
        else:
            convert_to_df.convert_to_df_synthetic(cfg=cfg, adcmax=cfg.adcmax)


        # # # Use the full inferencer when writing corrected columns back to disk.
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)
        add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer)

        # # # Use the selected inferencer for plots and summaries.
        inferencer_sel = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=selection)


        # # # Uncorrected
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer_sel, column_tag="")
        fit_covariance_noise_model.fit_covariance_noise_model(cfg=cfg, column_tag="", n_coherent=n_coherent_noise_model)


        # # # Analytic correction
        add_correction_analytic.add_correction_analytic(cfg=cfg, inferencer=inferencer)
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_analytic_k0")
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_analytic_k0")
        fit_covariance_noise_model.fit_covariance_noise_model(cfg=cfg, column_tag="_resid_analytic_k0", n_coherent=n_coherent_noise_model)


        # # # DNN correction
        add_correction_dnn.add_correction_dnn(cfg, inferencer, nodes=[256, 256, 256, 32], dropout=0.0, tag="", column_tag="", per_channel_cols=per_channel_cols, infer_batch=8192, plot_dir_loss=os.path.join(cfg.plotfolder_base, selection, "dnn_loss"))
        # add_correction_dnn.add_correction_dnn(cfg, inferencer, nodes=[32, 32], dropout=0.0, tag="", column_tag="", per_channel_cols=per_channel_cols, infer_batch=8192, plot_dir_loss=os.path.join(cfg.plotfolder_base, selection, "dnn_loss"))
        
        add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer)
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_dnn")
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_dnn")
        fit_covariance_noise_model.fit_covariance_noise_model(cfg=cfg, column_tag="_resid_dnn", n_coherent=n_coherent_noise_model)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_dnn", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_dnn", k=0)


        # # # Evaluation and plots on the target run.
        plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="", selection=selection, n_coherent_noise_model=n_coherent_noise_model)
        plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_analytic_k0", selection=selection, n_coherent_noise_model=n_coherent_noise_model)
        plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_dnn", selection=selection, n_coherent_noise_model=n_coherent_noise_model)
        if cfg.run == 112044:
            plot_summaries.plot_summaries(cfg=cfg, inferencer=inferencer_sel, selection=selection, column_tags=["_resid_analytic_k0", "_resid_dnn"], y_range=(0.8, 1.8), cm_x_range=(-12., 12.), cm_profile_y_range=(-2., 3.))
        else:
            plot_summaries.plot_summaries(cfg=cfg, inferencer=inferencer_sel, selection=selection, column_tags=["_resid_analytic_k0", "_resid_dnn"], y_range=(1., 4.), cm_x_range=(-50., 25.), cm_profile_y_range=(-2., 3.))


if __name__ == "__main__":
    main()
