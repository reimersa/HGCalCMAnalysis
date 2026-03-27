#! /eos/user/a/areimers/torch-env/bin/python
import os

import classes
import inferencers
import functions_plot

import calculate_means_stds
import convert_to_df
import compute_covariances_and_eigen
import fit_covariance_noise_model
import compute_predictor_analytic
import add_correction_analytic
import add_vars_and_selections
import compute_predictor_noisemodes
import add_correction_noisesubtraction
import add_projections_onto_noisemode
import plot
import prepare_dnn_inputs
import train_dnn
import add_correction_dnn




def main():


    # modulenames = ["ML_F3WC_IH0182"]
    # modulenames = ["ML_F3WC_IH0194"]
    modulenames = ["ML_F3WC_IH0191"]

    selection = "selection_full"
    # selection = "selection_test"
    # selection = "selection_train"
    # selection = "selection_trigtime"
    # selection = "selection_notot_notoa"
    # selection = "selection_toa0to10"
    # selection = "selection_toa10to20"
    # selection = "selection_toa20to30"

    n_coherent_noise_model = 3


    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            # run=112044,
            # run="112044_112050",
            # run="112044_112050_full",
            # run = "112044_112050_112060_112073_adcmax10",
            run=112050, # 300 GeV
            # run = "112050_adcmax10", # 300 GeV
            # run=112051, # 300 GeV
            # run = "112051_adcmax10", # 300 GeV
            # run=112060, # 100 GeV
            # run=112068, # 50 GeV
            # run="112068_adcmax10", # 50 GeV
            # run=112073, # 20 GeV
            # run="112073_outer",
            # run="112060_outer",

            run_for_pedestal=112044,
            run_for_correction="112044_112050_112060_112073_adcmax10", 
            module_for_correction="ML_F3WC_IH0182",
            standardize_std = False,
            inputfoldertag = "",
        ) 
        for x in modulenames
    ]

    for cfg in cfgs:

        if cfg.is_pedestal:
            calculate_means_stds.calculate_means_stds(cfg=cfg, print_vals=True)
            # raise ValueError("stop now")
        if isinstance(cfg.run, int):
            convert_to_df.convert_to_df(cfg=cfg, adcmax=cfg.adcmax)
        else:
            convert_to_df.convert_to_df_synthetic(cfg=cfg, adcmax=cfg.adcmax)

        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="")
        fit_covariance_noise_model.fit_covariance_noise_model(cfg=cfg, column_tag="", n_coherent=n_coherent_noise_model)

        # if cfg.is_pedestal:
        #     compute_predictor_analytic.compute_predictor_analytic(cfg=cfg)

        add_correction_analytic.add_correction_analytic(cfg=cfg, inferencer=inferencer)
        
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_k0")
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0")

        fit_covariance_noise_model.fit_covariance_noise_model(cfg=cfg, column_tag="_resid_analytic_k0", n_coherent=n_coherent_noise_model)

        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_with_unconnected_k0")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_with_unconnected_k0")

        add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer)
        
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_k0", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0", k=0)


        inferencer_sel = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=selection)


        # if cfg.is_pedestal:
            # prepare_dnn_inputs.prepare_dnn_inputs(cfg=cfg, column_tag="", inferencer=inferencer_sel, nch_to_use=None)
            # add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer) # rerun to add train/test split as columns to main df

            ### DNNs for ped+outer-beam
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=["channel_indices", "erx_indices"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], nodes=[32, 32], dropout=0.00, tag="", batch_samples=32, epochs=1000)
        
        add_correction_dnn.add_correction_dnn(cfg, inferencer, nodes=[32, 32], dropout=0.0, tag="", column_tag="", per_channel_cols=["channel_indices", "erx_indices"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], infer_batch=8192, plot_dir_loss=os.path.join(cfg.plotfolder_base, selection, "dnn_loss"))

        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_dnn")
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_dnn")

        fit_covariance_noise_model.fit_covariance_noise_model(cfg=cfg, column_tag="_resid_dnn", n_coherent=n_coherent_noise_model)

        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_dnn", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_dnn", k=0)

        # plot.plot_coherent_noise(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_analytic_k0", selection=selection, trunc_fracs=(1.0,))
        # plot.plot_coherent_noise(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_dnn", selection=selection, trunc_fracs=(1.0,))


        plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="", selection=selection, n_coherent_noise_model=n_coherent_noise_model)
        plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_analytic_k0", selection=selection, n_coherent_noise_model=n_coherent_noise_model)
        plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_dnn", selection=selection, n_coherent_noise_model=n_coherent_noise_model)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_analytic_k0", selection=selection, n_coherent_noise_model=n_coherent_noise_model)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_dnn", selection=selection, n_coherent_noise_model=n_coherent_noise_model)

        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_analytic_with_unconnected_k0", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_analytic_with_unconnected_k0", selection=selection)














if __name__ == "__main__":
    main()
    
