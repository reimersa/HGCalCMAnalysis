#! /eos/user/a/areimers/torch-env/bin/python
import os

import classes
import inferencers
import functions_plot

import calculate_means_stds
import convert_to_df
import compute_covariances_and_eigen
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


    # modulenames = ["ML_F3W_WXIH0190"]
    modulenames = ["ML_F3WC_IH0182"]

    selection = "selection_full"
    # selection = "selection_test"
    # selection = "selection_train"
    # selection = "selection_trigtime"
    # selection = "selection_notot_notoa"
    # selection = "selection_toa0to10"
    # selection = "selection_toa10to20"
    # selection = "selection_toa20to30"


    # cfgs = [classes.AnalysisConfig(
    #         modulename=x, 
    #         run=112044,
    #         # run=112050,
    #         run_for_pedestal=112044, # this determines where the means will be written, so must be the same as the run we're reading in
    #         run_for_correction=112044, 
    #         standardize_std = False,
    #         inputfoldertag = "",
    #     ) 
    #     for x in modulenames
    # ]


    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            # run=112044,
            # run="112044_112050",
            # run="112044_112050_full",
            # run = "112050_adcmax10",
            # run = "112044_112050_112060_112073_adcmax10",
            run=112050, # 300 GeV
            # run=112060, # 100 GeV
            # run=112068, # 50 GeV
            # run=112051, # 300 GeV
            # run="112073_outer",
            # run="112060_outer",

            run_for_pedestal=112044,
            # run_for_correction="112044_112050", 
            # run_for_correction="112050_adcmax50", 
            run_for_correction="112044_112050_112060_112073_adcmax10", 
            # run_for_correction="112044_112050_full", 
            standardize_std = False,
            inputfoldertag = "",
        ) 
        for x in modulenames
    ]

    for cfg in cfgs:

        # if cfg.is_pedestal:
        #     calculate_means_stds.calculate_means_stds(cfg=cfg, print_vals=True)
        if isinstance(cfg.run, int):
            convert_to_df.convert_to_df(cfg=cfg, adcmax=cfg.adcmax)
        else:
            convert_to_df.convert_to_df_synthetic(cfg=cfg, adcmax=cfg.adcmax)

        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="")

        # if cfg.is_pedestal:
        #     compute_predictor_analytic.compute_predictor_analytic(cfg=cfg)

        add_correction_analytic.add_correction_analytic(cfg=cfg, inferencer=inferencer)
        
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_k0")
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_with_unconnected_k0")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_with_unconnected_k0")

        add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer)
        
        # if cfg.is_pedestal:
        #   compute_predictor_noisemodes.compute_predictor_noisemodes(cfg=cfg, column_tag="", k=1)
        #   compute_predictor_noisemodes.compute_predictor_noisemodes(cfg=cfg, column_tag="_resid_analytic_k0", k=1)
        #   compute_predictor_noisemodes.compute_predictor_noisemodes(cfg=cfg, column_tag="", k=3)
        #   compute_predictor_noisemodes.compute_predictor_noisemodes(cfg=cfg, column_tag="_resid_analytic_k0", k=3)
        #   compute_predictor_noisemodes.compute_predictor_noisemodes(cfg=cfg, column_tag="", k=5)
        #   compute_predictor_noisemodes.compute_predictor_noisemodes(cfg=cfg, column_tag="_resid_analytic_k0", k=5)

        # add_correction_noisesubtraction.add_correction_noisesubtraction(cfg=cfg, inferencer=inferencer, column_tag="", k=1)
        # add_correction_noisesubtraction.add_correction_noisesubtraction(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0", k=1)
        # add_correction_noisesubtraction.add_correction_noisesubtraction(cfg=cfg, inferencer=inferencer, column_tag="", k=3)
        # add_correction_noisesubtraction.add_correction_noisesubtraction(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0", k=3)
        # add_correction_noisesubtraction.add_correction_noisesubtraction(cfg=cfg, inferencer=inferencer, column_tag="", k=5)
        # add_correction_noisesubtraction.add_correction_noisesubtraction(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0", k=5)

        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_submodes1")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_submodes1")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_k0_pred_submodes1")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0_resid_submodes1")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_submodes3")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_submodes3")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_k0_pred_submodes3")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0_resid_submodes3")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_submodes5")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_submodes5")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_k0_pred_submodes5")
        # compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0_resid_submodes5")

        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_k0", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_with_unconnected_k0", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_with_unconnected_k0", k=0)

        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_k0_pred_submodes1", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0_resid_submodes1", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_k0_pred_submodes3", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0_resid_submodes3", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_analytic_k0_pred_submodes5", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_analytic_k0_resid_submodes5", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_submodes1", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_submodes1", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_submodes3", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_submodes3", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_submodes5", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_submodes5", k=0)


        inferencer_sel = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=selection)


        # if cfg.is_pedestal:
            # prepare_dnn_inputs.prepare_dnn_inputs(cfg=cfg, column_tag="", inferencer=inferencer_sel, nch_to_use=None)
            # add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer) # rerun to add train/test split as columns to main df

            ### DNNs for ped+outer-beam
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=["channel_indices", "erx_indices"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], nodes=[32, 32], dropout=0.00, tag="", batch_samples=32, epochs=1000)
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=["channel_indices", "erx_indices"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], nodes=[32, 32], dropout=0.00, tag="", batch_samples=1024, epochs=1000)
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=["channel_indices", "erx_indices"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], nodes=[32, 32], dropout=0.00, tag="", batch_samples=8192, epochs=1000)
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=["channel_indices", "erx_indices"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], nodes=[512, 512, 512, 64], dropout=0.00, tag="", batch_samples=1024, epochs=1000)
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=["cm_thiserx", "cm_othererxonroc"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], nodes=[32, 32], dropout=0.00, tag="nochannelID", batch_samples=1024, epochs=1000)

            ### DNNs for ped-only to optimize performance
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=["channel_indices", "erx_indices"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], nodes=[32, 32], dropout=0.00, tag="", batch_samples=32, epochs=1000)
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=["channel_indices", "erx_indices"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], nodes=[32, 32], dropout=0.00, tag="leakyrelu", batch_samples=32, epochs=1000)


            # for DNN that predicts C channels at once
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], nodes=[32, 32], dropout=0.00, tag="", batch_samples=1024, epochs=1000)
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], nodes=[512, 512], dropout=0.00, tag="", batch_samples=1024, epochs=1000)
        
        add_correction_dnn.add_correction_dnn(cfg, inferencer, nodes=[32, 32], dropout=0.0, tag="", column_tag="", per_channel_cols=["channel_indices", "erx_indices"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], infer_batch=8192, plot_dir_loss=os.path.join(cfg.plotfolder_base, selection, "dnn_loss"))
        # add_correction_dnn.add_correction_dnn(cfg, inferencer, nodes=[512, 512, 512, 64], dropout=0.0, tag="", column_tag="", per_channel_cols=["channel_indices", "erx_indices"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], infer_batch=8192, plot_dir_loss=os.path.join(cfg.plotfolder_base, selection, "dnn_loss"))
        # add_correction_dnn.add_correction_dnn(cfg, inferencer, nodes=[32, 32], dropout=0.0, tag="nochannelID", column_tag="", per_channel_cols=["cm_thiserx", "cm_othererxonroc"]+[f"eigvec_{i}" for i in range(20)]+[f"adc_unconnected_{i:02d}" for i in range(4)], infer_batch=8192, plot_dir_loss=os.path.join(cfg.plotfolder_base, selection, "dnn_loss"))

        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_pred_dnn")
        compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer, column_tag="_resid_dnn")
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_pred_dnn", k=0)
        # add_projections_onto_noisemode.add_projections_onto_noisemode(cfg=cfg, inferencer=inferencer, column_tag="_resid_dnn", k=0)




        # functions_plot.plot_coh_inc(value_iterator=inferencer_sel.full_df_iter, adc_colname_template_true=f"adc_ch*_pedsub", adc_colname_template_corr=f"adc_ch*_pedsub_resid_analytic_k0", nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, out_root=os.path.join(cfg.plotfolder_base, selection, "coherent_noise", "resid_analytic_k0"), trunc_fracs = (1.0,))
        # functions_plot.plot_coh_inc(value_iterator=inferencer_sel.full_df_iter, adc_colname_template_true=f"adc_ch*_pedsub", adc_colname_template_corr=f"adc_ch*_pedsub_resid_analytic_with_unconnected_k0", nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, out_root=os.path.join(cfg.plotfolder_base, selection, "coherent_noise", "resid_analytic_with_unconnected_k0"), trunc_fracs = (1.0,))
        # functions_plot.plot_coh_inc(value_iterator=inferencer_sel.full_df_iter, adc_colname_template_true=f"adc_ch*_pedsub", adc_colname_template_corr=f"adc_ch*_pedsub_resid_dnn", nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, out_root=os.path.join(cfg.plotfolder_base, selection, "coherent_noise", "resid_dnn"), trunc_fracs = (1.0,))
        # functions_plot.plot_coh_inc(value_iterator=inferencer_sel.full_df_iter, adc_colname_template_true=f"adc_ch*_pedsub", adc_colname_template_corr=f"adc_ch*_pedsub_resid_submodes1", nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, out_root=os.path.join(cfg.plotfolder_base, selection, "coherent_noise", "resid_submodes1"), trunc_fracs = (1.0,))
        # functions_plot.plot_coh_inc(value_iterator=inferencer_sel.full_df_iter, adc_colname_template_true=f"adc_ch*_pedsub", adc_colname_template_corr=f"adc_ch*_pedsub_resid_submodes3", nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, out_root=os.path.join(cfg.plotfolder_base, selection, "coherent_noise", "resid_submodes3"), trunc_fracs = (1.0,))
        # functions_plot.plot_coh_inc(value_iterator=inferencer_sel.full_df_iter, adc_colname_template_true=f"adc_ch*_pedsub", adc_colname_template_corr=f"adc_ch*_pedsub_resid_submodes5", nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, out_root=os.path.join(cfg.plotfolder_base, selection, "coherent_noise", "resid_submodes5"), trunc_fracs = (1.0,))
        # functions_plot.plot_coh_inc(value_iterator=inferencer_sel.full_df_iter, adc_colname_template_true=f"adc_ch*_pedsub", adc_colname_template_corr=f"adc_ch*_pedsub_resid_analytic_k0_resid_submodes1", nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, out_root=os.path.join(cfg.plotfolder_base, selection, "coherent_noise", "resid_analytic_k0_resid_submodes1"), trunc_fracs = (1.0,))
        # functions_plot.plot_coh_inc(value_iterator=inferencer_sel.full_df_iter, adc_colname_template_true=f"adc_ch*_pedsub", adc_colname_template_corr=f"adc_ch*_pedsub_resid_analytic_k0_resid_submodes3", nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, out_root=os.path.join(cfg.plotfolder_base, selection, "coherent_noise", "resid_analytic_k0_resid_submodes3"), trunc_fracs = (1.0,))
        # functions_plot.plot_coh_inc(value_iterator=inferencer_sel.full_df_iter, adc_colname_template_true=f"adc_ch*_pedsub", adc_colname_template_corr=f"adc_ch*_pedsub_resid_analytic_k0_resid_submodes5", nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx, out_root=os.path.join(cfg.plotfolder_base, selection, "coherent_noise", "resid_analytic_k0_resid_submodes5"), trunc_fracs = (1.0,))


        plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_submodes1", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_submodes1", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_submodes3", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_submodes3", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_submodes5", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_submodes5", selection=selection)

        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_analytic_k0", selection=selection)
        plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_analytic_k0", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_analytic_with_unconnected_k0", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_analytic_with_unconnected_k0", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_dnn", selection=selection)
        plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_dnn", selection=selection)

        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_analytic_k0_pred_submodes1", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_analytic_k0_resid_submodes1", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_analytic_k0_pred_submodes3", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_analytic_k0_resid_submodes3", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_pred_analytic_k0_pred_submodes5", selection=selection)
        # plot.plot(cfg=cfg, inferencer=inferencer_sel, column_tag="_resid_analytic_k0_resid_submodes5", selection=selection)














if __name__ == "__main__":
    main()
    
