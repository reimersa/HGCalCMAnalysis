#! /eos/user/a/areimers/torch-env/bin/python

import argparse
import os

import inferencers
import classes
import utils
import functions_plot




def main():

    parser = argparse.ArgumentParser(description="Make all the plots.")
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        default=112044,
        help="Run number to make plots for (e.g. 112044).",
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
        help="Column tag to be appended at the end of 'adc_ch{i:03d}_pedsub'. E.g.: '_pred_analytic_k0' to make plots for the analytic_k0 predictions.",
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
        help="List of module names to make plots for.",
    )
    parser.add_argument(
        "-s",
        "--selection",
        type=str,
        default="full",
        metavar="SEL",
        help="Only plot events for which the column 'selection_{SEL}' is true. These need to be constructed and added to the df before, of course.",
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
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=args.selection)
        plot(cfg=cfg, inferencer=inferencer, column_tag=args.column_tag, selection=args.selection)


def plot(cfg, inferencer, column_tag, selection) -> None:
    print("Hello from plot()!")
    plot_dir = os.path.join(cfg.plotfolder_base, selection)
    os.makedirs(plot_dir, exist_ok=True)

    method_subfolder = column_tag.strip("_")
    if method_subfolder == "":
        method_subfolder = "true"

    adc_ch_pattern = f"adc_ch*_pedsub{column_tag}"


    if "notot_notoa" in selection:
        yrange = (-30., 100) 
        if cfg.is_pedestal:
            if isinstance(cfg.run, int):
                yrange = (-20., 20.)
            else:
                yrange = (-30., 200.)
    else:
        yrange = (-100., 200.)
        if cfg.is_pedestal:
            if isinstance(cfg.run, int):
                yrange = (-20., 20.)
            else:
                yrange = (-30., 200.)
        if cfg.run == "112073_outer":
            yrange = (-20., 50.)
        if cfg.run == "112060_outer":
            yrange = (-20., 50.)
        if cfg.run == "112050_adcmax10":
            yrange = (-20., 20.)
        if cfg.run == "112044_112050_112060_112073_adcmax10":
            yrange = (-20., 20.)
    zrange_cov = (-1., 1.) if cfg.standardize_std else (-4., 4.)

    functions_plot.plot_cov_corr(cfg=cfg, column_tag=column_tag, axis_title=adc_ch_pattern, zrange_cov=zrange_cov, plot_dir=os.path.join(plot_dir, "covcorr", method_subfolder))
    # functions_plot.plot_eigenvalues(cfg=cfg, column_tag=column_tag, out_root=os.path.join(plot_dir, "eigen", method_subfolder))
    # functions_plot.plot_eigenvectors(cfg=cfg, column_tag=column_tag, top=4, out_root=os.path.join(plot_dir, "eigen", method_subfolder))

    functions_plot.plot_vs_chidx(cfg=cfg, varname_y_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_chidx", method_subfolder), nbins_x=cfg.nerx*cfg.nch_per_erx, nbins_y=80, y_range=yrange)
    # functions_plot.plot_2d_multicol_vs_var(varname_x="event_id", varname_y_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_event_id", method_subfolder), nbins_x=200, nbins_y=80, y_range=yrange, make_profile_plot=True)
    # functions_plot.plot_2d_multicol_vs_var(varname_x=f"adc_sum_pedsub", varname_y_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_adcsum", method_subfolder), nbins_y=80, y_range=yrange, nbins_x=100)
    # functions_plot.plot_2d_multicol_vs_var(varname_x=f"adc_sum_allchannels_pedsub", varname_y_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_adcsum", method_subfolder), nbins_y=80, y_range=yrange, nbins_x=100)

    # ## plots per eRx
    for erx in range(cfg.nerx):
        functions_plot.plot_2d_multicol_vs_var(varname_x=f"cm_erx{erx:02d}_pedsub", varname_y_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_cm", method_subfolder), nbins_y=75, x_range=(-50, 25), y_range=yrange)
        # functions_plot.plot_2d_multicol_vs_var(varname_x=f"cm_erx{erx:02d}_pedsub", varname_y_template=f"proj_mode0_adc_chall_pedsub{column_tag}", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_projmode0_2d_vs_cm", method_subfolder), nbins_y=80, y_range=yrange)
        # functions_plot.plot_2d_multicol_vs_var(varname_x=f"nchtoa", varname_y_template=f"cm_erx{erx:02d}_pedsub", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "cm_2d_vs_ntoa", method_subfolder), nbins_y=80, y_range=(-30, 30))
        # functions_plot.plot_2d_multicol_vs_var(varname_x=f"adc_sum_pedsub", varname_y_template=f"cm_erx{erx:02d}_pedsub", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "cm_2d_vs_adcsum", method_subfolder), nbins_y=80, y_range=(-30, 30), nbins_x=100)
        # functions_plot.plot_2d_multicol_vs_var(varname_x=f"adc_sum_allchannels_pedsub", varname_y_template=f"cm_erx{erx:02d}_pedsub", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "cm_2d_vs_adcsum", method_subfolder), nbins_y=80, y_range=(-30, 30), nbins_x=100)

        # functions_plot.plot_1d_multicol(varname_template=f"erx{erx:02d}_hastot", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=(-0.5, 1.5), nbins_x=2)
        # functions_plot.plot_1d_multicol(varname_template=f"erx{erx:02d}_hastoa", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=(-0.5, 1.5), nbins_x=2)
        # functions_plot.plot_1d_multicol(varname_template=f"cm_erx{erx:02d}_pedsub", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=(-70, 70), nbins_x=140)

    # 1d plots
    # functions_plot.plot_1d_multicol(varname_template="nchtot", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=(0, 30), nbins_x=30)
    # functions_plot.plot_1d_multicol(varname_template="nchtoa", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=(0, 222), nbins_x=222)
    # functions_plot.plot_1d_multicol(varname_template=f"adc_sum_pedsub{column_tag}", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=(-1000, 1000), nbins_x=100)
    # functions_plot.plot_1d_multicol(varname_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=yrange, nbins_x=100)

    # 1d plots with Gauss fits around 0 (only makes sense for distributions expected to peak at 0...)
    functions_plot.plot_1d_multicol(varname_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=yrange, nbins_x=200, do_gauss_fit=True, gauss_p0=(1E6, 0, 3))
    functions_plot.plot_1d_multicol(varname_template=f"adc_ch005_pedsub{column_tag}", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=yrange, nbins_x=200, do_gauss_fit=True, gauss_p0=(1E6, 0, 3))
    functions_plot.plot_1d_multicol(varname_template=f"adc_ch196_pedsub{column_tag}", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=yrange, nbins_x=200, do_gauss_fit=True, gauss_p0=(1E6, 0, 3))
    functions_plot.plot_1d_multicol(varname_template=f"adc_ch195_pedsub{column_tag}", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=yrange, nbins_x=200, do_gauss_fit=True, gauss_p0=(1E6, 0, 3))
    functions_plot.plot_1d_multicol(varname_template=f"adc_ch121_pedsub{column_tag}", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "distributions_1d", method_subfolder), x_range=yrange, nbins_x=200, do_gauss_fit=True, gauss_p0=(1E6, 0, 3))

    functions_plot.plot_summary_1d_multicol(varname_template=adc_ch_pattern, varname_true_template="adc_ch*_pedsub", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "summaries_1d", method_subfolder), x_range=yrange, nbins_x=200)


    # # ### 2d plots
    functions_plot.plot_2d_multicol_vs_var(varname_x=f"nchtoa", varname_y_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_ntoa", method_subfolder), nbins_y=80, y_range=yrange)
    functions_plot.plot_2d_multicol_vs_var(varname_x=f"nchtot", varname_y_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_ntot", method_subfolder), nbins_y=80, y_range=yrange)
    functions_plot.plot_2d_multicol_vs_var(varname_x=f"adc_sum_allchannels_pedsub", varname_y_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_adcsum", method_subfolder), nbins_y=80, y_range=yrange, nbins_x=80)

    # # trig time plots
    functions_plot.plot_2d_multicol_vs_var(varname_x="trig_time", varname_y_template=adc_ch_pattern, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_trigtime", method_subfolder), nbins_y=80, y_range=(0., 1000.))
    functions_plot.plot_2d_multicol_vs_var(varname_x="trig_time", varname_y_template="adc_sum_pedsub", value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_trigtime", method_subfolder), nbins_y=80, y_range=(0., 10000.))



    print(f"Successfully finished plotting everything!")


def plot_coherent_noise(cfg, inferencer, column_tag, selection, trunc_fracs=(1.0,)) -> None:
    plot_dir = os.path.join(cfg.plotfolder_base, selection)
    method_subfolder = column_tag.strip("_")
    if method_subfolder == "":
        method_subfolder = "true"

    functions_plot.plot_coh_inc(
        value_iterator=inferencer.full_df_iter,
        adc_colname_template_true="adc_ch*_pedsub",
        adc_colname_template_corr=f"adc_ch*_pedsub{column_tag}",
        nch_per_erx=cfg.nch_per_erx,
        nerx=cfg.nerx,
        out_root=os.path.join(plot_dir, "coherent_noise", method_subfolder),
        trunc_fracs=trunc_fracs,
    )









if __name__ == "__main__":
    main()
    
