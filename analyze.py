#! /eos/user/a/areimers/torch-env/bin/python

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np # type: ignore
import pandas as pd # type: ignore
import json
import matplotlib.pyplot as plt # type: ignore
from tqdm import tqdm # type: ignore
# import dcor  # type: ignore
import sys
from fnmatch import fnmatch

import classes
import inferencers
import utils


# -----------------------------
# Entry point
# -----------------------------
def main():

    run = 112050
    # run = 112068
    # run = 110398
    # run = 112044

    # run_for_pedestal = None # for the super large one from Run110398
    run_for_pedestal = 112044 # for all others done properly with a run tag (:

    # era_tag_pedestal = "/Sep2025TB" + "" if not run_for_pedestal else f"/Run{run_for_pedestal}" 
    # era_tag = f"/Sep2025TB/Run{run}" # 20 GeV electron

    standardize_std = False

    basetag = ""

    # configure paths and layout
    cfgs = [classes.AnalysisConfig(
            modulename=x, 
            run=run,
            run_for_pedestal=run_for_pedestal,
            # era_tag=era_tag,
            standardize_std = False,
            maxfiles_for_eval = 1,
            is_pedestal = run in [110398, 112044,],

            inputfoldertag = basetag,
        ) 
        for x in ["ML_F3WC_IH0182"]
        # for x in ["ML_F3WC_IH0181"]
        # for x in ["ML_F3WC_IH0182", "ML_F3WC_IH0190", "ML_F3WC_IH0191", "ML_F3WC_IH0192", "ML_F3WC_IH0194", "ML_F3WC_IH0196", "ML_F3WC_IH0197", "ML_F3WC_IH0198"]
    ]

    selstring = "full"
    # selstring = "trigtime"
    # selstring = "notot"
    # selstring = "notot_notoa"
    # selstring = "notot_notoa_roc0"
    # selstring = "notot_notoa_roc2"
    # selstring = "notot_roc0"
    # selstring = "notot_roc2"
    # selstring = "notot_notoa_nosaturatedadc"
    # selstring = "notot_notoa_nosaturatedadc_adcsumlt50"
    # selstring = "notot_notoa_nosaturatedadc_adcsumgt50"
    # selstring = "notot_notoa_nosaturatedadc_adcsumltm170"
    # selstring = "notot_notoa_nosaturatedadc_adcsumgtm170"
    # selstring = "withtot"


    selections_equal_per_selstring = {
        "full": [],
        "notot": [("nchtot", 0)],
        "notot_notoa": [("nchtot", 0), ("nchtoa", 0)],
        "notot_notoa_roc0": [("erx00_hastot", 0), ("erx01_hastot", 0), ("erx00_hastoa", 0), ("erx01_hastoa", 0)],
        "notot_notoa_roc2": [("erx04_hastot", 0), ("erx05_hastot", 0), ("erx04_hastoa", 0), ("erx05_hastoa", 0)],
        "notot_roc0": [("erx00_hastot", 0), ("erx01_hastot", 0)],
        "notot_roc2": [("erx04_hastot", 0), ("erx05_hastot", 0)],
        "notot_notoa_nosaturatedadc": [("nchtot", 0), ("nchtoa", 0)],
        "notot_notoa_nosaturatedadc_adcsumlt50": [("nchtot", 0), ("nchtoa", 0)],
        "notot_notoa_nosaturatedadc_adcsumgt50": [("nchtot", 0), ("nchtoa", 0)],
        "notot_notoa_nosaturatedadc_adcsumltm170": [("nchtot", 0), ("nchtoa", 0)],
        "notot_notoa_nosaturatedadc_adcsumgtm170": [("nchtot", 0), ("nchtoa", 0)],
        "withtot": [],
        "trigtime": [],
    }
    selections_smaller_than_per_selstring = {
        "full": [],
        "notot": [],
        "notot_notoa": [],
        "notot_notoa_roc0": [],
        "notot_notoa_roc2": [],
        "notot_roc0": [],
        "notot_roc2": [],
        "notot_notoa_nosaturatedadc": [("adc_max_pedsub", 900.)],
        "notot_notoa_nosaturatedadc_adcsumlt50": [("adc_max_pedsub", 900.), ("adc_sum_pedsub", 50)],
        "notot_notoa_nosaturatedadc_adcsumgt50": [("adc_max_pedsub", 900.)],
        "notot_notoa_nosaturatedadc_adcsumltm170": [("adc_max_pedsub", 900.), ("adc_sum_pedsub", -170)],
        "notot_notoa_nosaturatedadc_adcsumgtm170": [("adc_max_pedsub", 900.)],
        "withtot": [],
        "trigtime": [("trig_time", 113)],
    }
    selections_greater_than_per_selstring = {
        "full": [],
        "notot": [],
        "notot_notoa": [],
        "notot_notoa_roc0": [],
        "notot_notoa_roc2": [],
        "notot_roc0": [],
        "notot_roc2": [],
        "notot_notoa_nosaturatedadc": [],
        "notot_notoa_nosaturatedadc_adcsumlt50": [],
        "notot_notoa_nosaturatedadc_adcsumgt50": [("adc_sum_pedsub", 50)],
        "notot_notoa_nosaturatedadc_adcsumltm170": [],
        "notot_notoa_nosaturatedadc_adcsumgtm170": [("adc_sum_pedsub", -170)],
        "withtot": [("nchtot", 0)],
        "trigtime": [("trig_time", 107)],
    }

    selections_per_selstring = {
        "full": 
            "",
        "notot":
            "nchtot == 0",
        "notot_notoa":
            "nchtot == 0 and nchtoa == 0",
        "notot_notoa_roc0":
            "nchtot == 0 and nchtoa == 0 and erx00_hastot == 0 and erx01_hastot == 0 and erx00_hastoa == 0 and erx01_hastoa == 0",
        "notot_notoa_roc2":
            "nchtot == 0 and nchtoa == 0 and erx04_hastot == 0 and erx05_hastot == 0 and erx04_hastoa == 0 and erx05_hastoa == 0",
        "notot_roc0":
            "erx00_hastot == 0 and erx01_hastot == 0",
        "notot_roc2":
            "erx04_hastot == 0 and erx05_hastot == 0",
        "notot_notoa_nosaturatedadc":
            "nchtot == 0 and nchtoa == 0 and adc_max_pedsub < 900",
        "notot_notoa_nosaturatedadc_adcsumlt50":
            "nchtot == 0 and nchtoa == 0 and adc_max_pedsub < 900 and adc_sum_pedsub < 50",
        "notot_notoa_nosaturatedadc_adcsumgt50":
            "nchtot == 0 and nchtoa == 0 and adc_max_pedsub < 900 and adc_sum_pedsub > 50",
        "notot_notoa_nosaturatedadc_adcsumltm170":
            "nchtot == 0 and nchtoa == 0 and adc_max_pedsub < 900 and adc_sum_pedsub < -170",
        "notot_notoa_nosaturatedadc_adcsumgtm170":
            "nchtot == 0 and nchtoa == 0 and adc_max_pedsub < 900 and adc_sum_pedsub > -170",
        "withtot":
            "nchtot > 0",
        "trigtime":
            "trig_time < 113 and trig_time > 107",
    }




    selections_equal = selections_equal_per_selstring[selstring]
    selections_smaller_than = selections_smaller_than_per_selstring[selstring]
    selections_greater_than = selections_greater_than_per_selstring[selstring]
    for cfg in cfgs:
        print(f"--> Evaluating module {cfg.modulename}")
        print(f"inputfolder: {cfg.inputfolder}")
        print(f"pedestalfolder_covs: {cfg.pedestalfolder_covs}")
        print(f"plotfolder_base: {cfg.plotfolder_base}")

        list_of_inferencers = []

        # Truth
        # list_of_inferencers.append(inferencers.AnalysisTruthInferencer(cfg=cfg, selections_equal=selections_equal, selections_smaller_than=selections_smaller_than, selections_greater_than=selections_greater_than))
        list_of_inferencers.append(inferencers.AnalysisTruthInferencer(cfg=cfg, selection=selections_per_selstring[selstring]))

        # Analytic k0
        # list_of_inferencers.append(inferencers.AnalyticInferencer(cfg=cfg, split_name=split_name, is_pedestal_data=cfg.is_pedestal_data, k=0, path_to_cache=f"{path_to_cache_base}/predictions{cfg.inputfoldertag}/{cfg.modulename_for_evaluation}/analytic", path_to_covs=f"{path_to_cache_pedestal_base}/covs{cfg.inputfoldertag}/{cfg.modulename_for_evaluation}", drop_constant_cm=True))

        # DNN
        # list_of_inferencers.append(inferencers.DNNTensorInferencer(cfg=cfg, split_name=split_name, path_to_cache=f"{path_to_cache_base}/predictions{cfg.inputfoldertag}/{cfg.modulename_for_evaluation}/dnn", dtype=cfg.dnn_dtype, device="cpu"))

        # truth_inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg, selections_equal=selections_equal, selections_smaller_than=selections_smaller_than, selections_greater_than=selections_greater_than)
        truth_inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=selections_per_selstring[selstring])

        covs_true = precompute_covs(inferencer=truth_inferencer)

        for inferencer in list_of_inferencers:
            print(f"    --> Using inferencer {inferencer.name}")


            plot_all_diagnostics(cfg=cfg, selstring=selstring, inferencer=inferencer, truth_inferencer=truth_inferencer, covs_true=covs_true, plotfolder_base=cfg.plotfolder_base)






def plot_all_diagnostics(cfg: classes.EvalConfig, selstring, inferencer, truth_inferencer, covs_true, plotfolder_base):

    if inferencer.name == "true":
        covs_inferencer = covs_true
    else:
        covs_inferencer = precompute_covs(inferencer=inferencer)

    eig_true = compute_modes(method=truth_inferencer.name, cov=covs_true.pred)
    eig_cms = compute_modes(method=truth_inferencer.name, cov=utils.compute_cov_streaming(truth_inferencer.cm_iter, truth_inferencer.cm_iter))

    method_subfolder = inferencer.name
    if inferencer.name == "dnn":
        train_tag = "_".join(cfg.modulenames_used_for_training)
        method_subfolder = os.path.join(inferencer.name, f"{train_tag}{cfg.inputfoldertag}", inferencer.model.get_model_string())
    plot_dir = os.path.join(plotfolder_base, selstring)

    if inferencer.name == "dnn":
        inferencer.plot_loss(plot_dir=os.path.join(plot_dir, "loss", method_subfolder))

    if "notot_notoa" in selstring:
        yrange = (-30., 100) 
        if cfg.is_pedestal:
            yrange = (-20., 20.)
    else:
        yrange = (-30., 1000.)
        if cfg.is_pedestal:
            yrange = (-20., 20.)
    zrange_cov = (-1., 1.) if cfg.standardize_std else (-4., 4.)

    plot_cov_corr_for_inferencer(cfg=cfg, split_name=selstring, inferencer=inferencer, covs=covs_inferencer, zrange_cov=zrange_cov, plot_dir=os.path.join(plot_dir, "covcorr", method_subfolder))

    # plot_vs_eventid(split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.pred_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_eventid", method_subfolder), typetag="predictions", channel="all", nbins_x=40, nbins_y=80, y_range=tuple(x for x in yrange))
    if cfg.is_pedestal:
        plot_vs_eventid(split_name=selstring, inferencer_name=inferencer.name, value_iterator=lambda: inferencer.proj_pred_iter(vecs=eig_true.eigenvectors, k=0), out_root=os.path.join(plot_dir, "per_channel_projoneig_2d_vs_eventid", method_subfolder), typetag="predictions", channel="all", nbins_x=40, nbins_y=80, y_range=tuple(x*4 for x in yrange))

    plot_vs_chidx(cfg=cfg, varname_y_template="adc_ch*_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_chidx", method_subfolder), typetag="predictions", nbins_x=cfg.nerx*cfg.nch_per_erx, nbins_y=80, y_range=yrange)
    plot_vs_chidx(cfg=cfg, varname_y_template="adcm1_ch*", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_chidx", method_subfolder), typetag="predictions", nbins_x=cfg.nerx*cfg.nch_per_erx, nbins_y=80, y_range=yrange)

    if not cfg.is_pedestal:
        plot_vs_var(cfg=cfg, varname="trig_time", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_trigtime", method_subfolder), typetag="predictions", nbins_x=80, x_range=(50, 130), nbins_y=80, y_range=yrange)
        # plot_adcsum_vs_var(varname="trig_time", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_adcsum_vs_trigtime", method_subfolder), typetag="predictions", nbins_x=80, x_range=(50, 130), nbins_y=80, y_range=(-30., 10000))
    plot_vs_var(cfg=cfg, varname="adc_ch008_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_vs_var(cfg=cfg, varname="adc_ch017_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_vs_var(cfg=cfg, varname="adc_ch019_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_vs_var(cfg=cfg, varname="adc_ch028_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_vs_var(cfg=cfg, varname="adc_ch193_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_vs_var(cfg=cfg, varname="adc_ch202_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_vs_var(cfg=cfg, varname="adc_ch204_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_vs_var(cfg=cfg, varname="adc_ch213_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)

    plot_singlecol_vs_var(varname_x="adc_ch008_pedsub", varname_y="adc_ch196_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "channel196_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_singlecol_vs_var(varname_x="adc_ch017_pedsub", varname_y="adc_ch196_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "channel196_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_singlecol_vs_var(varname_x="adc_ch019_pedsub", varname_y="adc_ch196_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "channel196_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_singlecol_vs_var(varname_x="adc_ch028_pedsub", varname_y="adc_ch196_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "channel196_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_singlecol_vs_var(varname_x="adc_ch193_pedsub", varname_y="adc_ch196_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "channel196_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_singlecol_vs_var(varname_x="adc_ch202_pedsub", varname_y="adc_ch196_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "channel196_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_singlecol_vs_var(varname_x="adc_ch204_pedsub", varname_y="adc_ch196_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "channel196_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)
    plot_singlecol_vs_var(varname_x="adc_ch213_pedsub", varname_y="adc_ch196_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "channel196_2d_vs_disconnected", method_subfolder), typetag="predictions", nbins_x=80, x_range=(-30., 0.), nbins_y=80, y_range=yrange)

    plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname="adc_ch194_pedsub", xmin=-30., xmax=+300., nbins_x=150)
    plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname="adc_ch195_pedsub", xmin=-30., xmax=+300., nbins_x=150)
    plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname="adc_ch196_pedsub", xmin=-30., xmax=+300., nbins_x=150)
    plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname="trig_time", xmin=0., xmax=+130., nbins_x=131)
    plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname="nchtot", xmin=0., xmax=+30., nbins_x=30)
    plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname="nchtoa", xmin=0., xmax=+222., nbins_x=223)
    plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname="toa_ch196", xmin=0., xmax=+1000., nbins_x=100)
    plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname="adc_sum_pedsub", xmin=-1000., xmax=+1000., nbins_x=100)
    for erx in range(cfg.nerx):
        plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname=f"erx{erx:02d}_hastot", xmin=-0.5, xmax=1.5, nbins_x=2)
        plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname=f"erx{erx:02d}_hastoa", xmin=-0.5, xmax=1.5, nbins_x=2)
        plot_1d_singlecol(inferencer_name=inferencer.name, split_name=selstring, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_1d_singlecol", method_subfolder), typetag="predictions", colname=f"cm_erx{erx:02d}_pedsub", xmin=-70., xmax=70., nbins_x=140)
    
    # plot_multicol_vs_var(varname_x=f"cm_erx{erx:02d}_pedsub", varname_y_template="adcm1_ch*", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_cm", method_subfolder), typetag="predictions", nbins_y=80, y_range=yrange)

    for erx in range(cfg.nerx):
        plot_multicol_vs_var(varname_x=f"cm_erx{erx:02d}_pedsub", varname_y_template="adc_ch*_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_cm", method_subfolder), typetag="predictions", nbins_y=80, y_range=yrange)
        plot_multicol_vs_var(varname_x=f"cm_erx{erx:02d}_pedsub", varname_y_template="adcm1_ch*", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "per_channel_2d_vs_cm", method_subfolder), typetag="predictions", nbins_y=80, y_range=yrange)
        plot_singlecol_vs_var(varname_x=f"cm_erx{erx:02d}_pedsub", varname_y="adc_ch196_pedsub", split_name=selstring, inferencer_name=inferencer.name, value_iterator=inferencer.full_df_iter, out_root=os.path.join(plot_dir, "channel196_2d_vs_cm", method_subfolder), typetag="predictions", nbins_y=80, y_range=yrange)







def plot_eigenvals_eigenvecs(cfg: classes.EvalConfig, split_name: str, inferencer, covs: CovContainer, plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)

    plot_eigenvalues(eig=compute_modes(method=inferencer.name, cov=covs.pred), output_filename=os.path.join(plot_dir, f"Eigenvalues_predictions_{inferencer.name}_{split_name}.pdf"))
    plot_topk_eigenvectors_1d(eig=compute_modes(method=inferencer.name, cov=covs.pred), output_filename=os.path.join(plot_dir, f"Eigenvectors_1d_predictions_{inferencer.name}_{split_name}.pdf"), k=4, nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx)

    if inferencer.name in ["true"]:
        return

    plot_eigenvalues(eig=compute_modes(method=inferencer.name, cov=covs.resid), output_filename=os.path.join(plot_dir, f"Eigenvalues_residuals_{inferencer.name}_{split_name}.pdf"))
    plot_topk_eigenvectors_1d(eig=compute_modes(method=inferencer.name, cov=covs.resid), output_filename=os.path.join(plot_dir, f"Eigenvectors_1d_residuals_{inferencer.name}_{split_name}.pdf"), k=4, nch_per_erx=cfg.nch_per_erx, nerx=cfg.nerx)


# Eigendecomposition
@dataclass
class CovContainer:
    pred: pd.DataFrame
    pred_with_cm: pd.DataFrame
    resid: Optional[pd.DataFrame] = None
    resid_with_cm: Optional[pd.DataFrame] = None

def precompute_covs(inferencer):
    print(f"--> Precomputing covariance matrices for inferencer '{inferencer.name}'")

    cov_pred = utils.compute_cov_streaming(inferencer.pred_iter, inferencer.pred_iter)
    cov_pred_with_cm = utils.compute_cov_streaming(inferencer.pred_with_cm_iter, inferencer.pred_with_cm_iter)

    result = CovContainer(pred=cov_pred, pred_with_cm=cov_pred_with_cm)
    if inferencer.name not in ["true"]:
        result.resid = utils.compute_cov_streaming(inferencer.resid_iter, inferencer.resid_iter)
        result.resid_with_cm = utils.compute_cov_streaming(inferencer.resid_with_cm_iter, inferencer.resid_with_cm_iter)

    return result

def f_corr_from_cov(cov) -> float:
    # Convert DataFrame to NumPy array if needed
    if isinstance(cov, pd.DataFrame):
        cov = cov.to_numpy()
    elif not isinstance(cov, np.ndarray):
        raise TypeError("cov must be a pandas.DataFrame or numpy.ndarray")

    total_sum2 = np.sum(cov ** 2)
    diag_sum2  = np.sum(np.diag(cov) ** 2)

    if total_sum2 == 0:
        return np.nan  # avoid division by zero for empty matrices

    f_corr = 1.0 - diag_sum2 / total_sum2
    return f_corr


def plot_cov_corr_for_inferencer(cfg: classes.EvalConfig, split_name: str, inferencer, covs: CovContainer, zrange_cov, plot_dir: str) -> None:

    os.makedirs(plot_dir, exist_ok=True)
    f_corr_pred = f_corr_from_cov(covs.pred)
    
    utils.plot_covariance(df=covs.pred_with_cm, nch_per_erx=cfg.nch_per_erx, title=f"Covariance (predictions: {inferencer.name}, {split_name}) - f_corr = {f_corr_pred}", xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)", zrange=zrange_cov, output_filename=os.path.join(plot_dir, f"Covariance_predictions_{inferencer.name}_{split_name}.pdf"))
    utils.plot_covariance(df=utils.corr_from_cov(covs.pred_with_cm), nch_per_erx=cfg.nch_per_erx, title=f"Correlation (predictions: {inferencer.name}, {split_name}) - f_corr = {f_corr_pred}", xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)", zrange=(-1., 1.), output_filename=os.path.join(plot_dir, f"Correlation_predictions_{inferencer.name}_{split_name}.pdf"))

    if inferencer.name in ["true"]:
        return

    f_corr_resid = f_corr_from_cov(covs.resid)
    utils.plot_covariance(df=covs.resid_with_cm, nch_per_erx=cfg.nch_per_erx, title=f"Covariance (residuals: {inferencer.name}, {split_name}) - f_corr = {f_corr_resid}", xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)", zrange=zrange_cov, output_filename=os.path.join(plot_dir, f"Covariance_residuals_{inferencer.name}_{split_name}.pdf"))
    utils.plot_covariance(df=utils.corr_from_cov(covs.resid_with_cm), nch_per_erx=cfg.nch_per_erx, title=f"Correlation (residuals: {inferencer.name}, {split_name}) - f_corr = {f_corr_resid}", xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)", zrange=(-1., 1.), output_filename=os.path.join(plot_dir, f"Correlation_residuals_{inferencer.name}_{split_name}.pdf"))



class Streaming2DHist:
    """
    Streamed 2D histogram for y vs x with marginals + x-profile.
    Keeps:
      H[xbin, ybin], x_marg (sum over y), y_marg (sum over x),
      and the x-profile (sum_y, sum_y2, count per xbin).
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, nbins_x: int = None, nbins_y: int = 80):
        if nbins_x:
            self.nxb = nbins_x
        else:
            self.nxb = utils.round_nearest(x_max-x_min)
        
        self.nyb = nbins_y
        self.x_edges = np.linspace(x_min, x_max, self.nxb + 1, dtype=np.float64)
        self.y_edges = np.linspace(y_min, y_max, self.nyb + 1, dtype=np.float64)

        self.H = np.zeros((self.nxb, self.nyb), dtype=np.int64)
        self.x_count = np.zeros(self.nxb, dtype=np.int64)
        self.x_sum   = np.zeros(self.nxb, dtype=np.float64)
        self.x_sum2  = np.zeros(self.nxb, dtype=np.float64)

    def add(self, x: np.ndarray, y: np.ndarray):
        # Bin indices
        xi = np.searchsorted(self.x_edges, x, side="right") - 1
        yi = np.searchsorted(self.y_edges, y, side="right") - 1

        valid = (xi >= 0) & (xi < self.nxb) & (yi >= 0) & (yi < self.nyb)
        if not np.any(valid):
            return

        xi = xi[valid]
        yi = yi[valid]
        yv = y[valid]

        # 2D counts
        np.add.at(self.H, (xi, yi), 1)

        # x-profile stats
        np.add.at(self.x_count, xi, 1)
        np.add.at(self.x_sum,   xi, yv)
        np.add.at(self.x_sum2,  xi, yv * yv)

    def x_profile(self):
        centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        mean = np.full(self.nxb, np.nan, dtype=np.float64)
        rms  = np.full(self.nxb, np.nan, dtype=np.float64)
        nz = self.x_count > 0
        mean[nz] = self.x_sum[nz] / self.x_count[nz]
        var = np.zeros_like(mean)
        var[nz] = (self.x_sum2[nz] / self.x_count[nz]) - mean[nz]**2
        var[var < 0] = 0.0
        rms[nz] = np.sqrt(var[nz])
        return centers, mean, rms

def plot_vs_chidx(cfg, varname_y_template: str, split_name: str, inferencer_name: str, value_iterator, out_root: str, typetag: str, nbins_x: int = 80, nbins_y: int = 80, y_range: tuple[float,float] = (-20., 20.)):
    os.makedirs(out_root, exist_ok=True)

    chidx_min = 0
    chidx_max = cfg.nerx * cfg.nch_per_erx - 1

    # 3) streamed passes
    # --- PRED ---
    hist_pred = Streaming2DHist(x_min=chidx_min, x_max=chidx_max, y_min=y_range[0], y_max=y_range[1], nbins_x=nbins_x, nbins_y=nbins_y)
    for full_df in value_iterator():
        cols_y = [c for c in full_df.columns if fnmatch(c, varname_y_template)]
        if not cols_y:
            print(f"[WARNING] No columns found that match the template {varname_y_template}. Skipping.")
            continue
        
        x = np.tile(np.arange(chidx_min, chidx_max+1), full_df[cols_y].shape[0])
        y = full_df[cols_y].to_numpy().ravel()
        hist_pred.add(x, y)
    x_ct_p, mean_p, rms_p = hist_pred.x_profile()
    print(f"Maximum avg-per-channel-adc vs. channel index: {np.nanmax(mean_p)} at {x_ct_p[np.nanargmax(mean_p)]}")

    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_pred.H, x_edges=hist_pred.x_edges, y_edges=hist_pred.y_edges,
        x_prof_centers=x_ct_p, x_prof_mean=mean_p,
        label_x=f"channel index", label_y=f"{varname_y_template}", label_profile="profile",
        output_filename=os.path.join(out_root, f"{inferencer_name}_{typetag}_{varname_y_template.replace('*', 'all').replace('?', '')}_vs_chidx_{split_name}.pdf"),
    )

def plot_vs_var(cfg, varname: str, split_name: str, inferencer_name: str, value_iterator, out_root: str, typetag: str, nbins_x: int = 80, x_range=None, nbins_y: int = 80, y_range: tuple[float,float] = (-20., 20.)):
    os.makedirs(out_root, exist_ok=True)

    if x_range:
        var_min, var_max = x_range
    else:
        first = True
        for df in value_iterator():
            vals = df[varname]
            if first:
                var_min = min(vals)
                var_max = max(vals)
            else:
                var_min = min(min(vals), var_min)
                var_max = max(max(vals), var_max)
            first = False

    # 3) streamed passes
    # --- PRED ---
    hist_pred = Streaming2DHist(x_min=var_min, x_max=var_max, y_min=y_range[0], y_max=y_range[1], nbins_x=nbins_x, nbins_y=nbins_y)
    for full_df in value_iterator():
        x = np.repeat(full_df[varname].to_numpy(), cfg.nerx * cfg.nch_per_erx)
        y = full_df[[f"adc_ch{ch:03d}_pedsub" for ch in range(cfg.nerx * cfg.nch_per_erx)]].to_numpy().ravel()
        hist_pred.add(x, y)
    x_ct_p, mean_p, rms_p = hist_pred.x_profile()
    print(f"Maximum avg-per-channel-adc vs. {varname}: {np.nanmax(mean_p)} at {x_ct_p[np.nanargmax(mean_p)]}")


    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_pred.H, x_edges=hist_pred.x_edges, y_edges=hist_pred.y_edges,
        x_prof_centers=x_ct_p, x_prof_mean=mean_p,
        label_x=varname, label_y=f"{inferencer_name} {typetag} (ADC)", label_profile="profile",
        output_filename=os.path.join(out_root, f"{inferencer_name}_{typetag}_perchanneladc_vs_{varname}_{split_name}.pdf"),
    )

def plot_singlecol_vs_var(varname_x: str, varname_y: str, split_name: str, inferencer_name: str, value_iterator, out_root: str, typetag: str, nbins_x: int = None, x_range=None, nbins_y: int = 80, y_range: tuple[float,float] = (-20., 20.)):
    os.makedirs(out_root, exist_ok=True)

    if x_range:
        var_min, var_max = x_range
    else:
        first = True
        for df in value_iterator():
            vals = df[varname_x]
            if first:
                var_min = min(vals)
                var_max = max(vals)
            else:
                var_min = min(min(vals), var_min)
                var_max = max(max(vals), var_max)
            first = False

    # 3) streamed passes
    # --- PRED ---
    hist_pred = Streaming2DHist(x_min=var_min, x_max=var_max, y_min=y_range[0], y_max=y_range[1], nbins_x=nbins_x, nbins_y=nbins_y)
    for full_df in value_iterator():
        x = full_df[varname_x].to_numpy()
        y = full_df[varname_y].to_numpy()
        hist_pred.add(x, y)
    x_ct_p, mean_p, rms_p = hist_pred.x_profile()
    print(f"Maximum of {varname_y} vs. {varname_x}: {np.nanmax(mean_p)} at {x_ct_p[np.nanargmax(mean_p)]}")


    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_pred.H, x_edges=hist_pred.x_edges, y_edges=hist_pred.y_edges,
        x_prof_centers=x_ct_p, x_prof_mean=mean_p,
        label_x=varname_x, label_y=varname_y, label_profile="profile",
        output_filename=os.path.join(out_root, f"{inferencer_name}_{typetag}_{varname_y}_vs_{varname_x}_{split_name}.pdf"),
    )

def plot_multicol_vs_var(varname_x: str, varname_y_template: str, split_name: str, inferencer_name: str, value_iterator, out_root: str, typetag: str, nbins_x: int = None, x_range=None, nbins_y: int = 80, y_range: tuple[float,float] = (-20., 20.)):
    os.makedirs(out_root, exist_ok=True)

    if x_range:
        var_min, var_max = x_range
    else:
        first = True
        for df in value_iterator():
            vals = df[varname_x]
            if first:
                var_min = min(vals)
                var_max = max(vals)
            else:
                var_min = min(min(vals), var_min)
                var_max = max(max(vals), var_max)
            first = False

    # 3) streamed passes
    # --- PRED ---
    hist_pred = Streaming2DHist(x_min=var_min, x_max=var_max, y_min=y_range[0], y_max=y_range[1], nbins_x=nbins_x, nbins_y=nbins_y)
    for full_df in value_iterator():
        cols_y = [c for c in full_df.columns if fnmatch(c, varname_y_template)]
        if not cols_y:
            print(f"[WARNING] No columns found that match the template {varname_y_template}. Skipping.")
            continue
        x = full_df[varname_x].to_numpy()
        y = full_df[cols_y].to_numpy()
        x_rep = np.repeat(x, len(cols_y))      # (N*M,)
        y_flat = y.ravel()                     # (N*M,)

        hist_pred.add(x_rep, y_flat)
    x_ct_p, mean_p, rms_p = hist_pred.x_profile()
    print(f"Maximum of {varname_y_template} vs. {varname_x}: {np.nanmax(mean_p)} at {x_ct_p[np.nanargmax(mean_p)]}")


    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_pred.H, x_edges=hist_pred.x_edges, y_edges=hist_pred.y_edges,
        x_prof_centers=x_ct_p, x_prof_mean=mean_p,
        label_x=varname_x, label_y=varname_y_template, label_profile="profile",
        output_filename=os.path.join(out_root, f"{inferencer_name}_{typetag}_{varname_y_template.replace('*', 'all').replace('?', '')}_vs_{varname_x}_{split_name}.pdf"),
    )

def plot_adcsum_vs_var(varname: str, split_name: str, inferencer_name: str, value_iterator, out_root: str, typetag: str, nbins_x: int = 80, x_range=None, nbins_y: int = 80, y_range: tuple[float,float] = (-20., 20.)):
    os.makedirs(out_root, exist_ok=True)

    if x_range:
        var_min, var_max = x_range
    else:
        first = True
        for df in value_iterator():
            vals = df[varname]
            if first:
                var_min = min(vals)
                var_max = max(vals)
            else:
                var_min = min(min(vals), var_min)
                var_max = max(max(vals), var_max)
            first = False

    # 3) streamed passes
    # --- PRED ---
    hist_pred = Streaming2DHist(x_min=var_min, x_max=var_max, y_min=y_range[0], y_max=y_range[1], nbins_x=nbins_x, nbins_y=nbins_y)
    for full_df in value_iterator():
        x = full_df[varname].to_numpy()
        y = full_df[[f"adc_sum_pedsub"]].to_numpy().ravel()
        hist_pred.add(x, y)
    x_ct_p, mean_p, rms_p = hist_pred.x_profile()
    print(f"Maximum adc sum vs. {varname}: {np.nanmax(mean_p)} at {x_ct_p[np.nanargmax(mean_p)]}")


    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_pred.H, x_edges=hist_pred.x_edges, y_edges=hist_pred.y_edges,
        x_prof_centers=x_ct_p, x_prof_mean=mean_p,
        label_x=f"trig time", label_y=f"{inferencer_name} {typetag} (ADC)", label_profile="profile",
        output_filename=os.path.join(out_root, f"{inferencer_name}_{typetag}_adcsum_vs_{varname}_{split_name}.pdf"),
    )



def plot_vs_eventid(split_name: str, inferencer_name: str, value_iterator, out_root: str, typetag: str, channel: int | str = "all", nbins_x: int = 80, nbins_y: int = 80, y_range: tuple[float,float] = (-20., 20.)):
    os.makedirs(out_root, exist_ok=True)

    first = True
    for df in value_iterator():
        index = df.index
        if first:
            eventid_min = min(index)
            eventid_max = max(index)
        else:
            eventid_min = min(min(index), eventid_min)
            eventid_max = max(max(index), eventid_max)
        first = False


    # 2) helpers
    def flatten(df: pd.DataFrame, ch: int | str) -> np.ndarray:
        if ch == "all":
            return df.to_numpy().ravel()
        return df[[f"ch_{ch:03d}"]].to_numpy().ravel()

    # 3) streamed passes
    # --- PRED ---
    hist_pred = Streaming2DHist(x_min=eventid_min, x_max=eventid_max, y_min=y_range[0], y_max=y_range[1], nbins_x=nbins_x, nbins_y=nbins_y)
    for pred_df in value_iterator():
        x = pred_df.index.to_numpy()
        if channel == "all" and len(pred_df.shape) > 1:
            x = np.repeat(x, pred_df.shape[1])
        
        if len(pred_df.shape) > 1:
            y = flatten(pred_df, channel)
        else:
            y = pred_df
        hist_pred.add(x, y)
    x_ct_p, mean_p, rms_p = hist_pred.x_profile()

    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_pred.H, x_edges=hist_pred.x_edges, y_edges=hist_pred.y_edges,
        x_prof_centers=x_ct_p, x_prof_mean=mean_p*10,
        label_x=f"event ID", label_y=f"{inferencer_name} {typetag} (ADC)", label_profile="profile x 10",
        output_filename=os.path.join(out_root, f"{inferencer_name}_{typetag}_vs_eventid_{split_name}.pdf"),
    )

    # 5) compute autocorrelation of the binned mean vs eventID
    #    (this is the binned version)
    def autocorrelation(y: np.ndarray) -> np.ndarray:
        y = y - y.mean()
        acf = np.correlate(y, y, mode="full")
        acf = acf[acf.size // 2:]        # keep non-negative lags
        acf /= acf[0]                    # normalize to 1 at lag 0
        return acf

    acf = autocorrelation(mean_p)

    # convert lag in "bins" to approximate lag in event IDs
    # one bin ~ (eventid_max - eventid_min)/nbins_x events
    bin_width = (eventid_max - eventid_min) / nbins_x
    max_lag_bins = min(200, len(acf))    # look at first ~200 bins, or all if fewer
    lags_bins = np.arange(max_lag_bins)
    lags_events = lags_bins * bin_width

    plt.figure()
    plt.plot(lags_events, acf[:max_lag_bins])
    plt.xlabel("Lag (events)")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, f"{inferencer_name}_{typetag}_vs_eventid_acf_binned_{split_name}.pdf"))
    plt.close()

    freqs = np.fft.rfftfreq(len(mean_p), d=bin_width)
    periods = 1 / freqs[1:]
    power = np.abs(np.fft.rfft(mean_p - mean_p.mean()))**2
    power_nonzero = power[1:]
    plt.figure()
    plt.plot(periods, power_nonzero)
    plt.xlabel("Period (events)")
    plt.ylabel("Power")
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, f"{inferencer_name}_{typetag}_vs_eventid_periods_{split_name}.pdf"))
    plt.close()

def plot_vs_each_cm_streamed_full(
    split_name: str,
    inferencer_name: str,
    value_iterator,
    cm_iterator,
    out_root: str,
    typetag: str,
    channel: int | str | list[int] = "all",
    nbins_y: int = 80,
    y_range: tuple[float,float] = (-20., 20.),
):
    os.makedirs(out_root, exist_ok=True)

    # 1) global x-range for each CM
    cm_ranges = {}
    skip_this_cm = {}
    first = True
    for cm_df in cm_iterator():
        for cm in cm_df.columns:
            x = cm_df[cm].to_numpy()
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
            if first or cm not in cm_ranges:
                cm_ranges[cm] = [lo, hi]
            else:
                cm_ranges[cm][0] = min(cm_ranges[cm][0], lo)
                cm_ranges[cm][1] = max(cm_ranges[cm][1], hi)
        first = False
        
    for cm, (xmin, xmax) in cm_ranges.items():
        if xmin == xmax:
            skip_this_cm[cm] = True
        else:
            skip_this_cm[cm] = False


    # 2) helpers
    def flatten(df: pd.DataFrame, ch: int | str | list[int]) -> np.ndarray:
        if ch == "all":
            return df.to_numpy().ravel()
        if isinstance(ch, list):
            return df[[f"adc_ch{x:03d}_pedsub" for x in ch]].to_numpy().ravel()
        return df[[f"adc_ch{ch:03d}_pedsub"]].to_numpy().ravel()

    hists_pred = {}
    for cm_name, (xmin, xmax) in cm_ranges.items():
        hists_pred[cm_name] = Streaming2DHist(x_min=xmin, x_max=xmax, y_min=y_range[0], y_max=y_range[1], nbins_y=nbins_y)

    # 3) streamed passes

        # pred_iter  = inferencer.pred_iter
        # --- PRED ---
    for cm_df, pred_df in zip(cm_iterator(), value_iterator()):
        for cm_name, (xmin, xmax) in cm_ranges.items():
            if skip_this_cm[cm_name]:
                print(f"--> Skipping to fill the 2D histogram for CM '{cm_name}', which seems to have a constant value of {xmin} (=0?)")
                continue
            x = cm_df[cm_name].to_numpy()
            if channel == "all" and len(pred_df.shape) > 1:
                x = np.repeat(x, pred_df.shape[1])
            elif isinstance(channel, list):
                x = np.repeat(x, len(channel))
            if len(pred_df.shape) > 1:
                y = flatten(pred_df, channel)
            else:
                y = pred_df
            hists_pred[cm_name].add(x, y)

    for cm_name, (xmin, xmax) in cm_ranges.items():
        if skip_this_cm[cm_name]:
            print(f"--> Skipping to plot CM '{cm_name}', which seems to have a constant value of {xmin} (=0?)")
            continue
        x_ct_p, mean_p, rms_p = hists_pred[cm_name].x_profile()

        subdir = os.path.join(out_root, cm_name)
        os.makedirs(subdir, exist_ok=True)
        utils.plot_y_vs_x_with_marginals_hist(
            H=hists_pred[cm_name].H, x_edges=hists_pred[cm_name].x_edges, y_edges=hists_pred[cm_name].y_edges,
            x_prof_centers=x_ct_p, x_prof_mean=mean_p,
            label_x=f"{cm_name} (ADC)", label_y=f"{inferencer_name} {typetag} (ADC)", label_profile="profile",
            output_filename=os.path.join(subdir, f"{inferencer_name}_{typetag}_vs_{cm_name}_{split_name}.pdf"),
        )


class Streaming1DHist:
    def __init__(self, x_min: float, x_max: float, nbins_x: int = None):
        if nbins_x:
            self.nxb = nbins_x
        else:
            self.nxb = utils.round_nearest(x_max-x_min)
        
        self.x_edges = np.linspace(x_min, x_max, self.nxb + 1, dtype=np.float64)

        self.H = np.zeros(self.nxb, dtype=np.int64)
        self.x_count = 0
        self.x_sum   = 0.
        self.x_sum2  = 0.

    def add(self, x: np.ndarray):

        x = x[~np.isnan(x)]
        # Bin indices
        xi = np.searchsorted(self.x_edges, x, side="right") - 1

        valid = (xi >= 0) & (xi < self.nxb)
        if not np.any(valid):
            return

        xi = xi[valid]

        # 1D counts
        np.add.at(self.H, xi, 1)

        # x-profile stats
        self.x_count += x.shape[0]
        self.x_sum += np.sum(x)
        self.x_sum2 += np.sum(x*x)

    def x_mean_rms(self):
        if self.x_count == 0:
            return np.nan, np.nan
        mean = self.x_sum / self.x_count
        rms = np.sqrt(self.x_sum2 / self.x_count)
        return mean, rms

def plot_1d(split_name: str, inferencer_name: str, value_iterator, out_root: str, typetag: str, channel: int | str = "all", xmin: float = -20., xmax: float = +20., nbins_x: int = 40):
    os.makedirs(out_root, exist_ok=True)

    # # 2) helpers
    # def flatten(df: pd.DataFrame, ch: int | str) -> np.ndarray:
    #     if ch == "all":
    #         return df.to_numpy().ravel()
    #     return df[[f"ch_{ch:03d}"]].to_numpy().ravel()
        
    hist_pred_means = Streaming1DHist(x_min=xmin, x_max=xmax, nbins_x=nbins_x)
    hist_pred_rms   = Streaming1DHist(x_min=0., x_max=2*np.abs(xmax/5), nbins_x=nbins_x)
    for pred_df in value_iterator():
        # if len(pred_df.shape) > 1:
        #     x = flatten(pred_df, channel)
        # else:
        #     x = pred_df
        # hist_pred.add(x)
        channel_means = pred_df.mean(axis=0).to_numpy()
        channel_rms   = np.sqrt((pred_df**2).mean(axis=0))
        hist_pred_means.add(channel_means)
        hist_pred_rms.add(channel_rms)
    utils.plot_hist_single_precomputed(x=hist_pred_means.H, mean=hist_pred_means.x_mean_rms()[0], rms=hist_pred_means.x_mean_rms()[1], bins=hist_pred_means.x_edges, xlabel=f"per-channel mean: {inferencer_name} {typetag} (ADC)", title="", color="gray", outpath=os.path.join(out_root, f"per_channel_mean_{inferencer_name}_{typetag}_{split_name}.pdf"), show_mean_line=True)
    utils.plot_hist_single_precomputed(x=hist_pred_rms.H, mean=hist_pred_rms.x_mean_rms()[0], rms=hist_pred_rms.x_mean_rms()[1], bins=hist_pred_rms.x_edges, xlabel=f"per-channel mean: {inferencer_name} {typetag} (ADC)", title="", color="gray", outpath=os.path.join(out_root, f"per_channel_rms_{inferencer_name}_{typetag}_{split_name}.pdf"), show_mean_line=True)

def plot_1d_singlecol(inferencer_name: str, split_name: str, value_iterator, out_root: str, typetag: str, colname: str, xmin: float = -20., xmax: float = +20., nbins_x: int = 40):
    os.makedirs(out_root, exist_ok=True)
        
    hist = Streaming1DHist(x_min=xmin, x_max=xmax, nbins_x=nbins_x)
    for df in value_iterator():
        hist.add(df[colname].to_numpy())
    utils.plot_hist_single_precomputed(x=hist.H, mean=hist.x_mean_rms()[0], rms=hist.x_mean_rms()[1], bins=hist.x_edges, xlabel=f"{colname}", ylabel="Number of events", title="", color="gray", outpath=os.path.join(out_root, f"{colname}_1d_{inferencer_name}_{typetag}_{split_name}.pdf"), show_mean_line=True)


# # ---------- Coherent / Incoherent noise: computations ----------
@dataclass
class CoherentNoiseResult:
    method: str
    trunc_frac: float
    erx_idx: np.ndarray
    coh_true: np.ndarray
    inc_true: np.ndarray
    coh_corr: np.ndarray
    inc_corr: np.ndarray
    coh_ratio: np.ndarray
    inc_ratio: np.ndarray
    coh_over_inc_true: np.ndarray
    coh_over_inc_corr: np.ndarray


# # ---------- Coherent / Incoherent noise: plotting only ----------

def plot_coherent_noise_from_result(split_name: str, result: CoherentNoiseResult, plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax1); axc = fig.add_subplot(gs[2], sharex=ax1)

    ax1.plot(result.erx_idx, result.inc_true, "o-",  label="incoherent (raw)",   color="tab:blue")
    ax1.plot(result.erx_idx, result.coh_true, "s-",  label="coherent (raw)",     color="tab:orange")
    ax1.plot(result.erx_idx, result.inc_corr, "o--", label="incoherent (corr.)", color="tab:blue")
    ax1.plot(result.erx_idx, result.coh_corr, "s--", label="coherent (corr.)",   color="tab:orange")

    for ax in (ax1, axr, axc):
        ax.tick_params(axis="both", direction="in", top=True, bottom=True, left=True, right=True, labelsize=12)
        ax.grid(ls="--", alpha=0.3)

    ax1.set_ylabel("Noise (ADC)", fontsize=16, loc="top", labelpad=12)
    ax1.set_ylim(0., ax1.get_ylim()[1]*1.2)
    ax1.legend(loc="upper right", fontsize=12)

    axr.plot(result.erx_idx, result.inc_ratio, "o--", color="tab:blue")
    axr.plot(result.erx_idx, result.coh_ratio, "s--", color="tab:orange")
    axr.set_ylabel("corr./raw", fontsize=14, loc="top", labelpad=10)
    axr.set_ylim(0., 1.1)

    axc.plot(result.erx_idx, result.coh_over_inc_true, "D-",  color="black")
    axc.plot(result.erx_idx, result.coh_over_inc_corr, "D--", color="black")
    axc.set_xlabel("e-Rx", fontsize=16, loc="right", labelpad=8)
    axc.set_ylabel("coh/inc", fontsize=14, loc="top", labelpad=8)
    axc.set_ylim(0., max(axc.get_ylim()[1], 2.))

    plt.setp(ax1.get_xticklabels(), visible=False); plt.setp(axr.get_xticklabels(), visible=False)
    plt.tight_layout()

    frac_tag = f"{int(round(result.trunc_frac * 100))}"
    outname = f"noise_fractions_with_ratio_{result.method}_{split_name}_trunc-{frac_tag}.pdf"
    fig.savefig(os.path.join(plot_dir, outname), bbox_inches="tight", pad_inches=0.05)
    plt.close()

def infer_erx_groups_from_first_batch(cfg: classes.EvalConfig, split_name: str, nch_per_erx: int) -> dict[int, list[str]]:
    """Decide ERx→columns once from the first batch (cheap and stable)."""
    first_batch = next(iter(classes.BatchIter(cfg, split_name)))
    cols = list(first_batch["measurements_df"].columns)

    # reuse your logic
    ch_nums = np.array([int(c[3:]) if c.startswith("ch_") else int(c) for c in cols], dtype=int)
    erx_ids = ch_nums // nch_per_erx

    groups: dict[int, list[str]] = {}
    for erx in np.unique(erx_ids):
        erx_cols = [col for col, e in zip(cols, erx_ids) if e == erx]
        if len(erx_cols) >= 2:
            groups[erx] = erx_cols
    return dict(sorted(groups.items()))

def compute_coherent_noise_streamed_for_inferencer(
    cfg: classes.EvalConfig,
    split_name: str,
    inferencer,
    truth_inferencer,
    trunc_frac: float = 1.0,
) -> CoherentNoiseResult:
    """
    Streaming version: accumulates per-ERx direct/alternating sums for
    (a) true and (b) corrected (i.e. residuals after method), then applies
    your coh/inc formulas with truncated RMS.
    """

    # 1) ERx groups from first batch
    groups = infer_erx_groups_from_first_batch(cfg, split_name, cfg.nch_per_erx)
    if not groups:
        raise RuntimeError(f"[coh/inc] No ERx with >=2 channels (nch_per_erx={cfg.nch_per_erx}).")

    erx_ids_sorted = np.array(sorted(groups.keys()), dtype=int)

    # 2) accumulators: lists of vectors per ERx (to apply truncated RMS later)
    true_dir: dict[int, list[np.ndarray]] = {erx: [] for erx in erx_ids_sorted}
    true_alt: dict[int, list[np.ndarray]] = {erx: [] for erx in erx_ids_sorted}
    corr_dir: dict[int, list[np.ndarray]] = {erx: [] for erx in erx_ids_sorted}
    corr_alt: dict[int, list[np.ndarray]] = {erx: [] for erx in erx_ids_sorted}

    # 3) stream once over the split for TRUE and RESIDUALS in lockstep
    for meas_df, res_df in zip(truth_inferencer.pred_iter(), inferencer.resid_iter()):
        for erx in erx_ids_sorted:
            cols = groups[erx]

            # E_b × ncols arrays
            true_2d = meas_df.loc[:, cols].to_numpy()
            corr_2d = res_df.loc[:,  cols].to_numpy()

            # per-event sums (length E_b)
            d_true = true_2d.sum(axis=1)
            a_true = true_2d[:, ::2].sum(axis=1) - true_2d[:, 1::2].sum(axis=1)

            d_corr = corr_2d.sum(axis=1)
            a_corr = corr_2d[:, ::2].sum(axis=1) - corr_2d[:, 1::2].sum(axis=1)

            true_dir[erx].append(d_true); true_alt[erx].append(a_true)
            corr_dir[erx].append(d_corr); corr_alt[erx].append(a_corr)

    # 4) finalize per-ERx RMS and derive coherent/incoherent components
    coh_true, inc_true, coh_corr, inc_corr = [], [], [], []
    for erx in erx_ids_sorted:
        ncols = len(groups[erx])

        d_t = np.concatenate(true_dir[erx], axis=0)
        a_t = np.concatenate(true_alt[erx], axis=0)
        d_c = np.concatenate(corr_dir[erx], axis=0)
        a_c = np.concatenate(corr_alt[erx], axis=0)

        # your existing formula (uses truncated_rms)
        rms_dir_t = utils.truncated_rms(d_t, trunc_frac)
        rms_alt_t = utils.truncated_rms(a_t, trunc_frac)
        delta_t = rms_dir_t**2 - rms_alt_t**2
        inc_t = rms_alt_t / np.sqrt(ncols)
        coh_t = np.sign(delta_t) * np.sqrt(abs(delta_t)) / ncols

        rms_dir_c = utils.truncated_rms(d_c, trunc_frac)
        rms_alt_c = utils.truncated_rms(a_c, trunc_frac)
        delta_c = rms_dir_c**2 - rms_alt_c**2
        inc_c = rms_alt_c / np.sqrt(ncols)
        coh_c = np.sign(delta_c) * np.sqrt(abs(delta_c)) / ncols

        coh_true.append(coh_t); inc_true.append(inc_t)
        coh_corr.append(coh_c); inc_corr.append(inc_c)

    coh_true = np.asarray(coh_true); inc_true = np.asarray(inc_true)
    coh_corr = np.asarray(coh_corr); inc_corr = np.asarray(inc_corr)

    with np.errstate(divide="ignore", invalid="ignore"):
        inc_ratio = np.nan_to_num(inc_corr / inc_true, nan=0.0)
        coh_ratio = np.nan_to_num(coh_corr / coh_true, nan=0.0)
        coh_over_inc_true = np.nan_to_num(coh_true / inc_true, nan=0.0)
        coh_over_inc_corr = np.nan_to_num(coh_corr / inc_corr, nan=0.0)

    return CoherentNoiseResult(
        method=inferencer.name,
        trunc_frac=trunc_frac,
        erx_idx=erx_ids_sorted,
        coh_true=coh_true,
        inc_true=inc_true,
        coh_corr=coh_corr,
        inc_corr=inc_corr,
        coh_ratio=coh_ratio,
        inc_ratio=inc_ratio,
        coh_over_inc_true=coh_over_inc_true,
        coh_over_inc_corr=coh_over_inc_corr,
    )

def compute_and_plot_coh_inc_streamed(
    cfg: classes.EvalConfig,
    split_name: str,
    inferencer,
    truth_inferencer,
    plot_dir: str,
    trunc_fracs: tuple[float, ...] = (1.0, 0.95, 0.90),
):
    os.makedirs(plot_dir, exist_ok=True)

    for f in trunc_fracs:
        res = compute_coherent_noise_streamed_for_inferencer(
            cfg=cfg, split_name=split_name, inferencer=inferencer, truth_inferencer=truth_inferencer, trunc_frac=f
        )
        plot_coherent_noise_from_result(split_name=split_name, result=res, plot_dir=plot_dir)

# Eigendecomposition
@dataclass
class EigDecompResult:
    method: str
    eigenvalues: np.ndarray         # shape (N,)
    eigenvectors: np.ndarray        # shape (N, N) columns = eigenvectors, descending λ


def compute_modes(method: str, cov):
    vals, vecs = utils.compute_eig_from_cov(C=cov)  # descending
    result = EigDecompResult(method=method, eigenvalues=vals, eigenvectors=vecs)
    return result


def plot_eigenvalues(eig: EigDecompResult, output_filename: str) -> None:
    x = np.arange(1, eig.eigenvalues.size + 1)

    # log-y (clip at tiny positive to avoid -inf)
    vals = np.clip(eig.eigenvalues, 1e-12, None)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(x, vals, marker='o', lw=1)
    ax.set_xlabel("mode index", loc="right")
    ax.set_ylabel("eigenvalue", loc="top")
    ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_filename)
    print(f"--> Plotted eigenvalues: {output_filename}")
    plt.close(fig)

def plot_topk_eigenvectors_1d(eig: EigDecompResult, output_filename: str, k: int, nch_per_erx: int, nerx: int) -> None:
    k = int(min(k, eig.eigenvectors.shape[1]))

    # (a) line plot across channel index
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    for i in range(k):
        v = eig.eigenvectors[:, i]
        lam = eig.eigenvalues[i]
        plt.plot(np.arange(v.size), v, label=f'Mode {i+1} ($\lambda$={lam:.3g})')
    
    for pos in range(0, nch_per_erx*(nerx+1), nch_per_erx):
        plt.axvline(pos, color='black', linestyle='--', linewidth=1)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylim((-0.3, 0.3))
    plt.xlabel('Channel')
    plt.ylabel('Eigenvector component')
    plt.legend(ncol=2, fontsize='small')
    fig.tight_layout()
    fig.savefig(output_filename)
    print(f"--> Plotted top-{k} eigenvectors in 1d: {output_filename}")
    plt.close(fig)



if __name__ == "__main__":
    main()