#! /eos/user/a/areimers/torch-env/bin/python

import argparse
import os
from typing import Optional, Tuple
import mplhep as mh

import classes
import inferencers
import functions_plot



def _default_per_channel_adc_range(cfg, selection) -> Tuple[float, float]:
    if "notot_notoa" in selection:
        yrange = (-30.0, 100.0)
        if cfg.is_pedestal:
            if isinstance(cfg.run, int):
                yrange = (-20.0, 20.0)
            else:
                yrange = (-30.0, 200.0)
    else:
        yrange = (-100.0, 200.0)
        if cfg.is_pedestal:
            if isinstance(cfg.run, int):
                yrange = (-20.0, 20.0)
            else:
                yrange = (-30.0, 200.0)
        if cfg.run == "112073_outer":
            yrange = (-20.0, 50.0)
        if cfg.run == "112060_outer":
            yrange = (-20.0, 50.0)
        if cfg.run == "112050_adcmax10":
            yrange = (-20.0, 20.0)
        if cfg.run == "112044_112050_112060_112073_adcmax10":
            yrange = (-20.0, 20.0)
    return yrange


def main():
    parser = argparse.ArgumentParser(description="Make summary comparison plots.")
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
        "--module-for-correction",
        type=str,
        required=True,
        help="Module whose correction context should be plotted.",
    )
    parser.add_argument(
        "-s",
        "--selection",
        type=str,
        default="selection_full",
        metavar="SEL",
        help="Selection flag column to apply before plotting.",
    )
    parser.add_argument(
        "-c",
        "--column-tags",
        nargs="+",
        default=["_resid_analytic_k0", "_resid_dnn"],
        help="Correction column tags to compare. The uncorrected baseline is always added automatically.",
    )
    parser.add_argument(
        "--selection-for-correction",
        type=str,
        default="",
        help="Optional selection tag encoded in the correction-artifact folder.",
    )
    args = parser.parse_args()

    cfgs = [
        classes.AnalysisConfig(
            modulename=x,
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            run_for_correction=args.run,
            module_for_correction=args.module_for_correction,
            selection_for_correction=args.selection_for_correction,
            standardize_std=False,
            inputfoldertag="",
        )
        for x in args.modules
    ]

    for cfg in cfgs:
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=args.selection)
        plot_summaries(
            cfg=cfg,
            inferencer=inferencer,
            selection=args.selection,
            column_tags=args.column_tags
        )


def plot_summaries(
    cfg,
    inferencer,
    selection,
    column_tags,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    cm_x_range: Tuple[float, float] = (-50.0, 25.0),
    cm_profile_y_range: Optional[Tuple[float, float]] = None,
    cm_adc_range: Optional[Tuple[float, float]] = None,
) -> None:
    print("Hello from plot_summaries()!")
    plot_dir = os.path.join(cfg.plotfolder_base, selection)
    os.makedirs(plot_dir, exist_ok=True)
    if cm_adc_range is None:
        cm_adc_range = _default_per_channel_adc_range(cfg=cfg, selection=selection)


    functions_plot.plot_noise_vs_cell_area_window_rms(
        cfg=cfg,
        value_iterator=inferencer.full_df_iter,
        column_tags=column_tags,
        out_root=os.path.join(plot_dir, "summaries_compare", "noise_vs_cell_area"),
        x_range=x_range,
        y_range=y_range,
    )
    functions_plot.plot_mean_vs_cell_area_window_mean(
        cfg=cfg,
        value_iterator=inferencer.full_df_iter,
        column_tags=column_tags,
        out_root=os.path.join(plot_dir, "summaries_compare", "noise_vs_cell_area"),
        x_range=x_range,
        y_range=None,
    )
    for erx in range(cfg.nerx):
        functions_plot.plot_adc_profile_vs_cm_overlay(
            cfg=cfg,
            value_iterator=inferencer.full_df_iter,
            column_tags=column_tags,
            erx=erx,
            out_root=os.path.join(plot_dir, "summaries_compare", "per_erx_adc_profile_vs_cm"),
            x_range=cm_x_range,
            nbins_y=75,
            y_range=cm_adc_range,
            profile_y_range=cm_profile_y_range,
        )

    print("Successfully finished summary plotting!")


if __name__ == "__main__":
    main()
