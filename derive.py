#!/usr/bin/env python3
import argparse

import classes
import inferencers

import calculate_means_stds
import convert_to_df
import compute_covariances_and_eigen
import compute_predictor_analytic
import add_vars_and_selections
import prepare_dnn_inputs
import train_dnn
import submit_train


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Derive HGCal CM correction artifacts. Edit the setup block in derive.py "
            "for modules/runs/selections, then choose which workflow steps to run."
        )
    )
    parser.add_argument("-p", "--pedestals", action="store_true", help="Calculate pedestal means/stds.")
    parser.add_argument("-c", "--convert", action="store_true", help="Convert ROOT inputs to parquet analysis inputs.")
    parser.add_argument("-s", "--selections", action="store_true", help="Add variables and event selections.")
    parser.add_argument("-a", "--analytic", action="store_true", help="Compute covariance/eigen artifacts and analytic predictor.")
    parser.add_argument("-d", "--localdnn", action="store_true", help="Prepare DNN inputs, refresh split selections, and train one DNN locally.")
    parser.add_argument("-q", "--submitdnn", action="store_true", help="Prepare DNN inputs, refresh split selections, and submit DNN Condor jobs.")
    parser.add_argument("--all", action="store_true", help="Run all derivation steps, using Condor submission for DNN training.")
    parser.add_argument("--show", action="store_true", help="Print the configured setup and available steps, then exit.")
    return parser


def any_step_requested(args) -> bool:
    return any(
        [
            args.pedestals,
            args.convert,
            args.selections,
            args.analytic,
            args.localdnn,
            args.submitdnn,
            args.all,
        ]
    )


def print_setup(parser, modulenames, correction_run, pedestal_run, selection_for_correction, per_channel_cols) -> None:
    print("derive.py setup:")
    print(f"  modules: {modulenames}")
    print(f"  correction_run: {correction_run}")
    print(f"  pedestal_run: {pedestal_run}")
    print(f"  selection_for_correction: {selection_for_correction}")
    print(f"  per_channel_cols: {per_channel_cols}")
    print("")
    print(parser.format_help().rstrip())


def make_pedestal_cfg(modulename, pedestal_run):
    return classes.AnalysisConfig(
        modulename=modulename,
        run=pedestal_run,
        derive_correction=True,
        run_for_pedestal=pedestal_run,
        run_for_correction=pedestal_run,
        module_for_correction=modulename,
        standardize_std=False,
        inputfoldertag="",
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Setup: edit these values when changing modules/runs/selections.
    modulenames = ["ML_F3WC_IH0182"]
    # modulenames = ["ML_F3WC_IH0180", "ML_F3WC_IH0190", "ML_F3WC_IH0191", "ML_F3WC_IH0192", "ML_F3WC_IH0194", "ML_F3WC_IH0196", "ML_F3WC_IH0197", "ML_F3WC_IH0198"]

    # Events used to derive the correction artifacts.
    selection_for_correction = "selection_trigtime"
    correction_run = "112044_112050_112060_112073_adcmax10"
    pedestal_run = 112044

    per_channel_cols = ["channel_indices", "erx_indices", "cell_area_fraction"] + [f"adc_unconnected_{i:02d}" for i in range(4)]

    if args.show or not any_step_requested(args):
        print_setup(
            parser=parser,
            modulenames=modulenames,
            correction_run=correction_run,
            pedestal_run=pedestal_run,
            selection_for_correction=selection_for_correction,
            per_channel_cols=per_channel_cols,
        )
        return

    cfgs = [
        classes.AnalysisConfig(
            modulename=x,
            run=correction_run,
            derive_correction=True,
            selection_for_correction=selection_for_correction,
            run_for_pedestal=pedestal_run,
            run_for_correction=correction_run,
            module_for_correction=x,
            standardize_std=False,
            inputfoldertag="",
        )
        for x in modulenames
    ]

    for cfg in cfgs:
        if args.pedestals or args.all:
            pedestal_cfg = make_pedestal_cfg(modulename=cfg.modulename, pedestal_run=pedestal_run)
            calculate_means_stds.calculate_means_stds(cfg=pedestal_cfg, print_vals=True)

        if args.convert or args.all:
            if isinstance(cfg.run, int):
                convert_to_df.convert_to_df(cfg=cfg, adcmax=cfg.adcmax)
            else:
                convert_to_df.convert_to_df_synthetic(cfg=cfg, adcmax=cfg.adcmax)

        inferencer = None
        if args.selections or args.localdnn or args.submitdnn or args.all:
            inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)

        if args.selections or args.all:
            add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer)

        inferencer_sel = None
        if args.analytic or args.localdnn or args.submitdnn or args.all:
            inferencer_sel = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=selection_for_correction)

        if args.analytic or args.all:
            compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer_sel, column_tag="")
            compute_predictor_analytic.compute_predictor_analytic(cfg=cfg)

        if args.localdnn or args.submitdnn or args.all:
            prepare_dnn_inputs.prepare_dnn_inputs(cfg=cfg, column_tag="", inferencer=inferencer_sel, nch_to_use=None)
            add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer, split_selections_only=True)

        if args.localdnn:
            # Baseline alternative:
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=per_channel_cols, nodes=[256, 256, 256, 32], dropout=0.00, tag="", batch_samples=1024, epochs=500, preprocess_inputs=False)

            # current SOTA
            train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=per_channel_cols, nodes=[256, 256, 256, 32], dropout=0.00, tag="chunkshuffle_modulesummaries_targetspreproc", batch_samples=1024, epochs=500, shuffle_mode="buffered_chunk_events", shuffle_buffer_chunks=10, preprocess_inputs=True)

    if args.submitdnn or args.all:
        submit_train.submit_train(
            modules_list=[[x] for x in modulenames],
            run=correction_run,
            pedestal_run=pedestal_run,
            selection_for_correction=selection_for_correction,
            per_channel_cols=per_channel_cols,
            nodes_choices=[[256, 256, 256, 32]],
            dropout_choices=[0.0],
            epoch_choices=[500],
            weight_decay_choices=[0.0],
            modeltag="chunkshuffle_modulesummaries_targetspreproc",
            preprocess_inputs=True,
            batch_samples=1024,
            shuffle_mode="buffered_chunk_events",
            shuffle_buffer_samples=1024 * 400,
            shuffle_buffer_chunks=10,
            exclude_unconnected_targets=False,
            sample_weighting="none",
            submit_jobs=True,
        )


if __name__ == "__main__":
    main()
