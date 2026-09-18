#!/usr/bin/env python3
import argparse

import classes
import inferencers

import calculate_means_stds
import convert_to_df
import compute_covariances_and_eigen
import compute_predictor_analytic
import compute_predictor_analytic_allinputs
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
    parser.add_argument(
        "-A",
        "--analytic-allinputs",
        action="store_true",
        help="Derive one analytic linear predictor per target from the prepared all-channel training inputs.",
    )
    parser.add_argument("-i", "--dnninputs", action="store_true", help="Prepare DNN inputs and refresh train/test split selections.")
    parser.add_argument(
        "--plot-dnninputs",
        action="store_true",
        help="When preparing DNN inputs, write one full-distribution histogram per DNN feature.",
    )
    parser.add_argument("-d", "--localdnn", action="store_true", help="Train one DNN locally using existing prepared DNN inputs.")
    parser.add_argument("-q", "--submitdnn", action="store_true", help="Submit DNN Condor jobs using existing prepared DNN inputs.")
    parser.add_argument("--all", action="store_true", help="Run all derivation steps, including DNN input preparation and Condor DNN submission.")
    parser.add_argument("--show", action="store_true", help="Print the configured setup and available steps, then exit.")
    return parser


def any_step_requested(args) -> bool:
    return any(
        [
            args.pedestals,
            args.convert,
            args.selections,
            args.analytic,
            args.analytic_allinputs,
            args.dnninputs,
            args.localdnn,
            args.submitdnn,
            args.all,
        ]
    )


def print_setup(parser, modulenames, correction_run, pedestal_run, selection_for_correction, per_channel_cols, dnn_feature_version, combine_modules, dnn_model_tag, dnn_preprocess_inputs, train_event_fractions, validation_event_fractions, analytic_allinputs_rcond, include_help=False) -> None:
    print("derive.py setup:")
    print(f"  modules: {modulenames}")
    print(f"  correction_run: {correction_run}")
    print(f"  pedestal_run: {pedestal_run}")
    print(f"  selection_for_correction: {selection_for_correction}")
    print(f"  per_channel_cols: {per_channel_cols}")
    print(f"  dnn_feature_version: {dnn_feature_version}")
    feature_spec = prepare_dnn_inputs.make_feature_spec(
        feature_version=dnn_feature_version,
        module_vocabulary=modulenames,
    )
    print(f"  dnn_input_schema_id: {prepare_dnn_inputs.dnn_input_schema_id(feature_spec)}")
    print(f"  combine_modules: {combine_modules}")
    print(f"  dnn_model_tag: {dnn_model_tag}")
    print(f"  dnn_preprocess_inputs: {dnn_preprocess_inputs}")
    print(f"  train_event_fractions: {train_event_fractions}")
    print(f"  validation_event_fractions: {validation_event_fractions}")
    print(f"  analytic_allinputs_rcond: {analytic_allinputs_rcond}")
    if include_help:
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
        campaign="Sep2025TB",
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Setup: edit these values when changing modules/runs/selections.
    # modulenames = ["ML_F3WC_IH0182"]
    # modulenames = ["ML_F3WC_IH0180", "ML_F3WC_IH0190", "ML_F3WC_IH0191", "ML_F3WC_IH0192", "ML_F3WC_IH0194"]
    modulenames = [
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
    ]

    # Events used to derive the correction artifacts.
    selection_for_correction = "selection_trigtime"
    correction_run = "112044_112050_112060_112073_adcmax10"
    pedestal_run = 112044
    # selection_for_correction = "selection_full"
    # correction_run = 118212
    # pedestal_run = 118212

    dnn_feature_version = prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS
    combine_modules = len(modulenames) > 1
    dnn_model_tag = "allchannels_multimodule" if combine_modules else "allchannels"
    # False disables both feature z-scoring and per-channel target z-scoring.
    dnn_preprocess_inputs = True
    per_channel_cols = ["channel_indices", "erx_indices", "cell_area_fraction"]
    feature_spec = prepare_dnn_inputs.make_feature_spec(
        feature_version=dnn_feature_version,
        module_vocabulary=modulenames,
    )
    frac = 0.5 if len(modulenames) > 5 else 1.0
    train_event_fractions = {module: frac for module in modulenames}
    validation_event_fractions = {module: frac for module in modulenames}
    analytic_allinputs_rcond = compute_predictor_analytic_allinputs.DEFAULT_RCOND

    if args.show or not any_step_requested(args):
        print_setup(
            parser=parser,
            modulenames=modulenames,
            correction_run=correction_run,
            pedestal_run=pedestal_run,
            selection_for_correction=selection_for_correction,
            per_channel_cols=per_channel_cols,
            dnn_feature_version=dnn_feature_version,
            combine_modules=combine_modules,
            dnn_model_tag=dnn_model_tag,
            dnn_preprocess_inputs=dnn_preprocess_inputs,
            train_event_fractions=train_event_fractions,
            validation_event_fractions=validation_event_fractions,
            analytic_allinputs_rcond=analytic_allinputs_rcond,
            include_help=True,
        )
        return

    print_setup(
        parser=parser,
        modulenames=modulenames,
        correction_run=correction_run,
        pedestal_run=pedestal_run,
        selection_for_correction=selection_for_correction,
        per_channel_cols=per_channel_cols,
        dnn_feature_version=dnn_feature_version,
        combine_modules=combine_modules,
        dnn_model_tag=dnn_model_tag,
        dnn_preprocess_inputs=dnn_preprocess_inputs,
        train_event_fractions=train_event_fractions,
        validation_event_fractions=validation_event_fractions,
        analytic_allinputs_rcond=analytic_allinputs_rcond,
    )
    print("")

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
            campaign="Sep2025TB",
        )
        for x in modulenames
    ]
    cfg_out = cfgs[0]
    if combine_modules:
        model_group = f"MULTI_{'_'.join(modulenames)}"
        cfg_out = classes.AnalysisConfig(
            modulename=modulenames[0],
            run=correction_run,
            run_for_pedestal=pedestal_run,
            run_for_correction=correction_run,
            module_for_correction=model_group,
            selection_for_correction=selection_for_correction,
            standardize_std=False,
            inputfoldertag="",
            campaign="Sep2025TB",
        )

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
        if args.selections or args.dnninputs or args.all:
            inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)

        if args.selections or args.all:
            add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer)

        inferencer_sel = None
        if args.analytic or args.dnninputs or args.all:
            inferencer_sel = inferencers.AnalysisTruthInferencer(cfg=cfg, selection=selection_for_correction)

        if args.analytic or args.all:
            compute_covariances_and_eigen.compute_covariances_and_eigen(cfg=cfg, inferencer=inferencer_sel, column_tag="")
            compute_predictor_analytic.compute_predictor_analytic(cfg=cfg)

        if args.dnninputs or args.all:
            prepare_dnn_inputs.prepare_dnn_inputs(
                cfg=cfg,
                column_tag="",
                inferencer=inferencer_sel,
                nch_to_use=None,
                plot_inputs=args.plot_dnninputs,
                feature_spec=feature_spec,
            )
            add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer, split_selections_only=True)

        if args.localdnn and not combine_modules:
            # Baseline alternative:
            # train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=per_channel_cols, nodes=[256, 256, 256, 32], dropout=0.00, tag="", batch_samples=1024, epochs=500, preprocess_inputs=False)

            # current SOTA
            train_dnn.train_dnn(cfg=cfg, noprogbar=False, per_channel_cols=per_channel_cols, nodes=[256, 256, 256, 32], dropout=0.00, tag=dnn_model_tag, batch_samples=1024, epochs=200, shuffle_mode="buffered_chunk_events", shuffle_buffer_chunks=10, preprocess_inputs=dnn_preprocess_inputs, feature_spec=feature_spec, train_event_fractions={cfg.modulename: train_event_fractions[cfg.modulename]}, validation_event_fractions={cfg.modulename: validation_event_fractions[cfg.modulename]})

    if args.analytic_allinputs or args.all:
        compute_predictor_analytic_allinputs.compute_predictor_analytic_allinputs(
            cfgs=cfgs,
            cfg_out=cfg_out,
            feature_spec=feature_spec,
            train_event_fractions=train_event_fractions,
            rcond=analytic_allinputs_rcond,
        )

    if args.localdnn and combine_modules:
        train_dnn.train_dnn(
            cfg=cfgs,
            cfg_out=cfg_out,
            noprogbar=False,
            per_channel_cols=per_channel_cols,
            nodes=[256, 256, 256, 32],
            dropout=0.00,
            tag=dnn_model_tag,
            batch_samples=1024,
            epochs=200,
            shuffle_mode="buffered_chunk_events",
            shuffle_buffer_chunks=10,
            preprocess_inputs=dnn_preprocess_inputs,
            feature_spec=feature_spec,
            train_event_fractions=train_event_fractions,
            validation_event_fractions=validation_event_fractions,
        )

    if args.submitdnn or args.all:
        submit_train.submit_train(
            modules_list=[modulenames] if combine_modules else [[x] for x in modulenames],
            run=correction_run,
            pedestal_run=pedestal_run,
            selection_for_correction=selection_for_correction,
            per_channel_cols=per_channel_cols,
            nodes_choices=[[256, 256, 256, 32]],
            dropout_choices=[0.0],
            epoch_choices=[200],
            weight_decay_choices=[0.0],
            modeltag=dnn_model_tag,
            preprocess_inputs=dnn_preprocess_inputs,
            batch_samples=1024,
            shuffle_mode="buffered_chunk_events",
            shuffle_buffer_samples=1024 * 400,
            shuffle_buffer_chunks=10,
            exclude_unconnected_targets=False,
            sample_weighting="none",
            submit_jobs=True,
            combine_modules=combine_modules,
            feature_version=dnn_feature_version,
            train_event_fractions=train_event_fractions,
            validation_event_fractions=validation_event_fractions,
        )


if __name__ == "__main__":
    main()
