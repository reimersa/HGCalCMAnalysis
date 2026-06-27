#!/usr/bin/env python3
import argparse
import os

import pandas as pd  # type: ignore

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


def main(args, parser):
    # Setup: edit these values when changing modules/runs/selections.
    modulenames = ["ML_F3WC_IH0182"]

    # Selection used only for plots/evaluation on the target run.
    selection = "selection_trigtime"

    # Selection that was used when the stored correction was derived.
    selection_for_correction = "selection_trigtime"

    # Target run to which already-derived corrections are applied.
    # target_run = 112044
    # target_run = 112048
    # target_run = 1120480000
    # target_run = 11204800001
    target_run = 112049
    # target_run = 112050
    # target_run = 112051
    # target_run = 112060
    # target_run = 112068
    # target_run = 112078
    # target_run = "112044_112050_112060_112073_adcmax5"
    # target_run = "112044_112050_112060_112073_adcmax10"
    # target_run = "112046_112047_112048_112049_112050_adcmax10"
    # target_run = "112050_adcmax10"
    # target_run = "112051_adcmax10"
    # target_run = "112060_adcmax10"
    # target_run = "112068_adcmax10"

    pedestal_run = 112044
    # correction_run = "112044_112050_112060_112073_adcmax5"
    correction_run = "112044_112050_112060_112073_adcmax10"
    # correction_run = "112050_112060_112073_adcmax10"
    # correction_run = "112046_112047_112048_112049_112050_adcmax10"
    # correction_run = "112050_adcmax10"
    module_for_correction = "ML_F3WC_IH0182"

    n_coherent_noise_model = 3
    per_channel_cols = ["channel_indices", "erx_indices", "cell_area_fraction"] + [f"adc_unconnected_{i:02d}" for i in range(4)]

    dnn_nodes = [256, 256, 256, 32]
    dnn_dropout = 0.0
    dnn_infer_batch = 8192

    # Baseline:
    # dnn_tag = ""
    # dnn_preprocess_inputs = False

    # Current default.
    dnn_tag = "chunkshuffle_modulesummaries_targetspreproc"
    dnn_preprocess_inputs = True

    dnn_resolved_tag = add_correction_dnn.tag_with_input_preprocessing(dnn_tag, dnn_preprocess_inputs)
    dnn_output_tag = add_correction_dnn.dnn_output_tag_from_model_tag(dnn_resolved_tag)

    method_column_tags = {
        "uncorrected": "",
        "analytic": "_resid_analytic_k0",
        "dnn": f"_resid{dnn_output_tag}",
    }

    if args.show or not any_step_requested(args):
        print_setup(
            parser=parser,
            modulenames=modulenames,
            target_run=target_run,
            correction_run=correction_run,
            pedestal_run=pedestal_run,
            module_for_correction=module_for_correction,
            selection=selection,
            selection_for_correction=selection_for_correction,
            n_coherent_noise_model=n_coherent_noise_model,
            per_channel_cols=per_channel_cols,
            dnn_tag=dnn_tag,
            dnn_preprocess_inputs=dnn_preprocess_inputs,
            dnn_resolved_tag=dnn_resolved_tag,
            dnn_output_tag=dnn_output_tag,
            include_help=True,
        )
        return

    print_setup(
        parser=parser,
        modulenames=modulenames,
        target_run=target_run,
        correction_run=correction_run,
        pedestal_run=pedestal_run,
        module_for_correction=module_for_correction,
        selection=selection,
        selection_for_correction=selection_for_correction,
        n_coherent_noise_model=n_coherent_noise_model,
        per_channel_cols=per_channel_cols,
        dnn_tag=dnn_tag,
        dnn_preprocess_inputs=dnn_preprocess_inputs,
        dnn_resolved_tag=dnn_resolved_tag,
        dnn_output_tag=dnn_output_tag,
    )
    print("")

    methods_to_run = selected_methods(args)
    cfgs = make_cfgs(
        modulenames=modulenames,
        target_run=target_run,
        correction_run=correction_run,
        pedestal_run=pedestal_run,
        module_for_correction=module_for_correction,
        selection_for_correction=selection_for_correction,
    )

    for cfg in cfgs:
        if args.convert or args.all:
            if isinstance(cfg.run, int):
                convert_to_df.convert_to_df(cfg=cfg, adcmax=cfg.adcmax)
            else:
                convert_to_df.convert_to_df_synthetic(cfg=cfg, adcmax=cfg.adcmax)

        if args.selections or args.all:
            inferencer = make_full_inferencer(cfg)
            add_vars_and_selections.add_vars_and_selections(cfg=cfg, inferencer=inferencer)

        if args.compute or args.all:
            require_selection_column(cfg=cfg, selection=selection)

            for method in methods_to_run:
                if method != "uncorrected":
                    require_uncorrected_projection_basis(cfg)

                inferencer = make_full_inferencer(cfg)

                if method == "analytic":
                    add_correction_analytic.add_correction_analytic(cfg=cfg, inferencer=inferencer)
                    inferencer = make_full_inferencer(cfg)
                elif method == "dnn":
                    add_correction_dnn.add_correction_dnn(
                        cfg=cfg,
                        inferencer=inferencer,
                        nodes=dnn_nodes,
                        dropout=dnn_dropout,
                        tag=dnn_tag,
                        column_tag="",
                        per_channel_cols=per_channel_cols,
                        infer_batch=dnn_infer_batch,
                        plot_dir_loss=dnn_loss_plot_folder(cfg=cfg, selection=selection, dnn_output_tag=dnn_output_tag),
                        preprocess_inputs=dnn_preprocess_inputs,
                    )
                    inferencer = make_full_inferencer(cfg)

                inferencer_sel = make_selected_inferencer(cfg=cfg, selection=selection)
                compute_residual_diagnostics(
                    cfg=cfg,
                    inferencer_sel=inferencer_sel,
                    column_tag=method_column_tags[method],
                    n_coherent_noise_model=n_coherent_noise_model,
                )
                require_uncorrected_projection_basis(cfg)
                add_projection(cfg=cfg, inferencer=inferencer, column_tag=method_column_tags[method])

        if args.plots or args.all:
            require_selection_column(cfg=cfg, selection=selection)
            inferencer_sel = make_selected_inferencer(cfg=cfg, selection=selection)
            for method in methods_to_run:
                plot.plot(
                    cfg=cfg,
                    inferencer=inferencer_sel,
                    column_tag=method_column_tags[method],
                    selection=selection,
                    n_coherent_noise_model=n_coherent_noise_model,
                )

            summary_column_tags = [method_column_tags[method] for method in methods_to_run if method != "uncorrected"]
            make_summary_plots(
                cfg=cfg,
                inferencer_sel=inferencer_sel,
                selection=selection,
                column_tags=summary_column_tags,
                n_coherent_noise_model=n_coherent_noise_model,
            )


def any_step_requested(args) -> bool:
    return any(
        [
            args.convert,
            args.selections,
            args.compute,
            args.plots,
            args.all,
        ]
    )


def print_setup(
    parser,
    modulenames,
    target_run,
    correction_run,
    pedestal_run,
    module_for_correction,
    selection,
    selection_for_correction,
    n_coherent_noise_model,
    per_channel_cols,
    dnn_tag,
    dnn_preprocess_inputs,
    dnn_resolved_tag,
    dnn_output_tag,
    include_help=False,
) -> None:
    print("apply.py setup:")
    print(f"  modules: {modulenames}")
    print(f"  target_run: {target_run}")
    print(f"  correction_run: {correction_run}")
    print(f"  pedestal_run: {pedestal_run}")
    print(f"  module_for_correction: {module_for_correction}")
    print(f"  selection: {selection}")
    print(f"  selection_for_correction: {selection_for_correction}")
    print(f"  n_coherent_noise_model: {n_coherent_noise_model}")
    print(f"  per_channel_cols: {per_channel_cols}")
    print(f"  dnn_tag: {dnn_tag}")
    print(f"  dnn_preprocess_inputs: {dnn_preprocess_inputs}")
    print(f"  dnn_resolved_tag: {dnn_resolved_tag}")
    print(f"  dnn_output_tag: {dnn_output_tag}")
    if include_help:
        print("")
        print(parser.format_help().rstrip())


def selected_methods(args) -> list[str]:
    method_order = ["uncorrected", "analytic", "dnn"]
    if args.all:
        return method_order
    requested = set(args.methods)
    return [method for method in method_order if method in requested]


def make_cfgs(
    modulenames,
    target_run,
    correction_run,
    pedestal_run,
    module_for_correction,
    selection_for_correction,
):
    return [
        classes.AnalysisConfig(
            modulename=x,
            run=target_run,
            derive_correction=False,
            selection_for_correction=selection_for_correction,
            run_for_pedestal=pedestal_run,
            run_for_correction=correction_run,
            module_for_correction=module_for_correction,
            standardize_std=False,
            inputfoldertag="",
        )
        for x in modulenames
    ]


def make_full_inferencer(cfg):
    return inferencers.AnalysisTruthInferencer(cfg=cfg)


def make_selected_inferencer(cfg, selection):
    return inferencers.AnalysisTruthInferencer(cfg=cfg, selection=selection)


def require_selection_column(cfg, selection) -> None:
    if not selection:
        return

    try:
        inputfiles = classes.AnalysisBatchIter(cfg=cfg).inputfiles
    except RuntimeError as exc:
        raise RuntimeError(
            "Cannot use a selected inferencer because analysis input parquet files are missing.\n"
            "Run: python apply.py --convert"
        ) from exc

    missing_files = []
    for inputfile in inputfiles:
        try:
            pd.read_parquet(inputfile, columns=[selection])
        except Exception:
            missing_files.append(inputfile)

    if missing_files:
        raise KeyError(
            f"Selection column '{selection}' is required but missing in {len(missing_files)} input file(s).\n"
            f"First missing file: {missing_files[0]}\n"
            "Run: python apply.py --selections"
        )


def require_uncorrected_projection_basis(cfg) -> None:
    basis_path = os.path.join(cfg.own_covs_folder, "eigenvectors_mm.parquet")
    if not os.path.exists(basis_path):
        raise FileNotFoundError(
            "Uncorrected eigenvectors are required for projection diagnostics.\n"
            f"Missing file: {basis_path}\n"
            "Run: python apply.py -m uncorrected --compute"
        )


def compute_residual_diagnostics(cfg, inferencer_sel, column_tag, n_coherent_noise_model) -> None:
    compute_covariances_and_eigen.compute_covariances_and_eigen(
        cfg=cfg,
        inferencer=inferencer_sel,
        column_tag=column_tag,
    )
    fit_covariance_noise_model.fit_covariance_noise_model(
        cfg=cfg,
        column_tag=column_tag,
        n_coherent=n_coherent_noise_model,
    )


def add_projection(cfg, inferencer, column_tag) -> None:
    add_projections_onto_noisemode.add_projections_onto_noisemode(
        cfg=cfg,
        inferencer=inferencer,
        column_tag=column_tag,
        k=0,
    )


def dnn_loss_plot_folder(cfg, selection, dnn_output_tag) -> str:
    base = os.path.join(cfg.plotfolder_base, selection, "dnn_loss")
    if dnn_output_tag == "_dnn":
        return base
    return os.path.join(base, dnn_output_tag.strip("_"))


def make_summary_plots(cfg, inferencer_sel, selection, column_tags, n_coherent_noise_model) -> None:
    if cfg.run == 112044:
        plot_summaries.plot_summaries(
            cfg=cfg,
            inferencer=inferencer_sel,
            selection=selection,
            column_tags=column_tags,
            n_coherent_noise_model=n_coherent_noise_model,
            y_range=(0.8, 1.8),
            cm_x_range=(-12.0, 12.0),
            cm_profile_y_range=(-2.0, 3.0),
        )
    else:
        plot_summaries.plot_summaries(
            cfg=cfg,
            inferencer=inferencer_sel,
            selection=selection,
            column_tags=column_tags,
            n_coherent_noise_model=n_coherent_noise_model,
            y_range=(1.0, 4.0),
            cm_x_range=(-50.0, 25.0),
            cm_profile_y_range=(-2.0, 3.0),
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Apply HGCal CM corrections and make evaluation diagnostics. Edit the setup block "
            "in apply.py for modules/runs/selections, then choose which workflow steps to run."
        )
    )
    parser.add_argument("-c", "--convert", action="store_true", help="Convert target ROOT/synthetic inputs to parquet.")
    parser.add_argument("-s", "--selections", action="store_true", help="Add variables and event selections on the target run.")
    parser.add_argument(
        "-m",
        "--methods",
        nargs="+",
        choices=["uncorrected", "analytic", "dnn"],
        default=["uncorrected", "analytic", "dnn"],
        help="Methods to process or plot.",
    )
    parser.add_argument(
        "-k",
        "--compute",
        action="store_true",
        help="Add corrections where applicable, then compute covariance/eigen, noise-model, and projection diagnostics.",
    )
    parser.add_argument("-p", "--plots", action="store_true", help="Make detailed plots and summary comparison plots.")
    parser.add_argument("--all", action="store_true", help="Run convert, selections, compute, and plots for all methods.")
    parser.add_argument("--show", action="store_true", help="Print the configured setup and available steps, then exit.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    main(args=parser.parse_args(), parser=parser)
