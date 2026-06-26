#!/usr/bin/env python3
import os
import re
import shlex
import shutil
import stat
import subprocess
from itertools import product


current_dir = os.path.dirname(os.path.realpath(__file__))
train_script = os.path.join(current_dir, "train_dnn.py")
if not os.path.exists(train_script):
    raise FileNotFoundError(f"Could not find training script at {train_script}")

workdir = os.path.join(current_dir, "workdir_condor")


# Define training jobs here.
# modules_list = [["ML_F3WC_IH0182"]]
# modules_list = [["ML_F3WC_IH0180"], ["ML_F3WC_IH0190"], ["ML_F3WC_IH0191"], ["ML_F3WC_IH0192"], ["ML_F3WC_IH0194"], ["ML_F3WC_IH0196"], ["ML_F3WC_IH0197"], ["ML_F3WC_IH0198"], ["ML_F3WC_IH0199"]]
modules_list = [["ML_F3WC_IH0180"], ["ML_F3WC_IH0190"], ["ML_F3WC_IH0191"], ["ML_F3WC_IH0192"], ["ML_F3WC_IH0194"], ["ML_F3WC_IH0196"], ["ML_F3WC_IH0197"], ["ML_F3WC_IH0198"]]
run = "112044_112050_112060_112073_adcmax10"
# run = "112046_112047_112048_112049_112050_adcmax10"
pedestal_run = 112044
selection_for_correction = "selection_trigtime"

per_channel_cols = ["channel_indices", "erx_indices", "cell_area_fraction"] + [f"adc_unconnected_{i:02d}" for i in range(4)]
# per_channel_cols = ["channel_indices", "erx_indices", "cell_area_fraction", "u_centered", "v_centered"] + [f"adc_unconnected_{i:02d}" for i in range(4)] + ["same_erx_nchtoa", "same_erx_nchtot", "same_erx_nchadcgt10", "same_erx_nchadcgt50", "same_erx_nchadcgt200", "same_erx_nchadcgt500"]

nodes_choices = [[256, 256, 256, 32]]
dropout_choices = [0.0]
epoch_choices = [500]
weight_decay_choices = [0.0]

# modeltag = "globalshuffle"
# modeltag = "simpleshuffle"
# modeltag = "simpleshuffle_runchaweight"
# modeltag = "simpleshuffle_modulesummaries"
# modeltag = "simpleshuffle_modulesummaries_targetspreproc"
modeltag = "chunkshuffle_modulesummaries_targetspreproc"

preprocess_inputs = True

# batch_samples = 222 * 1024
# shuffle_mode = "global_samples"
# shuffle_buffer_samples = 222 * 1024 * 400
batch_samples = 1024
shuffle_mode = "buffered_chunk_events"
shuffle_buffer_samples = 1024 * 400
shuffle_buffer_chunks = 10

exclude_unconnected_targets = False
# sample_weighting = "source_run_channel"
sample_weighting = "none"
override_name = False
new_name = "NOTHING"


# Condor settings.
jobflavor = "tomorrow"
# jobflavor = "espresso"
request_gpus = 1
request_cpus = 1
request_memory_gb = 16
submit_jobs = True


def sanitize_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    name = name.strip("_")
    if not name:
        raise ValueError(f"Could not build a stable job name from {value!r}")
    return name


def quote_args(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def format_weight_decay_tag(weight_decay: float) -> str:
    value = f"{weight_decay:g}"
    if "e" in value:
        mantissa, exponent = value.split("e")
        exp = int(exponent)
        exp_tag = f"{'p' if exp >= 0 else 'm'}{abs(exp)}"
        return f"weightdecay{mantissa.replace('.', 'p')}e{exp_tag}"
    return f"weightdecay{value.replace('.', 'p')}"


def tag_with_weight_decay(tag: str, weight_decay: float) -> str:
    if weight_decay == 0.0:
        return tag
    if "weightdecay" in tag:
        return tag
    weight_decay_tag = format_weight_decay_tag(weight_decay)
    return f"{tag}_{weight_decay_tag}" if tag else weight_decay_tag


def tag_with_input_preprocessing(tag: str, preprocess_inputs: bool) -> str:
    if not preprocess_inputs:
        return tag
    input_preprocessing_tag = "inputzscore"
    if input_preprocessing_tag in tag.split("_"):
        return tag
    return f"{tag}_{input_preprocessing_tag}" if tag else input_preprocessing_tag


def make_job_name(
    modules: list[str],
    run,
    selection_for_correction: str,
    nodes: list[int],
    dropout: float,
    epochs: int,
    weight_decay: float,
    modeltag: str,
    preprocess_inputs: bool,
) -> str:
    effective_tag = tag_with_weight_decay(tag_with_input_preprocessing(modeltag, preprocess_inputs), weight_decay)
    parts = [
        "_".join(modules),
        str(run),
        selection_for_correction,
        effective_tag,
        "-".join(str(n) for n in nodes),
        f"dr{dropout:g}",
        f"e{epochs}",
    ]
    return sanitize_name("_".join(part for part in parts if part))


def submit_train(
    modules_list: list[list[str]],
    run,
    pedestal_run: int,
    selection_for_correction: str,
    per_channel_cols: list[str],
    nodes_choices: list[list[int]],
    dropout_choices: list[float],
    epoch_choices: list[int],
    weight_decay_choices: list[float],
    modeltag: str,
    preprocess_inputs: bool,
    batch_samples: int,
    shuffle_mode: str,
    shuffle_buffer_samples: int,
    shuffle_buffer_chunks: int,
    exclude_unconnected_targets: bool,
    sample_weighting: str,
    override_name: bool = False,
    new_name: str = "NOTHING",
    jobflavor: str = "tomorrow",
    request_gpus: int = 1,
    request_cpus: int = 1,
    request_memory_gb: int = 16,
    submit_jobs: bool = True,
) -> None:
    os.makedirs(workdir, exist_ok=True)

    proxy_filename_orig = f"/tmp/x509up_u{os.getuid()}"
    proxy_filename_forjob = os.path.join(workdir, "voms_proxy")
    if not os.path.exists(proxy_filename_orig):
        raise ValueError(f"VOMS proxy {proxy_filename_orig} does not exist, please set it up.")
    shutil.copy2(proxy_filename_orig, proxy_filename_forjob)
    print(f"Copied proxy file to {proxy_filename_forjob}.")

    wrapper_template = os.path.join(current_dir, "wrapper_train_dnn.sh.template")
    sub_template = os.path.join(current_dir, "train_dnn.sub.template")

    with open(wrapper_template) as f:
        tmpl_wrap = f.read()
    with open(sub_template) as f:
        tmpl_sub = f.read()

    for modules, nodes, dropout, epochs, weight_decay in product(
        modules_list, nodes_choices, dropout_choices, epoch_choices, weight_decay_choices
    ):
        args = []
        args += ["-m"] + modules
        args += ["--run", run]
        args += ["--pedestal-run", str(pedestal_run)]
        args += ["--selection-for-correction", selection_for_correction]
        args += ["-n"] + [str(n) for n in nodes]
        args += ["-d", str(dropout)]
        args += ["-e", str(epochs)]
        args += ["--weight-decay", str(weight_decay)]
        args += ["--batch-samples", str(batch_samples)]
        args += ["--shuffle-mode", shuffle_mode]
        args += ["--shuffle-buffer-samples", str(shuffle_buffer_samples)]
        args += ["--shuffle-buffer-chunks", str(shuffle_buffer_chunks)]
        args += ["--sample-weighting", sample_weighting]
        args += ["--per-channel-cols"] + per_channel_cols
        args += ["--noprogbar"]
        if modeltag:
            args += ["-t", modeltag]
        if preprocess_inputs:
            args += ["--preprocess-inputs"]
        if exclude_unconnected_targets:
            args += ["--exclude-unconnected-targets"]
        if override_name:
            args += ["--override-name", "--new-name", new_name]

        name = make_job_name(
            modules=modules,
            run=run,
            selection_for_correction=selection_for_correction,
            nodes=nodes,
            dropout=dropout,
            epochs=epochs,
            weight_decay=weight_decay,
            modeltag=modeltag,
            preprocess_inputs=preprocess_inputs,
        )
        logdir = os.path.join(workdir, name)
        os.makedirs(logdir, exist_ok=True)

        wrapper_thisjob = os.path.join(logdir, "wrapper.sh")
        content_wrap = tmpl_wrap.format(
            WORKDIR=current_dir,
            VENV=f"/eos/user/{os.getenv('USER')[0]}/{os.getenv('USER')}/torch-env",
            EXE=shlex.quote(train_script),
            ARGS=quote_args(args),
        )
        with open(wrapper_thisjob, "w") as f:
            f.write(content_wrap)
        os.chmod(wrapper_thisjob, os.stat(wrapper_thisjob).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        content_sub = tmpl_sub.format(
            GPUS=request_gpus,
            CPUS=request_cpus,
            LOGDIR=logdir,
            NAME=name,
            JOBFLAVOR=jobflavor,
            MEMGB=request_memory_gb,
            VOMS=proxy_filename_forjob,
            WRAPPER=wrapper_thisjob,
        )
        subfile = os.path.join(workdir, f"{name}.sub")
        with open(subfile, "w") as f:
            f.write(content_sub)

        if submit_jobs:
            subprocess.run(["condor_submit", subfile], check=True)
            print(f"Submitted job: {name}")
        else:
            print(f"Wrote job files without submitting: {name}")


def main():
    submit_train(
        modules_list=modules_list,
        run=run,
        pedestal_run=pedestal_run,
        selection_for_correction=selection_for_correction,
        per_channel_cols=per_channel_cols,
        nodes_choices=nodes_choices,
        dropout_choices=dropout_choices,
        epoch_choices=epoch_choices,
        weight_decay_choices=weight_decay_choices,
        modeltag=modeltag,
        preprocess_inputs=preprocess_inputs,
        batch_samples=batch_samples,
        shuffle_mode=shuffle_mode,
        shuffle_buffer_samples=shuffle_buffer_samples,
        shuffle_buffer_chunks=shuffle_buffer_chunks,
        exclude_unconnected_targets=exclude_unconnected_targets,
        sample_weighting=sample_weighting,
        override_name=override_name,
        new_name=new_name,
        jobflavor=jobflavor,
        request_gpus=request_gpus,
        request_cpus=request_cpus,
        request_memory_gb=request_memory_gb,
        submit_jobs=submit_jobs,
    )


if __name__ == "__main__":
    main()
