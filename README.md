# HGCalCMAnalysis

HGCal common-mode analysis workflow for Sep2025 test-beam data.

This directory contains the scripts used to prepare analysis inputs and derive common-mode correction artifacts. The currently supported correction derivation methods are:

- analytic linear regression
- DNN regression

Application and plotting/evaluation are still being refactored and will be documented after `apply.py` is updated.

## Get the Code

Clone the analysis repository into a location visible from lxplus, for example on AFS:

```bash
git clone git@github.com:reimersa/HGCalCMAnalysis.git
cd HGCalCMAnalysis
```

If working from a full `LocalCalibration` checkout instead, go to the analysis directory:

```bash
cd /path/to/LocalCalibration/scripts/HGCalCMAnalysis
```

All commands below should be run from this `HGCalCMAnalysis` directory.

## Environment Setup

Create the Python virtual environment on your own EOS area:

```bash
python3.9 -m venv /eos/user/${USER:0:1}/${USER}/torch-env
source /eos/user/${USER:0:1}/${USER}/torch-env/bin/activate
pip install -r requirements.txt
```

Start a working session from the analysis directory in your checkout. This can be inside CMSSW, but CMSSW setup is not required for the derive workflow:

```bash
cd /path/to/HGCalCMAnalysis
source /eos/user/${USER:0:1}/${USER}/torch-env/bin/activate
```

`requirements.txt` pins the package versions used for this workflow.

## Data Locations

Raw ROOT/histofiller inputs are intentionally read from:

```text
/eos/user/a/areimers/hgcal/Sep2025TB
```

Generated analysis inputs, correction artifacts, DNN inputs, trained models, and plots are written under the current user's EOS area:

```text
/eos/user/${USER:0:1}/${USER}/hgcal/Sep2025TB
```

These locations are configured in `classes.py`. Static geometry inputs needed by this workflow, such as `cellareas.json`, are stored inside this directory under `data/`.

## Deriving Corrections

The derivation workflow is controlled by `derive.py`.

First edit the setup block near the top of `derive.py`:

- `modulenames`
- `selection_for_correction`
- `correction_run`
- `pedestal_run`
- `per_channel_cols`

Then run the desired workflow steps with command-line flags. Running `derive.py` without flags prints the current setup and exits.

Show the configured setup and available options:

```bash
python derive.py --show
python derive.py --help
```

Available derive steps:

```text
-p, --pedestals   calculate pedestal means/stds
-c, --convert     convert ROOT/synthetic inputs to parquet
-s, --selections  add variables and event selections
-a, --analytic    compute covariance/eigen artifacts and analytic predictor
-d, --localdnn    prepare DNN inputs, refresh split selections, train one DNN locally
-q, --submitdnn   prepare DNN inputs, refresh split selections, submit DNN Condor jobs
    --all         run pedestals, convert, selections, analytic, and submitdnn
```

Typical full derivation with Condor DNN training:

```bash
python derive.py --all
```

Typical stepwise derivation:

```bash
python derive.py -p
python derive.py -c
python derive.py -s -a
python derive.py -q
```

Local DNN training alternative:

```bash
python derive.py -d
```

`--localdnn`, `--submitdnn`, and `--all` prepare DNN inputs and then refresh only the train/test split selections from the DNN split file.

## Condor DNN Submission

`python derive.py --submitdnn` and `python derive.py --all` submit DNN training jobs through Condor via `submit_train.py`.

Before submitting, make sure a VOMS proxy exists at:

```bash
/tmp/x509up_u$(id -u)
```

## Apply Workflow

The apply/evaluation workflow is still being cleaned up. This section will be updated after the `apply.py` refactor.
