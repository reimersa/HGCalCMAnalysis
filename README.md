# HGCalCMAnalysis

HGCal common-mode analysis workflow for test beam data.

## Overview

This repository contains the Python code used to:

- prepare analysis inputs from ROOT files
- derive covariance matrices, eigenvectors, and correction objects
- apply analytic and DNN-based corrections
- produce plots for evaluation

The analysis code lives at the repository root. Input data, intermediate products, trained models, and other large outputs are stored on EOS.

## Running the Workflow

The main steering script is:

```bash
steer.py
```

The workflow is configured by editing `steer.py` and commenting or uncommenting the steps to run.

Because the code uses flat local imports such as `import classes`, run scripts from the repository root:

```bash
python steer.py
```

## Environment Setup

Create a Python virtual environment on EOS and install the required packages:

```bash
python3.9 -m venv /eos/user/${USER:0:1}/${USER}/torch-env
source /eos/user/${USER:0:1}/${USER}/torch-env/bin/activate
pip install -r requirements.txt
```

Start a working session with:

```bash
cd /afs/cern.ch/user/a/areimers/CMSSW_15_1_0_pre1/src/HGCalCommissioning/LocalCalibration/scripts/HGCalCMAnalysis
source /eos/user/${USER:0:1}/${USER}/torch-env/bin/activate
```

For convenience, users may define an alias such as:

```bash
alias hgcalcm='cd /afs/cern.ch/user/a/areimers/CMSSW_15_1_0_pre1/src/HGCalCommissioning/LocalCalibration/scripts/HGCalCMAnalysis; source /eos/user/${USER:0:1}/${USER}/torch-env/bin/activate'
```

`requirements.txt` lists the package versions used to run the workflow.

## Data Location

The code expects access to the EOS paths configured in `classes.py`.
