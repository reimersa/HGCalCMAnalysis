# HGCalCMAnalysis

HGCal common-mode analysis workflow for Sep2025 test-beam data.

This directory contains the scripts used to prepare analysis inputs and derive common-mode correction artifacts. The currently supported correction derivation methods are:

- analytic linear regression
- DNN regression

Application and plotting/evaluation are controlled by `apply.py`.

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
-i, --dnninputs   prepare DNN inputs and refresh train/test split selections
-d, --localdnn    train one DNN locally using existing prepared DNN inputs
-q, --submitdnn   submit DNN Condor jobs using existing prepared DNN inputs
    --all         run pedestals, convert, selections, analytic, dnninputs, and submitdnn
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
python derive.py -i
python derive.py -q
```

Local DNN training alternative:

```bash
python derive.py -i
python derive.py -d
```

`--dnninputs` prepares DNN inputs and then refreshes only the train/test split selections from the DNN split file. It only needs to be rerun when the underlying analysis inputs, selections, or DNN input features change. If only the DNN architecture, training tag, or training hyperparameters change, rerun `--localdnn` or `--submitdnn` without `--dnninputs`.

Prepared parquet inputs are fully materialized under a deterministic schema subfolder of `dnn_training_inputs`. The schema key includes the feature version and ordered module vocabulary, so inputs for different module combinations coexist without overwriting one another. Network layout, preprocessing, and other training hyperparameters reuse the same prepared schema.

### All-channel, multi-module DNN

The `all_channels_multimodule_v1` schema passes every unmasked channel ADC to the DNN. When predicting channel `i`, its own ADC slot is set to zero, leaving the other `N-1` measurements. It also adds one-hot module inputs with one reserved final `UNKNOWN` entry.

Set the following in `derive.py`:

```python
modulenames = ["ML_F3WC_IH0182", "ML_F3WC_IH0190"]
dnn_feature_version = prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS
combine_modules = True
per_channel_cols = ["channel_indices", "erx_indices", "cell_area_fraction"]
dnn_model_tag = "allchannels_multimodule"
dnn_preprocess_inputs = True  # set False to train on raw inputs/targets
train_event_fractions = {
    "ML_F3WC_IH0182": 1.0,
    "ML_F3WC_IH0190": 0.5,
}
validation_event_fractions = {
    "ML_F3WC_IH0182": 1.0,
    "ML_F3WC_IH0190": 1.0,
}
```

Fractions are applied independently after the train/validation split. Event-ID hashing chooses a stable subset, so the same physical events are retained in every epoch and under every shuffle mode. The values must be in `(0, 1]`; remove a module from `modulenames` instead of assigning it zero.

When `dnn_preprocess_inputs` is true, both inputs and per-channel targets are z-score transformed using statistics computed only from the retained training events. The resolved model name receives an automatic `inputzscore` suffix. When false, no preprocessing file is used and the base model tag is retained. Set the corresponding `dnn_preprocess_inputs` and `dnn_tag` values in `apply.py` when applying the checkpoint.

Run conversion again before preparing this schema because it requires the pre-cut `adc_chNNN_pedsub_nocut` columns. Training writes `model_manifest.json`, including the ordered module vocabulary and reserved unknown index. At application time, set `module_for_correction` to the corresponding `MULTI_...` model group. A target module absent from the training vocabulary automatically activates the saved `UNKNOWN` entry.

The reserved `UNKNOWN` input is intentionally never active during training, matching the initial student procedure; no module-ID dropout or unknown-category retraining is performed.

### Analytic all-inputs baseline

`analytic_allinputs` derives one independent linear predictor per target channel from the same materialized all-channel inputs and retained training events used by the DNN. Each predictor uses the CM values, event-summary counts, module one-hot values, and the other `N-1` channel ADCs. Its own ADC coefficient is fixed to zero. The DNN's channel-index, eRx-index, and cell-area inputs are unnecessary here because every target has its own coefficient row and intercept.

After preparing inputs, derive the predictor with:

```bash
./derive.py --analytic-allinputs
```

Use `./derive.py -i --analytic-allinputs` when the configured input schema has not yet been materialized. The derivation uses only the prepared `train` split and applies `train_event_fractions` independently to each module. It writes weights, intercepts, and a feature/module manifest under the combined correction group's `predictors` folder.

Apply and evaluate it like the existing methods:

```bash
./apply.py -k -p -m analytic_allinputs
```

Application does not need a preprocessing switch or preprocessing JSON. Feature scaling is used only internally to stabilize the pseudoinverse, after which weights are stored in the original input units. An unseen module activates the reserved `UNKNOWN` one-hot entry; because that entry is constant during derivation, its coefficient is fixed to zero.

## Condor DNN Submission

`python derive.py --submitdnn` and `python derive.py --all` submit DNN training jobs through Condor via `submit_train.py`.

Before submitting, make sure a VOMS proxy exists at:

```bash
/tmp/x509up_u$(id -u)
```

## Apply Workflow

The application workflow is controlled by `apply.py`.

First edit the setup block near the top of `apply.py`:

- `modulenames`
- `selection`
- `selection_for_correction`
- `target_run`
- `pedestal_run`
- `correction_run`
- `module_for_correction`
- DNN settings: `dnn_tag` and `dnn_preprocess_inputs`

Running `apply.py` without flags prints the current setup and exits.

Show the configured setup and available options:

```bash
python apply.py --show
python apply.py --help
```

Available apply steps:

```text
-c, --convert     convert target ROOT/synthetic inputs to parquet
-s, --selections  add variables and event selections on the target run
-m, --methods     choose methods: uncorrected analytic dnn
-k, --compute     add corrections where applicable, then compute diagnostics
-p, --plots       make detailed plots and summary comparison plots
    --all         run convert, selections, compute, and plots for all methods
```

`--compute` always runs the method-specific correction step together with the downstream diagnostics: covariance/eigen outputs, fitted covariance noise model, and projection onto the uncorrected noise mode. For `uncorrected`, no correction is added; only the diagnostics are computed.

`--plots` makes the detailed plots and summary comparison plots together. It is useful when the correction and diagnostics already exist and only the plots need to be rerun.

Typical stepwise application:

```bash
python apply.py -c
python apply.py -s
python apply.py -m uncorrected -k
python apply.py -m analytic dnn -k
python apply.py -m dnn -p
```

Compute and plot only the DNN method:

```bash
python apply.py -m dnn -k -p
```

Full sweep over all methods:

```bash
python apply.py -m uncorrected analytic dnn -c -s -k -p
```

The DNN strategy is selected inside the setup block. The current default is:

```python
dnn_tag = "chunkshuffle_modulesummaries_targetspreproc"
dnn_preprocess_inputs = True
```

The baseline DNN can be selected by switching to:

```python
dnn_tag = ""
dnn_preprocess_inputs = False
```

DNN output columns include the resolved model tag so different DNN corrections can coexist in the same parquet files. The baseline DNN keeps the historical names:

```text
adc_ch000_pedsub_pred_dnn
adc_ch000_pedsub_resid_dnn
```

A tagged/preprocessed DNN writes columns such as:

```text
adc_ch000_pedsub_pred_dnn_chunkshuffle_modulesummaries_targetspreproc_inputzscore
adc_ch000_pedsub_resid_dnn_chunkshuffle_modulesummaries_targetspreproc_inputzscore
```

## Standalone MIP/Landau Fit

`mip_landau.py` streams matching columns from parquet inputs, fits a pedestal plus one- and two-MIP Landau-Gaussian model, and writes linear/logarithmic PDFs and a JSON fit summary:

```bash
python mip_landau.py \
  '/eos/user/.../analysis_inputs/.../df_batch*.parquet' \
  --columns 'adc_ch*_pedsub_resid_dnn_*' \
  --output-dir plots/mip_landau \
  --range -10 40 \
  --fit-range -5 35
```

The numerical Landau lookup is initialized lazily on the first fit, so importing the ordinary plotting modules has no added cost.

For non-pedestal runs, the normal `apply.py --plots` workflow also performs this fit on the pooled all-channel distribution after applying the configured event selection. The method-specific `distributions_1d` folder receives linear/logarithmic fit PDFs and a JSON summary. The plot annotation includes the Landau location, Landau width `c_L`, fitted one-MIP peak, and fit quality.
