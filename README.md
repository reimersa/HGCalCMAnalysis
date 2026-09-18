# HGCalCMAnalysis

This repository derives, applies, and evaluates common-mode (CM) noise corrections for HGCal module data. It supports the Sep2025 and Jun2026 test-beam layouts and operates on module-level histofiller ROOT inputs.

The workflow is split into two entry points:

- `derive.py` prepares training data and derives correction artifacts.
- `apply.py` applies existing artifacts to a target run and produces diagnostics and plots.

Configuration is currently code-based: edit the setup block near the top of the relevant entry-point script, inspect the resolved configuration with `--show`, and then run the requested stages.

## Correction methods

Four evaluation modes are available through `apply.py`:

- `uncorrected` computes the reference diagnostics without applying a correction.
- `analytic` is the original linear CM-channel correction.
- `analytic_allinputs` fits an independent linear predictor for every target channel using all other channel measurements and the common event-level inputs.
- `dnn` applies a shared nonlinear per-channel network trained across one or more modules.

The primary comparison is between `analytic_allinputs` and `dnn`: both can use the full set of channel measurements, while the DNN additionally receives target-channel metadata and can learn nonlinear relations.

### All-channel, multi-module DNN

For a module with `C` channels and a training vocabulary of `N` modules, each DNN sample represents one `(event, target channel)` pair. One network is shared by every target channel and every training module.

The `all_channels_multimodule_v1` feature schema contains:

- 12 pedestal-subtracted CM values;
- event summaries: `nchtoa`, `nchtot`, `nchadcgt10`, `nchadcgt50`, `nchadcgt200`, and `nchadcgt500`;
- `N + 1` one-hot module entries, including a reserved `UNKNOWN` entry;
- all `C` unmasked pedestal-subtracted ADC measurements, ordered by channel index;
- target-channel index, eRx index, and relative cell area.

The input dimension is therefore `C + N + 22`. With 222 channels and 10 training modules, the network has 254 inputs, of which 222 are channel measurements.

When channel `i` is the prediction target, `adc_allch_i` is set to raw zero before optional preprocessing. The network consequently receives the other `C - 1` measured channels without being given the value it must predict. Unconnected channels remain part of the input vector and, with the current `exclude_unconnected_targets=False` setting, are also retained as prediction targets.

The module vocabulary is the ordered `modulenames` list in `derive.py`. Index zero belongs to the first module, index one to the second, and so on. The final one-hot entry is reserved for a module that was not present during training. `model_manifest.json` records this mapping, the exact feature order, the architecture, and the preprocessing setting. During application, a module absent from the saved vocabulary automatically activates `UNKNOWN`.

The reserved entry is never active during the current training procedure. Applying a model to an unseen module is supported mechanically, but its performance must be validated as an extrapolation.

Training uses buffered cross-module chunk shuffling. Events from the configured modules are mixed during optimization without loading the complete dataset into memory.

### Optional DNN preprocessing

Set `dnn_preprocess_inputs` in both entry-point scripts to select the matching model variant:

```python
dnn_preprocess_inputs = True
```

When enabled, every input feature and every per-channel target is z-score transformed. Means and standard deviations are computed once from the retained training events, saved in `input_preprocessing.json`, and reused for all batches and during application. They are not recomputed per batch. The target channel's ADC input is zeroed first and the resulting feature vector is then standardized.

When preprocessing is disabled, the network receives values in their original units and no preprocessing file is needed. A preprocessed model receives the automatic `inputzscore` suffix so that both variants can coexist.

### Analytic all-input predictor

`analytic_allinputs` derives one linear regression for each target channel. It uses the prepared event-level features, module one-hot entries, and all channel ADC measurements. The target channel's own ADC coefficient is fixed to zero.

Target-channel metadata are omitted because every channel has its own coefficient row and intercept. The regression is solved from streaming covariance moments using feature scaling and a pseudoinverse; the saved weights are converted back to the original input units. Application therefore does not require DNN preprocessing or a normalization file.

For an unseen module, the reserved `UNKNOWN` one-hot entry is activated. Its coefficient is zero because that entry is constant during derivation.

## Installation

Clone the repository in a location visible from lxplus:

```bash
git clone git@github.com:reimersa/HGCalCMAnalysis.git
cd HGCalCMAnalysis
```

Alternatively, enter this directory from a full `LocalCalibration` checkout:

```bash
cd /path/to/LocalCalibration/scripts/HGCalCMAnalysis
```

Create the virtual environment in the EOS location expected by the Condor wrapper:

```bash
python3.9 -m venv /eos/user/${USER:0:1}/${USER}/torch-env
source /eos/user/${USER:0:1}/${USER}/torch-env/bin/activate
pip install -r requirements.txt
```

Run all commands below from the `HGCalCMAnalysis` directory with this environment active. A CMSSW runtime is not required by the Python derivation and application scripts.

## Data and output locations

`classes.AnalysisConfig` defines the directory layout. In the current configuration:

- raw histofiller data are read from `/eos/user/a/areimers/hgcal/<campaign>`;
- generated parquet files and correction artifacts are written to `/eos/user/<initial>/<user>/hgcal/<campaign>`;
- plots are written below `plots/<campaign>` in the repository;
- geometry and cell-area inputs are read from `data/`.

Users who do not read from the shared raw-data location should update `raw_datafolder_base` in `classes.py`. Supported campaign names are `Sep2025TB` and `Jun2026TB`; the entry-point setup blocks currently select `Sep2025TB`.

Generated products are separated by run, pedestal run, correction module or module group, correction run, and selection. Prepared DNN inputs have an additional deterministic schema folder containing the feature version and ordered module vocabulary. This allows, for example, five-module and ten-module input sets to coexist.

## Deriving corrections

### 1. Configure `derive.py`

Edit the setup block near the top of `derive.py`. The main settings are:

```python
modulenames = [
    "ML_F3WC_IH0180",
    "ML_F3WC_IH0182",
    # ...
]

selection_for_correction = "selection_trigtime"
correction_run = "112044_112050_112060_112073_adcmax10"
pedestal_run = 112044

dnn_feature_version = prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS
dnn_model_tag = "allchannels_multimodule"
dnn_preprocess_inputs = True
per_channel_cols = ["channel_indices", "erx_indices", "cell_area_fraction"]

frac = 0.5 if len(modulenames) > 5 else 1.0
train_event_fractions = {module: frac for module in modulenames}
validation_event_fractions = {module: frac for module in modulenames}
```

`combine_modules` is derived from the number of configured modules. For multi-module training, correction artifacts are stored below a `MULTI_<ordered module names>` group.

The event fractions are applied independently to each module after the stable train/test split. Fractions must be in `(0, 1]`. Event-ID hashing selects a reproducible subset. These fractions are training settings: changing them does not require `-i` to regenerate the materialized feature files.

Before starting expensive work, inspect the complete resolved setup:

```bash
./derive.py --show
```

### 2. Prepare the analysis data

The stages can be run separately:

```bash
./derive.py -p       # pedestal means and standard deviations
./derive.py -c       # ROOT or configured synthetic runs to parquet
./derive.py -s       # derived variables and event selections
./derive.py -i       # materialized all-channel inputs, targets, and split files
```

Conversion is required before preparing the all-channel schema because `-i` reads the pre-cut `adc_chNNN_pedsub_nocut` columns. Use `--plot-dnninputs` together with `-i` to produce a histogram and summary for every materialized feature:

```bash
./derive.py -i --plot-dnninputs
```

Rerun `-i` when the input data, selection, feature schema, module vocabulary, or vocabulary order changes. It is not necessary when only the network architecture, epoch count, preprocessing choice, model tag, or event fractions change.

### 3. Derive analytic corrections

The original CM-channel predictor is derived with:

```bash
./derive.py -a
```

The full all-input linear predictor requires the materialized inputs from the preceding step:

```bash
./derive.py -A
```

Both can be requested after data preparation:

```bash
./derive.py -a -A
```

`analytic_allinputs` writes:

- `analytic_allinputs_weights.parquet`;
- `analytic_allinputs_intercepts.parquet`;
- `analytic_allinputs_manifest.json`.

### 4. Train the DNN

For a single local training process:

```bash
./derive.py -d
```

For the configured multi-module training on HTCondor:

```bash
voms-proxy-init --voms cms
./derive.py -q
```

Submission requires a valid proxy at `/tmp/x509up_u$(id -u)`. The current job request is one GPU, one CPU, 16 GB of memory, and the `tomorrow` job flavour. Generated submit files and live logs are placed under `workdir_condor/`:

```bash
condor_q "$USER"
tail -f workdir_condor/<job-name>/<job-name>.out
```

The default configured network has hidden layers `[256, 256, 256, 32]`, no dropout, batches of 1024 `(event, channel)` samples, and 200 epochs.

Each model directory contains at least:

- `dnn_best.pth` and `dnn_last.pth`;
- `model_manifest.json`;
- `train_losses.npy` and `test_losses.npy`;
- `input_preprocessing.json` when preprocessing is enabled.

To distribute an inference model, provide `dnn_best.pth` and `model_manifest.json`, plus `input_preprocessing.json` for a preprocessed model. The manifest is required to reproduce the feature order and module mapping.

### Combined derivation command

The complete pipeline, including Condor submission, can be launched with:

```bash
./derive.py --all
```

For production work, running the stages separately is usually easier to inspect and resume.

## Applying corrections

### 1. Configure `apply.py`

Edit the setup block near the top of `apply.py`:

```python
modulenames = ["ML_F3WC_IH0182"]
selection = "selection_trigtime"
selection_for_correction = "selection_trigtime"
target_run = 112049
pedestal_run = 112044
correction_run = "112044_112050_112060_112073_adcmax10"

training_modules = [
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
module_for_correction = f"MULTI_{'_'.join(training_modules)}"
dnn_tag = "allchannels_multimodule"
dnn_preprocess_inputs = True
```

`module_for_correction` identifies the artifact group, not necessarily the target module. For a multi-module model it must use the exact ordered training module list. The target module may be one of those modules or an unseen module; the saved DNN manifest determines the appropriate one-hot input.

The following settings must match the derived model:

- `correction_run`;
- `pedestal_run`;
- `selection_for_correction`;
- `module_for_correction`;
- `dnn_tag`, architecture, and `dnn_preprocess_inputs` for DNN inference.

Check the resolved paths and model tag before running:

```bash
./apply.py --show
```

### 2. Convert a fresh target run and apply corrections

For a fresh target run, compare the uncorrected data, all-input linear regression, and DNN with:

```bash
./apply.py \
  -c -s -k -p \
  -m uncorrected analytic_allinputs dnn
```

This performs:

1. target-data conversion;
2. variable and selection construction;
3. correction application;
4. covariance/eigen, covariance-noise-model, and projection diagnostics;
5. detailed and summary plots.

The uncorrected method is included because its leading noise eigenvector is the reference basis used by the corrected projection diagnostics.

If conversion and selections already exist, omit `-c -s`:

```bash
./apply.py -k -p -m uncorrected analytic_allinputs dnn
```

To recompute only plots from existing corrected columns and diagnostic artifacts:

```bash
./apply.py -p -m uncorrected analytic_allinputs dnn
```

To evaluate only one correction after the uncorrected reference has already been computed:

```bash
./apply.py -k -p -m dnn
```

The original CM-channel method can be included explicitly with `-m analytic`. `--all` runs conversion, selections, computation, and plotting for all four methods, and therefore requires artifacts for every method:

```bash
./apply.py --all
```

Use `--plot-dnninputs` with DNN computation to inspect every feature passed to the saved network. For preprocessed models, both raw and network-space distributions are written:

```bash
./apply.py -k -m dnn --plot-dnninputs
```

DNN prediction and residual column names include the resolved model tag. This prevents raw-input and preprocessed corrections from overwriting one another. Summary plot directories likewise include the compared correction tags.

## MIP/Landau fits

For non-pedestal target runs, the normal `apply.py -p` workflow fits the pooled all-channel distribution for each plotted method. The model contains pedestal, one-MIP, and two-MIP components with Landau-Gaussian convolution. Linear and logarithmic PDFs and a JSON fit summary are written below the method's `distributions_1d` plot directory. Plot annotations report the Landau location, Landau width `c_L`, fitted one-MIP peak, and fit quality.

The fitter can also be run directly on arbitrary parquet files:

```bash
./mip_landau.py \
  '/eos/user/.../analysis_inputs/.../df_batch*.parquet' \
  --columns 'adc_ch*_pedsub_resid_dnn_*' \
  --output-dir plots/mip_landau \
  --range -10 40 \
  --fit-range -5 35
```

Input paths may be literal parquet files or glob patterns. `--columns` uses shell-style `fnmatch` matching against parquet column names.

## Reproducibility and schema checks

The workflow writes manifests alongside prepared inputs, DNN models, and analytic all-input artifacts. At training and application time it validates:

- the feature schema and exact feature order;
- the ordered module vocabulary and reserved unknown index;
- the channel count and per-channel columns;
- the DNN architecture, model tag, and preprocessing choice.

Do not manually reorder columns or module names. Copy the associated manifest whenever moving a correction artifact. Prepared inputs for different vocabularies intentionally live in separate schema folders; training configurations that share a vocabulary and feature version reuse the same materialized inputs.

## Tests

Run the focused regression tests from the repository directory:

```bash
python -m unittest \
  tests.test_dnn_multimodule \
  tests.test_analytic_allinputs \
  tests.test_mip_plotting
```

The tests cover module encoding and unknown-module handling, target-channel masking, preprocessing and manifest checks, analytic all-input derivation/application, and MIP plotting integration.
