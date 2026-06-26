import os
import pwd
import re
from glob import glob
import numpy as np # type: ignore
import pandas as pd # type: ignore

from typing import List, Optional, Dict, Tuple, Iterable, Union
from dataclasses import dataclass, field

import utils

PEDESTAL_RUNS = {110398, 112044, 1747296821}


class Batch:
    def __init__(self, df, cfg):
        self.df = df
        self.cfg = cfg

    @property
    def full_df(self):
        return self.df

    @property
    def cm_df(self):
        return self.df[[f"cm_erx{i:02d}_pedsub" for i in range(self.cfg.ncmchannels)]]

    def df_cols(self, colnames):
        return self.df[colnames]

    @property
    def measurements_df(self):
        return self.df[[f"adc_ch{i:03d}_pedsub" for i in range(self.cfg.nch)]]

    @property
    def measurements_df_with_cm_df(self):
        return utils.add_cms_to_measurements_df(self.measurements_df, self.cm_df, drop_constant_cm=False)

    def select_flag(self, flag_col: str):
        if not flag_col:
            return Batch(self.df, self.cfg)

        if flag_col not in self.df.columns:
            raise KeyError(f"Column '{flag_col}' not found in DataFrame.")

        mask = self.df[flag_col].astype(bool)
        filtered_df = self.df[mask]

        return Batch(filtered_df, self.cfg)

class AnalysisBatchIter:

    def __init__(self, cfg: "AnalysisConfig"):
        self.cfg = cfg

        # Underlying ordered, unshuffled arrays (train+val together -> "combined")
        self.inputfiles  = sorted(glob(os.path.join(self.cfg.analysis_inputs_folder, "df_batch*.parquet")))
        if self.cfg.maxfiles_for_eval:
            maxfiles = min(self.cfg.maxfiles_for_eval, len(self.inputfiles))
            self.inputfiles = self.inputfiles[:maxfiles]

        if not self.inputfiles:
            raise RuntimeError(f"No ordered-combined batch files found in {self.cfg.analysis_inputs_folder}.\nExpected files matching:\ndf_batch*.parquet")

    def __iter__(self):
        for infile in self.inputfiles:
            df = pd.read_parquet(infile)
            yield Batch(df, self.cfg)


def is_pedestal_run(run: Union[int, str]) -> bool:
    return run in PEDESTAL_RUNS



@dataclass
class AnalysisConfig:
    # training/eval module selection
    modulename: str
    run: Union[int, str]
    run_for_pedestal: Union[int, str]
    run_for_correction: List[Union[int, str]]
    module_for_correction: str
    corrections_tag: str = ""
    runs_per_synthetic_run: dict[str, list[Union[int, str]]] = field(default_factory=dict)
    channel_ucoords_to_keep_per_run: dict[int, list[int]] = field(default_factory=dict)

    derive_correction: bool = False
    selection_for_correction: str = ""
    standardize_std: bool = False
    is_pedestal: bool = False
    maxfiles_for_eval: int = None


    # optional (set in postinit or kind of fixed)
    inputfoldertag: str = ""
    ncmchannels: int = 12
    nch_per_erx: Optional[int] = None
    nerx: Optional[int] = None
    nch: Optional[int] = None
    unconnected_channels: Optional[list[int]] = None

    # folders, all set internally for consistency
    raw_datafolder_base: str = None
    datafolder_base: str = None
    histofiller_folder: str = None
    analysis_inputs_folder: str = None
    pedestal_mean_std_folder: str = None
    pedestal_covs_folder: str = None
    own_covs_folder: str = None
    noise_model_fit_folder: str = None
    analytic_predictor_folder: str = None
    dnn_models_folder: str = None
    dnn_training_input_folder: str = None
    plotfolder_base: str = None


    def __post_init__(self):
        self.inputfoldertag = utils.get_input_tag(basetag=self.inputfoldertag, normalize_to_unit_area=False, remove_disconnected=False, standardize_std=self.standardize_std)
        self.infer_layout()
        correction_subfolder = self.get_correction_subfolder()

        username = pwd.getpwuid(os.getuid()).pw_name
        self.raw_datafolder_base             = "/eos/user/a/areimers/hgcal/Sep2025TB"
        self.datafolder_base                 = f"/eos/user/{username[0]}/{username}/hgcal/Sep2025TB"
        self.histofiller_folder              = self.get_histofiller_folder()
        self.analysis_inputs_folder          = os.path.join(self.datafolder_base, f"Run{self.run}/analysis_inputs{self.inputfoldertag}/{self.modulename}/pedestals_from_Run{self.run_for_pedestal}", correction_subfolder)
        self.pedestal_mean_std_folder        = os.path.join(self.datafolder_base, f"Run{self.run_for_pedestal}/means_stds{self.inputfoldertag}/{self.modulename}")
        self.own_covs_folder                 = os.path.join(self.datafolder_base, f"Run{self.run}/covs{self.inputfoldertag}/{self.modulename}/pedestals_from_Run{self.run_for_pedestal}", correction_subfolder)
        self.noise_model_fit_folder          = os.path.join(self.own_covs_folder, "noise_model_fits")

        self.corrections_base_folder   = os.path.join(self.datafolder_base, f"corrections{self.inputfoldertag}", str(self.get_correction_module()), f"pedestals_from_Run{self.run_for_pedestal}", correction_subfolder)
        self.analytic_predictor_folder = os.path.join(self.corrections_base_folder, "predictors")
        self.dnn_models_folder         = os.path.join(self.corrections_base_folder, "dnn_models")
        self.dnn_training_input_folder = os.path.join(self.corrections_base_folder, "dnn_training_inputs")
        self.corrections_covs_folder   = os.path.join(self.datafolder_base, f"Run{self.get_correction_run()}/covs{self.inputfoldertag}/{self.get_correction_module()}/pedestals_from_Run{self.run_for_pedestal}", correction_subfolder)

        self.is_pedestal = is_pedestal_run(self.run)

        self.runs_per_synthetic_run = {
            "112046_adcmax10": [112046],
            "112047_adcmax10": [112047],
            "112048_adcmax10": [112048],
            "112049_adcmax10": [112049],
            "112050_adcmax50": [112050],
            "112050_adcmax10": [112050],
            "112060_adcmax10": [112060],
            "112051_adcmax10": [112051],
            "112044_112050": [112044, 112050],
            "112044_112050_full": [112044, 112050],
            "112073_outer": [112073],
            "112060_outer": [112060],
            "112068_adcmax10": [112068],
            "112044_112050_112060_112073_adcmax5": [112044, 112050, 112060, 112073],
            "112044_112050_112060_112073_adcmax10": [112044, 112050, 112060, 112073],
            "112044_112050_112060_112073_adcmax30": [112044, 112050, 112060, 112073],
            "112050_112060_112073_adcmax10": [112050, 112060, 112073],
            "112046_112047_112048_112049_112050_adcmax10": [112046, 112047, 112048, 112049, 112050],
        }

        self.runs_to_select_rings_for = ["112044_112050", "112073_outer", "112060_outer"]

        # -1 are the unconnected channels
        self.channel_rings_to_keep_per_run = {
            112044: "all",
            112050: [-1, 0, 1],
            # 112050: "all", 
            112073: [-1, 0, 1], 
            112060: [-1, 0, 1], 
        }

        self.adcmax_per_run = {
            "112046_adcmax10": 10,
            "112047_adcmax10": 10,
            "112048_adcmax10": 10,
            "112049_adcmax10": 10,
            "112050_adcmax50": 50,
            "112050_adcmax10": 10,
            "112060_adcmax10": 10,
            "112051_adcmax10": 10,
            "112068_adcmax10": 10,
            "112044_112050_112060_112073_adcmax5": 5,
            "112044_112050_112060_112073_adcmax10": 10,
            "112044_112050_112060_112073_adcmax30": 30,
            "112050_112060_112073_adcmax10": 10,
            "112046_112047_112048_112049_112050_adcmax10": 10,
        }
        self.adcmax = None
        if self.run in self.adcmax_per_run.keys():
            self.adcmax = self.adcmax_per_run[self.run]

        self.unconnected_channels = [erxidx*self.nch_per_erx+opt for erxidx in range(self.nerx) for opt in [8, 17, 19, 28]]

        self.plotfolder_base = os.path.join(
            ".",
            "plots",
            "Sep2025TB",
            f"Run{self.run}",
            f"{self.modulename}{self.inputfoldertag}",
            f"pedestals_from_Run{self.run_for_pedestal}",
            correction_subfolder,
        )

    def get_histofiller_folder(self):
        return os.path.join(self.raw_datafolder_base, f"output/Run{self.run}/histofiller")

    def _sanitize_correction_tag(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
        cleaned = cleaned.strip("_")
        if not cleaned:
            raise ValueError(f"Could not derive a stable correction tag from value={value!r}")
        return cleaned

    def get_correction_module(self):
        return self.modulename if self.derive_correction else self.module_for_correction

    def get_correction_run(self):
        return self.run if self.derive_correction else self.run_for_correction

    def get_correction_tag(self):
        tags = []
        if self.corrections_tag:
            tags.append(self._sanitize_correction_tag(self.corrections_tag))
        if self.selection_for_correction:
            tags.append(self._sanitize_correction_tag(self.selection_for_correction))
        return "__".join(tags)

    def get_correction_subfolder(self):
        parts = [
            f"corrections_from_Module{self.get_correction_module()}",
            f"corrections_from_Run{self.get_correction_run()}",
        ]
        correction_tag = self.get_correction_tag()
        if correction_tag:
            parts.append(f"corrections_tag_{correction_tag}")
        return os.path.join(*parts)

    # derive defaults from module naming convention
    def infer_layout(self) -> None:
        if self.modulename.startswith("ML"):
            self.nch_per_erx = 37
            self.nerx = 6 if self.nerx is None else self.nerx
        else:
            self.nch_per_erx = 74
            self.nerx = 12 if self.nerx is None else self.nerx
        self.nch = self.nch_per_erx * self.nerx

    def load_from_cov_folder(self, filename):
        return pd.read_parquet(os.path.join(self.own_covs_folder, filename))

    def load_from_noise_model_fit_folder(self, filename):
        return pd.read_parquet(os.path.join(self.noise_model_fit_folder, filename))

    def load_from_corrections_cov_folder(self, filename):
        return pd.read_parquet(os.path.join(self.corrections_covs_folder, filename))




def add_cms_to_measurements_df(measurements_df: pd.DataFrame, cm_df: pd.DataFrame, drop_constant_cm: bool = True) -> pd.DataFrame:
    X = pd.concat([measurements_df, cm_df], axis=1)
    if drop_constant_cm and cm_df.shape[1] > 0:
        # drop CM columns with zero variance (avoid NaNs in correlation)
        cm_std = cm_df.std(axis=0)
        keep = cm_std[cm_std > 0].index
        dropped = [c for c in cm_df.columns if c not in keep]
        if dropped:
            print(f"[info] Dropping {len(dropped)} constant CM columns: {dropped}")
        X = pd.concat([measurements_df, cm_df[keep]], axis=1)
    return X




class CovAccumulator:
    """
    Streaming covariance between two matrices X_i and X_j with matching rows (events),
    possibly different columns (channels/features).
    It accumulates the pairwise-valid counts, sums, and sum-of-products needed for a
    NaN-aware, pairwise-centered covariance estimate.
    """
    def __init__(self, cols_i: List[str], cols_j: List[str]):
        self.cols_i = list(cols_i)
        self.cols_j = list(cols_j)
        self.n_i = len(self.cols_i)
        self.n_j = len(self.cols_j)
        self.N = np.zeros((self.n_i, self.n_j), dtype=float)
        self.Sxy = np.zeros((self.n_i, self.n_j), dtype=float)
        self.Sx = np.zeros((self.n_i, self.n_j), dtype=float)
        self.Sy = np.zeros((self.n_i, self.n_j), dtype=float)

    def update(self, df_i: pd.DataFrame, df_j: pd.DataFrame) -> None:
        # sanity
        if not df_i.index.equals(df_j.index):
            raise RuntimeError("CovAccumulator.update: row indices must match.")
        if list(df_i.columns) != self.cols_i or list(df_j.columns) != self.cols_j:
            raise RuntimeError("CovAccumulator.update: column order changed.")

        # masks and zero-filled arrays
        M_i = df_i.notna().astype(float).to_numpy()
        M_j = df_j.notna().astype(float).to_numpy()
        X_i = df_i.fillna(0.0).to_numpy()
        X_j = df_j.fillna(0.0).to_numpy()

        # Pairwise-valid accumulators:
        # - N(i,j): number of rows where both entries are present
        # - Sx(i,j): sum of x_i over rows where y_j is present
        # - Sy(i,j): sum of y_j over rows where x_i is present
        # - Sxy(i,j): sum of x_i*y_j over rows where both are present
        self.N += M_i.T @ M_j
        self.Sx += X_i.T @ M_j
        self.Sy += M_i.T @ X_j
        self.Sxy += X_i.T @ X_j

    def finalize(self) -> pd.DataFrame:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_x = self.Sx / self.N
            mean_y = self.Sy / self.N
            C = (self.Sxy / self.N) - (mean_x * mean_y)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(C, index=self.cols_i, columns=self.cols_j)



class DNNBatch:
    def __init__(self, cfg, df_inputs: pd.DataFrame, df_targets: pd.DataFrame, df_weights: Optional[pd.DataFrame] = None):
        self.cfg = cfg
        self.df_inputs = df_inputs
        self.df_targets = df_targets
        self.df_weights = df_weights

    @property
    def full_inputs_df(self) -> pd.DataFrame:
        return self.df_inputs

    @property
    def full_targets_df(self) -> pd.DataFrame:
        return self.df_targets

class DNNBatchIter:
    def __init__(self, cfg, require_weights: bool = False):
        self.cfg = cfg
        self.require_weights = require_weights
        base = cfg.dnn_training_input_folder

        self.inputfiles   = sorted(glob(os.path.join(base, "inputs_chunk*.parquet")))
        self.targetfiles  = sorted(glob(os.path.join(base, "targets_chunk*.parquet")))
        self.weightfiles  = sorted(glob(os.path.join(base, "weights_chunk*.parquet")))

        if not self.inputfiles:
            raise RuntimeError(f"No DNN input chunks found in {base} (inputs_chunk*.parquet)")
        if not (len(self.inputfiles) == len(self.targetfiles)):
            raise RuntimeError(f"Mismatch chunk counts in {base}: inputs={len(self.inputfiles)} targets={len(self.targetfiles)}")
        if self.require_weights and len(self.weightfiles) != len(self.inputfiles):
            raise RuntimeError(
                f"Requested DNN sample weights, but found inputs={len(self.inputfiles)} and weights={len(self.weightfiles)} in {base}. "
                "Rerun prepare_dnn_inputs.py to create weights_chunk*.parquet sidecars."
            )

    def __iter__(self):
        for idx, (fin, ftg) in enumerate(zip(self.inputfiles, self.targetfiles)):
            df_inputs  = pd.read_parquet(fin)
            df_targets = pd.read_parquet(ftg)
            df_weights = None
            if self.require_weights:
                df_weights = pd.read_parquet(self.weightfiles[idx])
                if df_weights.shape != df_targets.shape:
                    raise ValueError(f"DNN weight shape mismatch for {self.weightfiles[idx]}: weights={df_weights.shape}, targets={df_targets.shape}")
                if not df_weights.index.equals(df_targets.index) or list(df_weights.columns) != list(df_targets.columns):
                    raise ValueError(f"DNN weights in {self.weightfiles[idx]} do not match target index/columns.")

            yield DNNBatch(cfg=self.cfg, df_inputs=df_inputs, df_targets=df_targets, df_weights=df_weights) 
