import os
import json
import subprocess
import tempfile
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from fnmatch import fnmatch
from scipy.optimize import curve_fit

import utils
import classes


def _coolwarm_palette_payload():
    cmap = plt.get_cmap("coolwarm")
    stops = np.linspace(0.0, 1.0, 255)
    colors = cmap(stops)
    return {
        "stops": stops.astype(float).tolist(),
        "red": colors[:, 0].astype(float).tolist(),
        "green": colors[:, 1].astype(float).tolist(),
        "blue": colors[:, 2].astype(float).tolist(),
    }


def _draw_wafer_hist_pdf(values, output_filename: str, module_type: str, ztitle: str, title: str, zrange=None, labels=None) -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cmssw_src = os.path.abspath(os.path.join(repo_root, "..", ".."))
    repo_python = os.path.join(repo_root, "python")
    root_python = os.environ.get("ROOT_PYTHON", "/usr/bin/python3")
    output_filename = os.path.abspath(output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    payload = {
        "values": list(values),
        "output_filename": output_filename,
        "module_type": module_type,
        "ztitle": ztitle,
        "title": title,
        "zrange": list(zrange) if zrange is not None else None,
        "labels": list(labels) if labels is not None else None,
        "palette": _coolwarm_palette_payload(),
    }
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [cmssw_src, repo_python]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    helper_script = os.path.join(os.path.dirname(__file__), "render_wafer_hist.py")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="wafer_payload_", dir="/tmp", delete=False) as payload_file:
        json.dump(payload, payload_file)
        payload_path = payload_file.name
    try:
        subprocess.run([root_python, helper_script, "--payload", payload_path], cwd=repo_root, env=env, check=True)
    finally:
        if os.path.exists(payload_path):
            os.unlink(payload_path)
    print(f"Saved wafer plot {output_filename}")



def plot_cov_corr(cfg: classes.AnalysisConfig, column_tag, axis_title, zrange_cov, plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)

    cov = cfg.load_from_cov_folder(filename=f"sigma_mcmc{column_tag}.parquet")
    
    # fraction of off-diag correlation over all correlations
    f_corr_pred = utils.f_corr_from_cov(cov)
    
    utils.plot_covariance(df=cov, nch_per_erx=cfg.nch_per_erx,                       title=f"Covariance ({axis_title}) - f_corr = {f_corr_pred}", xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)",  zrange=zrange_cov, output_filename=os.path.join(plot_dir, f"Covariance{column_tag}.pdf"))
    utils.plot_covariance(df=utils.corr_from_cov(cov), nch_per_erx=cfg.nch_per_erx, title=f"Correlation ({axis_title}) - f_corr = {f_corr_pred}", xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)", zrange=(-1., 1.), output_filename=os.path.join(plot_dir, f"Correlation{column_tag}.pdf"))


def plot_noise_model_fit(cfg: classes.AnalysisConfig, column_tag, zrange_cov, plot_dir: str, n_coherent: int = 1) -> None:
    os.makedirs(plot_dir, exist_ok=True)

    fit_tag = f"{column_tag}_nc{n_coherent}"
    summary_filename = f"noise_model_summary_mm{fit_tag}.parquet"
    summary_path = os.path.join(cfg.noise_model_fit_folder, summary_filename)
    if not os.path.exists(summary_path):
        print(f"[info] Skipping noise-model plots because fit file is missing: {summary_path}")
        return

    summary = cfg.load_from_noise_model_fit_folder(summary_filename).iloc[0]
    channels = cfg.load_from_noise_model_fit_folder(f"noise_model_channels_mm{fit_tag}.parquet")
    loadings = cfg.load_from_noise_model_fit_folder(f"noise_model_loadings_mm{fit_tag}.parquet")
    sigma_model = cfg.load_from_noise_model_fit_folder(f"sigma_mm_model{fit_tag}.parquet")
    sigma_coherent = cfg.load_from_noise_model_fit_folder(f"sigma_mm_coherent{fit_tag}.parquet")
    sigma_incoherent = cfg.load_from_noise_model_fit_folder(f"sigma_mm_incoherent{fit_tag}.parquet")
    sigma_residual = cfg.load_from_noise_model_fit_folder(f"sigma_mm_fitresidual{fit_tag}.parquet")

    title_suffix = (
        f"{n_coherent} coherent source"
        f"{'' if n_coherent == 1 else 's'}"
        f" | MSE = {summary['mse']:.3g}"
        f" | coherent fraction = {summary['coherent_fraction_trace']:.3g}"
    )
    filename_suffix = f"{column_tag}_nc{n_coherent}"

    utils.plot_covariance(
        df=sigma_model,
        nch_per_erx=cfg.nch_per_erx,
        title=f"Noise-Model Covariance ({title_suffix})",
        xtitle="channel i",
        ytitle="channel j",
        ztitle="cov(i,j)",
        zrange=zrange_cov,
        output_filename=os.path.join(plot_dir, f"NoiseModelCovariance{filename_suffix}.pdf"),
    )
    utils.plot_covariance(
        df=sigma_coherent,
        nch_per_erx=cfg.nch_per_erx,
        title=f"Coherent Covariance ({title_suffix})",
        xtitle="channel i",
        ytitle="channel j",
        ztitle="cov(i,j)",
        zrange=zrange_cov,
        output_filename=os.path.join(plot_dir, f"NoiseModelCoherent{filename_suffix}.pdf"),
    )
    utils.plot_covariance(
        df=sigma_incoherent,
        nch_per_erx=cfg.nch_per_erx,
        title=f"Incoherent Covariance ({title_suffix})",
        xtitle="channel i",
        ytitle="channel j",
        ztitle="cov(i,j)",
        zrange=zrange_cov,
        output_filename=os.path.join(plot_dir, f"NoiseModelIncoherent{filename_suffix}.pdf"),
    )
    utils.plot_covariance(
        df=sigma_residual,
        nch_per_erx=cfg.nch_per_erx,
        title=f"Fit-Residual Covariance ({title_suffix})",
        xtitle="channel i",
        ytitle="channel j",
        ztitle="cov(i,j)",
        zrange=zrange_cov,
        output_filename=os.path.join(plot_dir, f"NoiseModelFitResidual{filename_suffix}.pdf"),
    )

    x = np.arange(channels.shape[0])
    sigma_inc = np.sqrt(np.maximum(channels["incoherent_var"].to_numpy(dtype=float), 0.0))
    sigma_coh = np.sqrt(np.maximum(channels["coherent_var"].to_numpy(dtype=float), 0.0))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x, channels["data_diag"].to_numpy(), label="data diag", lw=1.5)
    ax.plot(x, channels["model_diag"].to_numpy(), label="model diag", lw=1.5)
    ax.plot(x, channels["incoherent_var"].to_numpy(), label="incoherent var", lw=1.5)
    ax.plot(x, channels["coherent_var"].to_numpy(), label="coherent var", lw=1.5)
    for pos in range(0, cfg.nch_per_erx * (cfg.nerx + 1), cfg.nch_per_erx):
        ax.axvline(pos, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("channel")
    ax.set_ylabel("variance")
    ax.grid(ls="--", alpha=0.3)
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    outfilename = os.path.join(plot_dir, f"NoiseModelChannelVariances{filename_suffix}.pdf")
    fig.savefig(outfilename)
    print(f"--> Plotted noise-model channel variances: {outfilename}")
    plt.close(fig)

    if column_tag != "":
        raw_fit_tag = f"_nc{n_coherent}"
        raw_channels_filename = f"noise_model_channels_mm{raw_fit_tag}.parquet"
        raw_channels_path = os.path.join(cfg.noise_model_fit_folder, raw_channels_filename)
        if os.path.exists(raw_channels_path):
            raw_channels = cfg.load_from_noise_model_fit_folder(raw_channels_filename)
            raw_sigma_inc = np.sqrt(np.maximum(raw_channels["incoherent_var"].to_numpy(dtype=float), 0.0))
            raw_sigma_coh = np.sqrt(np.maximum(raw_channels["coherent_var"].to_numpy(dtype=float), 0.0))
            inc_ratio = np.nan_to_num(sigma_inc / np.maximum(raw_sigma_inc, 1e-12), nan=0.0, posinf=0.0, neginf=0.0)
            coh_ratio = np.nan_to_num(sigma_coh / np.maximum(raw_sigma_coh, 1e-12), nan=0.0, posinf=0.0, neginf=0.0)
            coh_over_inc_true = np.nan_to_num(raw_sigma_coh / np.maximum(raw_sigma_inc, 1e-12), nan=0.0, posinf=0.0, neginf=0.0)
            coh_over_inc_corr = np.nan_to_num(sigma_coh / np.maximum(sigma_inc, 1e-12), nan=0.0, posinf=0.0, neginf=0.0)

            fig = plt.figure(figsize=(8.2, 6.2))
            gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.10)
            ax1 = fig.add_subplot(gs[0])
            axr = fig.add_subplot(gs[1], sharex=ax1)
            axc = fig.add_subplot(gs[2], sharex=ax1)

            ax1.plot(x, raw_sigma_inc, "-",  label="incoherent (meas.)", color="tab:blue")
            ax1.plot(x, raw_sigma_coh, "-",  label="coherent (meas.)",   color="tab:orange")
            ax1.plot(x, sigma_inc,     "--", label="incoherent (corr.)", color="tab:blue")
            ax1.plot(x, sigma_coh,     "--", label="coherent (corr.)",   color="tab:orange")

            for ax_this in (ax1, axr, axc):
                ax_this.tick_params(axis="both", direction="in", top=True, bottom=True, left=True, right=True, labelsize=12)
                ax_this.grid(ls="--", alpha=0.3)
                for pos in range(0, cfg.nch_per_erx * (cfg.nerx + 1), cfg.nch_per_erx):
                    ax_this.axvline(pos, color="black", linestyle="--", linewidth=1, alpha=0.35)

            ax1.set_ylabel("Noise (ADC)", fontsize=16, loc="top", labelpad=12)
            ax1.set_ylim(0., ax1.get_ylim()[1] * 1.2)
            ax1.legend(loc="upper right", fontsize=12)

            axr.plot(x, inc_ratio, "--", color="tab:blue")
            axr.plot(x, coh_ratio, "--", color="tab:orange")
            axr.set_ylabel("corr. / meas.", fontsize=11, loc="center", labelpad=10)
            axr.set_ylim(0., 1.1)

            axc.plot(x, coh_over_inc_true, "-",  color="black")
            axc.plot(x, coh_over_inc_corr, "--", color="black")
            axc.set_xlabel("channel", fontsize=16, loc="right", labelpad=8)
            axc.set_ylabel("coh. / inc.", fontsize=11, loc="center", labelpad=8)
            axc.set_ylim(0., max(axc.get_ylim()[1], 2.))

            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(axr.get_xticklabels(), visible=False)

            outfilename = os.path.join(plot_dir, f"NoiseModelCohIncRatio{filename_suffix}.pdf")
            fig.savefig(outfilename, bbox_inches="tight", pad_inches=0.05)
            print(f"--> Plotted noise-model coh/inc comparison: {outfilename}")
            plt.close(fig)

    if loadings.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        for col in loadings.columns:
            ax.plot(x, loadings[col].to_numpy(), label=col, lw=1.5)
        for pos in range(0, cfg.nch_per_erx * (cfg.nerx + 1), cfg.nch_per_erx):
            ax.axvline(pos, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("channel")
        ax.set_ylabel("loading")
        ax.grid(ls="--", alpha=0.3)
        ax.legend(fontsize="small")
        fig.tight_layout()
        outfilename = os.path.join(plot_dir, f"NoiseModelLoadings{filename_suffix}.pdf")
        fig.savefig(outfilename)
        print(f"--> Plotted noise-model loadings: {outfilename}")
        plt.close(fig)



class Streaming1DHist:
    def __init__(self, x_min: float, x_max: float, nbins_x: int = None):
        if nbins_x:
            self.nxb = nbins_x
        else:
            self.nxb = utils.round_nearest(x_max-x_min)
        
        self.x_edges = np.linspace(x_min, x_max, self.nxb + 1, dtype=np.float64)

        self.H = np.zeros(self.nxb, dtype=np.int64)
        self.x_count = 0
        self.x_sum   = 0.
        self.x_sum2  = 0.
        self.xw_count = 0
        self.xw_sum   = 0.
        self.xw_sum2  = 0.

    def add(self, x: np.ndarray):

        x = x[~np.isnan(x)]

        # Bin indices
        xi = np.searchsorted(self.x_edges, x, side="right") - 1

        valid = (xi >= 0) & (xi < self.nxb)
        if not np.any(valid):
            return

        xi = xi[valid]

        # 1D counts
        np.add.at(self.H, xi, 1)

        # x-profile stats
        self.x_count += x.shape[0]
        self.x_sum += np.sum(x)
        self.x_sum2 += np.sum(x*x)

        # window-restricted stats (-10, 10)
        w_mask = (x >= -10) & (x <= 10)
        xw = x[w_mask]
        self.xw_count += xw.shape[0]
        self.xw_sum   += np.sum(xw)
        self.xw_sum2  += np.sum(xw * xw)

    def x_mean_rms(self):
        if self.x_count == 0:
            return np.nan, np.nan
        mean = self.x_sum / self.x_count
        rms = np.sqrt(self.x_sum2 / self.x_count - mean**2)
        return mean, rms
    
    def x_mean_rms_window(self):
        if self.xw_count == 0:
            return np.nan, np.nan
        mean = self.xw_sum / self.xw_count
        rms = np.sqrt(self.xw_sum2 / self.xw_count - mean**2)
        return mean, rms


class Streaming2DHist:
    """
    Streamed 2D histogram for y vs x with marginals + x-profile.
    Keeps:
      H[xbin, ybin], x_marg (sum over y), y_marg (sum over x),
      and the x-profile (sum_y, sum_y2, count per xbin).
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, nbins_x: int = None, nbins_y: int = 80):
        if nbins_x:
            self.nxb = nbins_x
        else:
            self.nxb = utils.round_nearest(x_max-x_min)
        
        self.nyb = nbins_y
        self.x_edges = np.linspace(x_min, x_max, self.nxb + 1, dtype=np.float64)
        self.y_edges = np.linspace(y_min, y_max, self.nyb + 1, dtype=np.float64)

        self.H = np.zeros((self.nxb, self.nyb), dtype=np.int64)
        self.x_count = np.zeros(self.nxb, dtype=np.int64)
        self.x_sum   = np.zeros(self.nxb, dtype=np.float64)
        self.x_sum2  = np.zeros(self.nxb, dtype=np.float64)

    def add(self, x: np.ndarray, y: np.ndarray):
        # Bin indices
        xi = np.searchsorted(self.x_edges, x, side="right") - 1
        yi = np.searchsorted(self.y_edges, y, side="right") - 1

        valid = (xi >= 0) & (xi < self.nxb) & (yi >= 0) & (yi < self.nyb)
        if not np.any(valid):
            return

        xi = xi[valid]
        yi = yi[valid]
        yv = y[valid]

        # 2D counts
        np.add.at(self.H, (xi, yi), 1)

        # x-profile stats
        np.add.at(self.x_count, xi, 1)
        np.add.at(self.x_sum,   xi, yv)
        np.add.at(self.x_sum2,  xi, yv * yv)

    def x_profile(self):
        centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        mean = np.full(self.nxb, np.nan, dtype=np.float64)
        rms  = np.full(self.nxb, np.nan, dtype=np.float64)
        nz = self.x_count > 0
        mean[nz] = self.x_sum[nz] / self.x_count[nz]
        var = np.zeros_like(mean)
        var[nz] = (self.x_sum2[nz] / self.x_count[nz]) - mean[nz]**2
        var[var < 0] = 0.0
        rms[nz] = np.sqrt(var[nz])
        return centers, mean, rms

def plot_vs_chidx(cfg, varname_y_template: str, value_iterator, out_root: str, nbins_x: int = 80, nbins_y: int = 80, y_range: tuple[float,float] = (-20., 20.)):
    os.makedirs(out_root, exist_ok=True)

    chidx_min = 0
    chidx_max = cfg.nerx * cfg.nch_per_erx - 1

    # 3) streamed passes
    # --- PRED ---
    hist_pred = Streaming2DHist(x_min=chidx_min, x_max=chidx_max, y_min=y_range[0], y_max=y_range[1], nbins_x=nbins_x, nbins_y=nbins_y)
    for full_df in value_iterator():
        cols_y = [c for c in full_df.columns if fnmatch(c, varname_y_template)]
        if not cols_y:
            print(f"[WARNING] No columns found that match the template {varname_y_template}. Skipping.")
            continue
        
        x = np.tile(np.arange(chidx_min, chidx_max+1), full_df[cols_y].shape[0])
        y = full_df[cols_y].to_numpy().ravel()
        hist_pred.add(x, y)
    x_ct_p, mean_p, rms_p = hist_pred.x_profile()
    print(f"Maximum avg-per-channel-adc vs. channel index: {np.nanmax(mean_p)} at {x_ct_p[np.nanargmax(mean_p)]}")

    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_pred.H, x_edges=hist_pred.x_edges, y_edges=hist_pred.y_edges,
        x_prof_centers=x_ct_p, x_prof_mean=mean_p,
        label_x=f"channel index", label_y=f"{varname_y_template}", label_profile="profile",
        output_filename=os.path.join(out_root, f"{varname_y_template.replace('*', 'all').replace('?', '')}_vs_chidx.pdf"),
    )

def plot_2d_multicol_vs_var(varname_x: str, varname_y_template: str, value_iterator, out_root: str, nbins_x: int = None, x_range=None, nbins_y: int = 80, y_range: tuple[float,float] = (-20., 20.), make_profile_plot=False):
    os.makedirs(out_root, exist_ok=True)

    if x_range:
        var_min, var_max = x_range
    else:
        first = True
        for df in value_iterator():
            vals = df[varname_x]
            if len(vals) == 0:
                continue
            if first:
                var_min = min(vals)
                var_max = max(vals)
            else:
                var_min = min(min(vals), var_min)
                var_max = max(max(vals), var_max)
            first = False

    # 3) streamed passes
    # --- PRED ---
    hist_pred = Streaming2DHist(x_min=var_min, x_max=var_max, y_min=y_range[0], y_max=y_range[1], nbins_x=nbins_x, nbins_y=nbins_y)
    for full_df in value_iterator():
        cols_y = [c for c in full_df.columns if fnmatch(c, varname_y_template)]
        if not cols_y:
            print(f"[WARNING] No columns found that match the template {varname_y_template}. Skipping.")
            continue
        x = full_df[varname_x].to_numpy()
        y = full_df[cols_y].to_numpy()
        x_rep = np.repeat(x, len(cols_y))      # (N*M,)
        y_flat = y.ravel()                     # (N*M,)

        hist_pred.add(x_rep, y_flat)
    x_ct_p, mean_p, rms_p = hist_pred.x_profile()
    if np.all(np.isnan(mean_p)):
        print(f"Means are all NaN for {varname_y_template} vs. {varname_x}. Skipping this combination.")
        return
    print(f"Maximum of {varname_y_template} vs. {varname_x}: {np.nanmax(mean_p)} at {x_ct_p[np.nanargmax(mean_p)]}")


    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_pred.H, x_edges=hist_pred.x_edges, y_edges=hist_pred.y_edges,
        x_prof_centers=x_ct_p, x_prof_mean=mean_p,
        label_x=varname_x, label_y=varname_y_template, label_profile="profile",
        output_filename=os.path.join(out_root, f"{varname_y_template.replace('*', 'all').replace('?', '')}_vs_{varname_x}.pdf"), 
        make_profile_plot=make_profile_plot,
    )

def plot_2d_multicol_vs_multicol(varname_x_template: str, varname_y_template: str, value_iterator, out_root: str, nbins_x: int = None, x_range=None, nbins_y: int = 80, y_range: tuple[float,float] = (-20., 20.), make_profile_plot=False):
    os.makedirs(out_root, exist_ok=True)

    cols_x = None
    cols_y = None
    first = True
    var_min = None
    var_max = None

    for full_df in value_iterator():
        matched_x = [c for c in full_df.columns if fnmatch(c, varname_x_template)]
        matched_y = [c for c in full_df.columns if fnmatch(c, varname_y_template)]
        if not matched_x:
            print(f"[WARNING] No columns found that match the x template {varname_x_template}. Skipping.")
            return
        if not matched_y:
            print(f"[WARNING] No columns found that match the y template {varname_y_template}. Skipping.")
            return
        if len(matched_x) != len(matched_y):
            raise ValueError(
                f"Mismatched number of columns for paired 2D plot: "
                f"{varname_x_template} matched {len(matched_x)} columns, "
                f"{varname_y_template} matched {len(matched_y)} columns."
            )

        if cols_x is None:
            cols_x = matched_x
            cols_y = matched_y
        else:
            if matched_x != cols_x:
                raise ValueError(
                    f"Matched x columns changed across chunks for template {varname_x_template}. "
                    f"Expected {cols_x}, found {matched_x}."
                )
            if matched_y != cols_y:
                raise ValueError(
                    f"Matched y columns changed across chunks for template {varname_y_template}. "
                    f"Expected {cols_y}, found {matched_y}."
                )

        if x_range is not None:
            continue

        vals = full_df[cols_x].to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        if first:
            var_min = np.min(vals)
            var_max = np.max(vals)
            first = False
        else:
            var_min = min(np.min(vals), var_min)
            var_max = max(np.max(vals), var_max)

    if cols_x is None or cols_y is None:
        print(
            f"[WARNING] Could not resolve paired columns for templates "
            f"{varname_x_template} and {varname_y_template}. Skipping."
        )
        return

    if x_range is not None:
        var_min, var_max = x_range
    elif first:
        print(
            f"[WARNING] No valid x values found for paired 2D plot "
            f"{varname_y_template} vs. {varname_x_template}. Skipping."
        )
        return

    hist_pred = Streaming2DHist(x_min=var_min, x_max=var_max, y_min=y_range[0], y_max=y_range[1], nbins_x=nbins_x, nbins_y=nbins_y)
    for full_df in value_iterator():
        x = full_df[cols_x].to_numpy().ravel()
        y = full_df[cols_y].to_numpy().ravel()
        hist_pred.add(x, y)

    x_ct_p, mean_p, rms_p = hist_pred.x_profile()
    if np.all(np.isnan(mean_p)):
        print(f"Means are all NaN for {varname_y_template} vs. {varname_x_template}. Skipping this combination.")
        return
    print(f"Maximum of {varname_y_template} vs. {varname_x_template}: {np.nanmax(mean_p)} at {x_ct_p[np.nanargmax(mean_p)]}")

    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_pred.H, x_edges=hist_pred.x_edges, y_edges=hist_pred.y_edges,
        x_prof_centers=x_ct_p, x_prof_mean=mean_p,
        label_x=varname_x_template, label_y=varname_y_template, label_profile="profile",
        output_filename=os.path.join(out_root, f"{varname_y_template.replace('*', 'all').replace('?', '')}_vs_{varname_x_template.replace('*', 'all').replace('?', '')}.pdf"),
        make_profile_plot=make_profile_plot,
    )


def _sorted_channel_values(arr: np.ndarray, descending: bool = False) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")

    finite = np.isfinite(arr)
    fill_value = -np.inf if descending else np.inf
    work = np.where(finite, arr, fill_value)
    sorted_vals = np.sort(work, axis=1)
    if descending:
        sorted_vals = sorted_vals[:, ::-1]

    counts = finite.sum(axis=1)
    ranks = np.arange(arr.shape[1])[None, :]
    valid_positions = ranks < counts[:, None]
    sorted_vals = np.where(valid_positions, sorted_vals, np.nan)
    return sorted_vals


def _add_sorted_values_to_hist(hist: Streaming2DHist, values: np.ndarray) -> None:
    if values.size == 0:
        return
    ranks = np.broadcast_to(np.arange(values.shape[1], dtype=np.float32), values.shape)
    mask = np.isfinite(values)
    if not np.any(mask):
        return
    hist.add(ranks[mask], values[mask])


def _accumulate_profile(sum_arr: np.ndarray, count_arr: np.ndarray, values: np.ndarray) -> None:
    mask = np.isfinite(values)
    sum_arr += np.nansum(values, axis=0)
    count_arr += np.sum(mask, axis=0)


def plot_sorted_channel_contributions(
    varname_true_template: str,
    varname_corr_template: str,
    value_iterator,
    out_root: str,
    value_range: tuple[float, float] = (-30., 80.),
    cumsum_range: tuple[float, float] = (-800., 1200.),
    nbins_y: int = 120,
) -> None:
    os.makedirs(out_root, exist_ok=True)

    cols_true = None
    cols_corr = None
    n_channels = None
    for full_df in value_iterator():
        matched_true = [c for c in full_df.columns if fnmatch(c, varname_true_template)]
        matched_corr = [c for c in full_df.columns if fnmatch(c, varname_corr_template)]
        if not matched_true or not matched_corr:
            continue
        if len(matched_true) != len(matched_corr):
            raise ValueError(
                f"Mismatched number of columns for sorted contributions: "
                f"{varname_true_template} -> {len(matched_true)}, "
                f"{varname_corr_template} -> {len(matched_corr)}"
            )
        cols_true = matched_true
        cols_corr = matched_corr
        n_channels = len(cols_true)
        break

    if cols_true is None or cols_corr is None or n_channels is None:
        print(
            f"[WARNING] Could not resolve columns for sorted channel contributions with "
            f"{varname_true_template} and {varname_corr_template}. Skipping."
        )
        return

    hist_true = Streaming2DHist(x_min=-0.5, x_max=n_channels - 0.5, y_min=value_range[0], y_max=value_range[1], nbins_x=n_channels, nbins_y=nbins_y)
    hist_corr = Streaming2DHist(x_min=-0.5, x_max=n_channels - 0.5, y_min=value_range[0], y_max=value_range[1], nbins_x=n_channels, nbins_y=nbins_y)
    hist_cumsum_true = Streaming2DHist(x_min=-0.5, x_max=n_channels - 0.5, y_min=cumsum_range[0], y_max=cumsum_range[1], nbins_x=n_channels, nbins_y=nbins_y)
    hist_cumsum_corr = Streaming2DHist(x_min=-0.5, x_max=n_channels - 0.5, y_min=cumsum_range[0], y_max=cumsum_range[1], nbins_x=n_channels, nbins_y=nbins_y)

    sum_sorted_true = np.zeros(n_channels, dtype=np.float64)
    sum_sorted_corr = np.zeros(n_channels, dtype=np.float64)
    count_sorted_true = np.zeros(n_channels, dtype=np.int64)
    count_sorted_corr = np.zeros(n_channels, dtype=np.int64)
    sum_cumsum_true = np.zeros(n_channels, dtype=np.float64)
    sum_cumsum_corr = np.zeros(n_channels, dtype=np.float64)
    count_cumsum_true = np.zeros(n_channels, dtype=np.int64)
    count_cumsum_corr = np.zeros(n_channels, dtype=np.int64)

    for full_df in value_iterator():
        true_vals = full_df[cols_true].to_numpy(dtype=np.float32, copy=False)
        corr_vals = full_df[cols_corr].to_numpy(dtype=np.float32, copy=False)

        sorted_true = _sorted_channel_values(true_vals, descending=False)
        sorted_corr = _sorted_channel_values(corr_vals, descending=False)

        _add_sorted_values_to_hist(hist_true, sorted_true)
        _add_sorted_values_to_hist(hist_corr, sorted_corr)
        _accumulate_profile(sum_sorted_true, count_sorted_true, sorted_true)
        _accumulate_profile(sum_sorted_corr, count_sorted_corr, sorted_corr)

        cumsum_true = np.nancumsum(np.nan_to_num(sorted_true, nan=0.0), axis=1)
        cumsum_corr = np.nancumsum(np.nan_to_num(sorted_corr, nan=0.0), axis=1)

        _add_sorted_values_to_hist(hist_cumsum_true, cumsum_true)
        _add_sorted_values_to_hist(hist_cumsum_corr, cumsum_corr)
        _accumulate_profile(sum_cumsum_true, count_cumsum_true, cumsum_true)
        _accumulate_profile(sum_cumsum_corr, count_cumsum_corr, cumsum_corr)

    ranks = np.arange(n_channels, dtype=float)
    mean_sorted_true = np.divide(sum_sorted_true, count_sorted_true, out=np.full(n_channels, np.nan), where=count_sorted_true > 0)
    mean_sorted_corr = np.divide(sum_sorted_corr, count_sorted_corr, out=np.full(n_channels, np.nan), where=count_sorted_corr > 0)
    mean_cumsum_true = np.divide(sum_cumsum_true, count_cumsum_true, out=np.full(n_channels, np.nan), where=count_cumsum_true > 0)
    mean_cumsum_corr = np.divide(sum_cumsum_corr, count_cumsum_corr, out=np.full(n_channels, np.nan), where=count_cumsum_corr > 0)

    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_true.H,
        x_edges=hist_true.x_edges,
        y_edges=hist_true.y_edges,
        x_prof_centers=ranks,
        x_prof_mean=mean_sorted_true,
        label_x="sorted channel rank",
        label_y=varname_true_template,
        label_profile="profile",
        output_filename=os.path.join(out_root, f"{varname_true_template.replace('*', 'all').replace('?', '')}_sorted_vs_rank.pdf"),
    )
    utils.plot_y_vs_x_with_marginals_hist(
        H=hist_corr.H,
        x_edges=hist_corr.x_edges,
        y_edges=hist_corr.y_edges,
        x_prof_centers=ranks,
        x_prof_mean=mean_sorted_corr,
        label_x="sorted channel rank",
        label_y=varname_corr_template,
        label_profile="profile",
        output_filename=os.path.join(out_root, f"{varname_corr_template.replace('*', 'all').replace('?', '')}_sorted_vs_rank.pdf"),
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(ranks, mean_sorted_true, color="gray", label="Measured", lw=2.0)
    ax.plot(ranks, mean_sorted_corr, color="red", label="Corrected", lw=2.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("sorted channel rank")
    ax.set_ylabel("channel contribution")
    ax.grid(ls="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "sorted_channel_contributions_overlay.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(ranks, mean_cumsum_true, color="gray", label="Measured", lw=2.0)
    ax.plot(ranks, mean_cumsum_corr, color="red", label="Corrected", lw=2.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("sorted channel rank")
    ax.set_ylabel("cumulative sum")
    ax.grid(ls="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "sorted_channel_cumsum_overlay.pdf"))
    plt.close(fig)


def plot_module_hexmap(
    cfg,
    varname_template: str,
    value_iterator,
    out_root: str,
    title_suffix: str = "",
    cmap: str = "coolwarm",
    symmetric: bool = False,
) -> None:
    os.makedirs(out_root, exist_ok=True)

    val_cols = None
    erx_cols = [f"erx_ch{i:03d}" for i in range(cfg.nch)]
    erx_indices = None
    sum_vals = None
    count_vals = None

    for full_df in value_iterator():
        matched = [c for c in full_df.columns if fnmatch(c, varname_template)]
        if not matched:
            continue
        if val_cols is None:
            val_cols = matched
            erx_indices = np.full(cfg.nch, np.nan, dtype=float)
            sum_vals = np.zeros(len(val_cols), dtype=np.float64)
            count_vals = np.zeros(len(val_cols), dtype=np.int64)
        elif matched != val_cols:
            raise ValueError(
                f"Matched value columns changed across chunks for template {varname_template}. "
                f"Expected {val_cols}, found {matched}."
            )

        erx_chunk = full_df[erx_cols].to_numpy(dtype=np.float32, copy=False)
        for ch in range(cfg.nch):
            if np.isfinite(erx_indices[ch]):
                continue
            finite_erx = np.isfinite(erx_chunk[:, ch])
            if np.any(finite_erx):
                first = int(np.flatnonzero(finite_erx)[0])
                erx_indices[ch] = float(erx_chunk[first, ch])

        vals = full_df[val_cols].to_numpy(dtype=np.float32, copy=False)
        mask = np.isfinite(vals)
        sum_vals += np.nansum(vals, axis=0)
        count_vals += np.sum(mask, axis=0)

    if val_cols is None or erx_indices is None or sum_vals is None or count_vals is None:
        print(f"[WARNING] No columns found that match the template {varname_template}. Skipping module hexmap.")
        return

    mean_vals = np.divide(sum_vals, count_vals, out=np.full_like(sum_vals, np.nan), where=count_vals > 0)
    sanitized = varname_template.replace("*", "all").replace("?", "")
    module_type = cfg.modulename[:4].replace("-", "_")

    _draw_wafer_hist_pdf(
        values=np.nan_to_num(erx_indices, nan=-1.0).astype(float),
        output_filename=os.path.join(out_root, "channel_index_map.pdf"),
        module_type=module_type,
        ztitle="eRx index",
        title=f"Channel index map colored by eRx{title_suffix}",
        zrange=(-0.5, cfg.nerx - 0.5),
        labels=[str(i) for i in range(cfg.nch)],
    )

    finite_mean = mean_vals[np.isfinite(mean_vals)]
    if finite_mean.size == 0:
        zrange = (-1.0, 1.0)
    else:
        zmin = float(np.nanmin(finite_mean))
        zmax = float(np.nanmax(finite_mean))
        if symmetric:
            vmax = max(abs(zmin), abs(zmax))
            zrange = (-vmax, vmax)
        elif np.isclose(zmin, zmax):
            delta = 1.0 if np.isclose(zmin, 0.0) else 0.1 * abs(zmin)
            zrange = (zmin - delta, zmax + delta)
        else:
            zrange = (zmin, zmax)

    _draw_wafer_hist_pdf(
        values=np.nan_to_num(mean_vals, nan=0.0).astype(float),
        output_filename=os.path.join(out_root, f"{sanitized}_mean_hexmap.pdf"),
        module_type=module_type,
        ztitle=f"mean {sanitized}",
        title=f"Mean {sanitized}{title_suffix}",
        zrange=zrange,
        labels=None,
    )

def plot_1d_multicol(varname_template: str, value_iterator, out_root: str, nbins_x: int = None, x_range=None, do_gauss_fit=False, gauss_p0=None):
    os.makedirs(out_root, exist_ok=True)
        
    hist = Streaming1DHist(x_min=x_range[0], x_max=x_range[1], nbins_x=nbins_x)
    for full_df in value_iterator():
        cols = [c for c in full_df.columns if fnmatch(c, varname_template)]
        values = full_df[cols].to_numpy().ravel()
        hist.add(values)
    outpath = os.path.join(out_root, f"{varname_template.replace('*', 'all').replace('?', '')}_1d.pdf")
    if do_gauss_fit:
        outpath = outpath.replace("_1d.pdf", "_gaussfit_1d.pdf")
    utils.plot_hist_single_precomputed(x=hist.H, mean=hist.x_mean_rms()[0], rms=hist.x_mean_rms()[1], mean_window=hist.x_mean_rms_window()[0], rms_window=hist.x_mean_rms_window()[1], bins=hist.x_edges, xlabel=f"{varname_template}", ylabel="Number of events", title="", color="gray", outpath=outpath, show_mean_line=True, do_gauss_fit=do_gauss_fit, gauss_p0=gauss_p0)



def plot_summary_1d_multicol(varname_template: str, varname_true_template: str, value_iterator, out_root: str, nbins_x: int = None, x_range=None):
    os.makedirs(out_root, exist_ok=True)
        
    cols = []
    cols_true = []
    for full_df in value_iterator():
        cols = [c for c in full_df.columns if fnmatch(c, varname_template)]
        cols_true = [c for c in full_df.columns if fnmatch(c, varname_true_template)]
        break

    hists = {c: Streaming1DHist(x_min=x_range[0], x_max=x_range[1], nbins_x=nbins_x) for c in cols}
    hists_true = {c: Streaming1DHist(x_min=x_range[0], x_max=x_range[1], nbins_x=nbins_x) for c in cols_true}

    for full_df in value_iterator():
        values = full_df[cols].to_numpy()
        values_true = full_df[cols_true].to_numpy()
        for colidx, col in enumerate(cols):
            hists[col].add(values[:, colidx])
        for colidx_true, col_true in enumerate(cols_true):
            hists_true[col_true].add(values_true[:, colidx_true])

    means = [hists[col].x_mean_rms()[0] for col in cols]
    sigmas = [hists[col].x_mean_rms()[1] for col in cols]
    means_window = [hists[col].x_mean_rms_window()[0] for col in cols]
    sigmas_window = [hists[col].x_mean_rms_window()[1] for col in cols]

    means_true = [hists_true[col].x_mean_rms()[0] for col in cols_true]
    sigmas_true = [hists_true[col].x_mean_rms()[1] for col in cols_true]
    means_true_window = [hists_true[col].x_mean_rms_window()[0] for col in cols_true]
    sigmas_true_window = [hists_true[col].x_mean_rms_window()[1] for col in cols_true]

    # plot this here
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    means_window = np.asarray(means_window, dtype=float)
    sigmas_window = np.asarray(sigmas_window, dtype=float)
    means_true = np.asarray(means_true, dtype=float)
    sigmas_true = np.asarray(sigmas_true, dtype=float)
    means_true_window = np.asarray(means_true_window, dtype=float)
    sigmas_true_window = np.asarray(sigmas_true_window, dtype=float)

    ratio_sigmas = sigmas / sigmas_true
    ratio_sigmas_window = sigmas_window / sigmas_true_window

    mu_edges = np.linspace(-5.0, 5.0, 50)
    sg_edges = np.linspace(0., 10, 50)
    ds_edges = np.linspace(0, 2.5, 50)

    mu_counts, _ = np.histogram(means, bins=mu_edges)
    mu_counts_window, _ = np.histogram(means_window, bins=mu_edges)

    sg_counts, _ = np.histogram(sigmas, bins=sg_edges)
    sg_counts_window, _ = np.histogram(sigmas_window, bins=sg_edges)

    mu_counts_true, _ = np.histogram(means_true, bins=mu_edges)
    mu_counts_true_window, _ = np.histogram(means_true_window, bins=mu_edges)

    sg_counts_true, _ = np.histogram(sigmas_true, bins=sg_edges)
    sg_counts_true_window, _ = np.histogram(sigmas_true_window, bins=sg_edges)
    
    rs_counts, _ = np.histogram(ratio_sigmas, bins=ds_edges)
    rs_counts_window, _ = np.histogram(ratio_sigmas_window, bins=ds_edges)

    utils.overlay_hists_precomputed(
        xs=[mu_counts, mu_counts_true],
        means=[float(np.mean(means)), float(np.mean(means_true))],
        rmss=[float(np.sqrt(np.mean(means * means))), float(np.sqrt(np.mean(means_true * means_true)))],
        bins=mu_edges,
        xlabel=f"{varname_template} mean",
        ylabel="Number of channels",
        title="",
        colors=["red", "gray"],
        legnames=["Corrected", "Measured"],
        outpath=os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_mus.pdf"),
    )

    utils.overlay_hists_precomputed(
        xs=[mu_counts_window, mu_counts_true_window],
        means=[float(np.mean(means_window)), float(np.mean(means_true_window))],
        rmss=[float(np.sqrt(np.mean(means_window * means_window))), float(np.sqrt(np.mean(means_true_window * means_true_window)))],
        bins=mu_edges,
        xlabel=f"{varname_template} mean [-10, 10]",
        ylabel="Number of channels",
        title="",
        colors=["red", "gray"],
        legnames=["Corrected", "Measured"],
        outpath=os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_mus_window.pdf"),
    )

    utils.overlay_hists_precomputed(
        xs=[sg_counts, sg_counts_true],
        means=[float(np.mean(sigmas)), float(np.mean(sigmas_true))],
        rmss=[float(np.sqrt(np.mean(sigmas * sigmas))), float(np.sqrt(np.mean(sigmas_true * sigmas_true)))],
        bins=sg_edges,
        xlabel=f"{varname_template} RMS",
        ylabel="Number of channels",
        title="",
        colors=["red", "gray"],
        legnames=["Corrected", "Measured"],
        outpath=os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_sigmas.pdf"),
    )

    utils.overlay_hists_precomputed(
        xs=[sg_counts_window, sg_counts_true_window],
        means=[float(np.mean(sigmas_window)), float(np.mean(sigmas_true_window))],
        rmss=[float(np.sqrt(np.mean(sigmas_window * sigmas_window))), float(np.sqrt(np.mean(sigmas_true_window * sigmas_true_window)))],
        bins=sg_edges,
        xlabel=f"{varname_template} RMS [-10, 10]",
        ylabel="Number of channels",
        title="",
        colors=["red", "gray"],
        legnames=["Corrected", "Measured"],
        outpath=os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_sigmas_window.pdf"),
    )

    utils.plot_hist_single_precomputed(x=rs_counts, mean=float(np.mean(ratio_sigmas)), rms=float(np.sqrt(np.mean(ratio_sigmas * ratio_sigmas))), bins=ds_edges, xlabel=f"σ (corr.) / σ (meas.)", ylabel="Number of channels", title="", color="red", outpath=os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_ratio_sigma.pdf"), show_mean_line=True)

    utils.plot_hist_single_precomputed(x=rs_counts_window, mean=float(np.mean(ratio_sigmas_window)), rms=float(np.sqrt(np.mean(ratio_sigmas_window * ratio_sigmas_window))), bins=ds_edges, xlabel=f"σ (corr.) [-10, 10] / σ (meas.) [-10, 10]", ylabel="Number of channels", title="", color="red", outpath=os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_ratio_sigma_window.pdf"), show_mean_line=True)

    channel_indices = np.arange(len(means))
    plt.figure(figsize=(8,5))
    plt.plot(channel_indices, means_true, "-", label="Measured", color="gray")
    plt.plot(channel_indices, means, "-", label="Corrected", color="red")
    plt.xlabel("Channel index")
    plt.ylabel("Mean")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_mus_vs_channel.pdf"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(channel_indices, means_true_window, "-", label="Measured", color="gray")
    plt.plot(channel_indices, means_window, "-", label="Corrected", color="red")
    plt.xlabel("Channel index")
    plt.ylabel("Mean [-10, 10]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_mus_window_vs_channel.pdf"))
    plt.close()

    channel_indices = np.arange(len(means))
    plt.figure(figsize=(8,5))
    plt.plot(channel_indices, sigmas_true, "-", label="Measured", color="gray")
    plt.plot(channel_indices, sigmas, "-", label="Corrected", color="red")
    plt.xlabel("Channel index")
    plt.ylabel("RMS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_sigma_vs_channel.pdf"))
    plt.close()

    channel_indices = np.arange(len(means))
    plt.figure(figsize=(8,5))
    plt.plot(channel_indices, sigmas_true_window, "-", label="Measured", color="gray")
    plt.plot(channel_indices, sigmas_window, "-", label="Corrected", color="red")
    plt.xlabel("Channel index")
    plt.ylabel("RMS [-10, 10]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_sigma_window_vs_channel.pdf"))
    plt.close()

    channel_indices = np.arange(len(means))
    plt.figure(figsize=(8,5))
    plt.plot(channel_indices, ratio_sigmas, "-", color="red")
    plt.xlabel("Channel index")
    plt.ylabel("σ (corr.) / σ (meas.)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_ratio_sigma_vs_channel.pdf"))
    plt.close()

    channel_indices = np.arange(len(means))
    plt.figure(figsize=(8,5))
    plt.plot(channel_indices, ratio_sigmas_window, "-", color="red")
    plt.xlabel("Channel index")
    plt.ylabel("σ (corr.) [-10, 10] / σ (meas.) [-10, 10]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_ratio_sigma_window_vs_channel.pdf"))
    plt.close()




def plot_gaussfit_summary_1d_multicol(varname_template: str, varname_true_template: str, value_iterator, out_root: str, nbins_x: int = None, x_range=None, gauss_p0=None):
    os.makedirs(out_root, exist_ok=True)
        
    cols = []
    cols_true = []
    for full_df in value_iterator():
        cols = [c for c in full_df.columns if fnmatch(c, varname_template)]
        cols_true = [c for c in full_df.columns if fnmatch(c, varname_true_template)]
        break

    hists = {c: Streaming1DHist(x_min=x_range[0], x_max=x_range[1], nbins_x=nbins_x) for c in cols}
    hists_true = {c: Streaming1DHist(x_min=x_range[0], x_max=x_range[1], nbins_x=nbins_x) for c in cols_true}

    for full_df in value_iterator():
        values = full_df[cols].to_numpy()
        values_true = full_df[cols_true].to_numpy()
        for colidx, col in enumerate(cols):
            hists[col].add(values[:, colidx])
        for colidx_true, col_true in enumerate(cols_true):
            hists_true[col_true].add(values_true[:, colidx_true])

    means = np.full(len(cols), np.nan, dtype=float)
    sigmas = np.full(len(cols), np.nan, dtype=float)
    for colidx, col in enumerate(cols):
        # hist = Streaming1DHist(x_min=x_range[0], x_max=x_range[1], nbins_x=nbins_x)
        hist = hists[col]

        # bin centers for the fit
        x = hist.H
        edges = hist.x_edges
        centers = 0.5 * (edges[:-1] + edges[1:])

        # fit only bins with content > 0 (avoid tons of zeros dominating)
        mask = np.isfinite(x) & (x > 0)
        xmin, xmax = (-10, 10)
        mask &= (centers >= xmin) & (centers <= xmax)
        xc = centers[mask]
        yc = x[mask].astype(float)

        def gauss(xx, A, mu, sigma):
            return A * np.exp(-0.5 * ((xx - mu) / sigma) ** 2)

        # start params: user-provided or simple defaults from existing inputs
        if gauss_p0 is None:
            A0 = float(np.max(yc)) if yc.size else float(np.max(x))
            mu0 = float(hist.x_mean_rms()[0])
            sigma0 = float(hist.x_mean_rms()[1]) if hist.x_mean_rms()[1] > 0 else 1.0
            gauss_p0_thisfit = (A0, mu0, sigma0)
        else:
            gauss_p0_thisfit = gauss_p0

        try:
            # Poisson-ish uncertainties for weighted fit
            sigma_y = np.sqrt(np.maximum(yc, 1.0))
            popt, _ = curve_fit(gauss, xc, yc, p0=gauss_p0_thisfit, sigma=sigma_y, absolute_sigma=True, maxfev=20000)
            A_hat, mu_hat, sigma_hat = popt
            means[colidx] = mu_hat
            sigmas[colidx] = sigma_hat
        except Exception as e:
            print(f"[WARNING] Gaussian fit failed for column {colidx}: {e}")

    
    means_true = np.full(len(cols_true), np.nan, dtype=float)
    sigmas_true = np.full(len(cols_true), np.nan, dtype=float)
    for colidx_true, col_true in enumerate(cols_true):
        hist = hists_true[col_true]

        # bin centers for the fit
        x = hist.H
        edges = hist.x_edges
        centers = 0.5 * (edges[:-1] + edges[1:])

        # fit only bins with content > 0 (avoid tons of zeros dominating)
        mask = np.isfinite(x) & (x > 0)
        xmin, xmax = (-10, 10)
        mask &= (centers >= xmin) & (centers <= xmax)
        xc = centers[mask]
        yc = x[mask].astype(float)

        def gauss(xx, A, mu, sigma):
            return A * np.exp(-0.5 * ((xx - mu) / sigma) ** 2)

        # start params: user-provided or simple defaults from existing inputs
        if gauss_p0 is None:
            A0 = float(np.max(yc)) if yc.size else float(np.max(x))
            mu0 = float(hist.x_mean_rms()[0])
            sigma0 = float(hist.x_mean_rms()[1]) if hist.x_mean_rms()[1] > 0 else 1.0
            gauss_p0_thisfit = (A0, mu0, sigma0)
        else:
            gauss_p0_thisfit = gauss_p0

        try:
            # Poisson-ish uncertainties for weighted fit
            sigma_y = np.sqrt(np.maximum(yc, 1.0))
            popt, _ = curve_fit(gauss, xc, yc, p0=gauss_p0_thisfit, sigma=sigma_y, absolute_sigma=True, maxfev=20000)
            A_hat, mu_hat, sigma_hat = popt
            means_true[colidx_true] = mu_hat
            sigmas_true[colidx_true] = sigma_hat
        except Exception as e:
            print(f"[WARNING] Gaussian fit failed for column {colidx_true}: {e}")

    # plot this here
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    means_true = np.asarray(means_true, dtype=float)
    sigmas_true = np.asarray(sigmas_true, dtype=float)

    good = np.isfinite(sigmas) & np.isfinite(sigmas_true)   # drops failures from either side
    # delta_sigmas = (sigmas_true[good] - sigmas[good]) / sigmas_true[good]
    delta_sigmas = sigmas[good] / sigmas_true[good]
    # delta_sigmas = sigmas - sigmas_true

    mu_edges = np.linspace(-5.0, 5.0, 50)
    sg_edges = np.linspace(0., 10, 50)
    # ds_edges = np.linspace(-1., 1., 50)
    ds_edges = np.linspace(0, 2.5, 50)

    mu_counts, _ = np.histogram(means, bins=mu_edges)
    sg_counts, _ = np.histogram(sigmas, bins=sg_edges)
    mu_counts_true, _ = np.histogram(means_true, bins=mu_edges)
    sg_counts_true, _ = np.histogram(sigmas_true, bins=sg_edges)
    ds_counts, _ = np.histogram(delta_sigmas, bins=ds_edges)

    utils.overlay_hists_precomputed(
        xs=[mu_counts, mu_counts_true],
        means=[float(np.mean(means)), float(np.mean(means_true))],
        rmss=[float(np.sqrt(np.mean(means * means))), float(np.sqrt(np.mean(means_true * means_true)))],
        bins=mu_edges,
        xlabel=f"{varname_template} fitted μ",
        ylabel="Number of channels",
        title="",
        colors=["red", "gray"],
        legnames=["Corrected", "Measured"],
        outpath=os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_gaussfit_mus.pdf"),
    )

    utils.overlay_hists_precomputed(
        xs=[sg_counts, sg_counts_true],
        means=[float(np.mean(sigmas)), float(np.mean(sigmas_true))],
        rmss=[float(np.sqrt(np.mean(sigmas * sigmas))), float(np.sqrt(np.mean(sigmas_true * sigmas_true)))],
        bins=sg_edges,
        xlabel=f"{varname_template} fitted σ",
        ylabel="Number of channels",
        title="",
        colors=["red", "gray"],
        legnames=["Corrected", "Measured"],
        outpath=os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_gaussfit_sigmas.pdf"),
    )

    utils.plot_hist_single_precomputed(x=ds_counts, mean=float(np.mean(delta_sigmas)), rms=float(np.sqrt(np.mean(delta_sigmas * delta_sigmas))), bins=ds_edges, xlabel=f"σ (corr.) / σ (meas.)", ylabel="Number of channels", title="", color="red", outpath=os.path.join(out_root, f"{varname_template.replace('*','all').replace('?','')}_delta_sigma.pdf"), show_mean_line=True)




def infer_erx_groups_from_columns(value_iterator, adc_colname_template: str, nch_per_erx: int, nerx: int):
    """
    Map each present channel column to an ERx id = channel_index // nch_per_erx,
    and return {erx_id: [column names]} keeping only groups with >= 2 channels.
    """

    for df in value_iterator():
        adc_cols = [c for c in df.columns if fnmatch(c, adc_colname_template)]

        groups = {}
        for erx in range(nerx):
            cols = [col for col in adc_cols if any([f"ch{idx:03d}" in col for idx in range(erx*nch_per_erx, (erx+1)*nch_per_erx)])]
            # print(f"selected cols for erx {erx}: ")
            # print(cols)
            if len(cols) >= 2:
                groups[erx] = cols
        return dict(sorted(groups.items()))  # sort by erx id for stable plotting order

def compute_coherent_noise(value_iterator, adc_colname_template: str, nch_per_erx: int, nerx: int, trunc_frac = 1.0):
    """
    Streaming version: accumulates per-ERx direct/alternating sums for
    (a) true and (b) corrected (i.e. residuals after method), then applies
    your coh/inc formulas with truncated RMS.
    """

    # 1) ERx groups from first batch
    groups = infer_erx_groups_from_columns(value_iterator, adc_colname_template=adc_colname_template, nch_per_erx=nch_per_erx, nerx=nerx)
    if not groups:
        raise RuntimeError(f"[coh/inc] No ERx with >=2 channels (nch_per_erx={nch_per_erx}).")

    erx_ids_sorted = np.array(sorted(groups.keys()), dtype=int)

    # 2) accumulators: lists of vectors per ERx (to apply truncated RMS later)
    dir: dict[int, list[np.ndarray]] = {erx: [] for erx in erx_ids_sorted}
    alt: dict[int, list[np.ndarray]] = {erx: [] for erx in erx_ids_sorted}

    # 3) stream once over the split for TRUE and RESIDUALS in lockstep
    for full_df in value_iterator():
        print("Computing sums now for this df:")
        print(full_df)
        for erx in erx_ids_sorted:
            cols = groups[erx]

            # E_b × ncols arrays
            np_2d = full_df[cols].to_numpy()

            # per-event sums (length E_b)
            d = np_2d.sum(axis=1)
            a = np_2d[:, ::2].sum(axis=1) - np_2d[:, 1::2].sum(axis=1)

            dir[erx].append(d)
            alt[erx].append(a)

    # 4) finalize per-ERx RMS and derive coherent/incoherent components
    cohs, incs = [], []
    for erx in erx_ids_sorted:
        ncols = len(groups[erx])

        d = np.concatenate(dir[erx], axis=0)
        a = np.concatenate(alt[erx], axis=0)

        rms_dir = utils.truncated_rms(d, trunc_frac)
        rms_alt = utils.truncated_rms(a, trunc_frac)
        delta = rms_dir**2 - rms_alt**2
        inc = rms_alt / np.sqrt(ncols)
        coh = np.sign(delta) * np.sqrt(abs(delta)) / ncols

        cohs.append(coh)
        incs.append(inc)

    return (np.asarray(cohs), np.asarray(incs), erx_ids_sorted)


def plot_coh_inc(value_iterator, adc_colname_template_true, adc_colname_template_corr, nch_per_erx, nerx, out_root: str, trunc_fracs = (1.0,)):
    os.makedirs(out_root, exist_ok=True)

    for f in trunc_fracs:
        coh_true, inc_true, erx_ids = compute_coherent_noise(value_iterator=value_iterator, adc_colname_template=adc_colname_template_true, nch_per_erx=nch_per_erx, nerx=nerx, trunc_frac=f)
        coh_corr, inc_corr, _       = compute_coherent_noise(value_iterator=value_iterator, adc_colname_template=adc_colname_template_corr, nch_per_erx=nch_per_erx, nerx=nerx, trunc_frac=f)

        with np.errstate(divide="ignore", invalid="ignore"):
            inc_ratio = np.nan_to_num(inc_corr / inc_true, nan=0.0)
            coh_ratio = np.nan_to_num(coh_corr / coh_true, nan=0.0)
            coh_over_inc_true = np.nan_to_num(coh_true / inc_true, nan=0.0)
            coh_over_inc_corr = np.nan_to_num(coh_corr / inc_corr, nan=0.0)

        fig = plt.figure(figsize=(7, 6))
        gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.10)
        ax1 = fig.add_subplot(gs[0]); 
        axr = fig.add_subplot(gs[1], sharex=ax1)
        axc = fig.add_subplot(gs[2], sharex=ax1)

        ax1.plot(erx_ids, inc_true, "o-",  label="incoherent (meas.)", color="tab:blue")
        ax1.plot(erx_ids, coh_true, "s-",  label="coherent (meas.)",   color="tab:orange")
        ax1.plot(erx_ids, inc_corr, "o--", label="incoherent (corr.)", color="tab:blue")
        ax1.plot(erx_ids, coh_corr, "s--", label="coherent (corr.)",   color="tab:orange")

        for ax in (ax1, axr, axc):
            ax.tick_params(axis="both", direction="in", top=True, bottom=True, left=True, right=True, labelsize=12)
            ax.grid(ls="--", alpha=0.3)

        ax1.set_ylabel("Noise (ADC)", fontsize=16, loc="top", labelpad=12)
        ax1.set_ylim(0., ax1.get_ylim()[1]*1.2)
        ax1.legend(loc="upper right", fontsize=12)

        axr.plot(erx_ids, inc_ratio, "o--", color="tab:blue")
        axr.plot(erx_ids, coh_ratio, "s--", color="tab:orange")
        axr.set_ylabel("corr. / meas.", fontsize=11, loc="center", labelpad=10)
        axr.set_ylim(0., 1.1)

        axc.plot(erx_ids, coh_over_inc_true, "D-",  color="black")
        axc.plot(erx_ids, coh_over_inc_corr, "D--", color="black")
        axc.set_xlabel("e-Rx", fontsize=16, loc="right", labelpad=8)
        axc.set_ylabel("coh. / inc.", fontsize=11, loc="center", labelpad=8)
        axc.set_ylim(0., max(axc.get_ylim()[1], 2.))

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(axr.get_xticklabels(), visible=False)

        frac_tag = f"{int(round(f * 100))}"
        outname = f"coh_inc_ratio_trunc-{frac_tag}.pdf"
        fig.savefig(os.path.join(out_root, outname), bbox_inches="tight", pad_inches=0.05)
        plt.close()
    


def plot_eigenvalues(cfg, column_tag, out_root):
    os.makedirs(out_root, exist_ok=True)

    vals = cfg.load_from_cov_folder(filename=f"eigenvalues_mm{column_tag}.parquet")
    x = np.arange(1, vals.size + 1)

    # log-y (clip at tiny positive to avoid -inf)
    vals = np.clip(vals, 1e-12, None)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(x, vals, marker='o', lw=1)
    ax.set_xlabel("mode index", loc="right")
    ax.set_ylabel("eigenvalue", loc="top")
    ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    outfilename = os.path.join(out_root, "eigenvalues.pdf")
    fig.savefig(outfilename)
    print(f"--> Plotted eigenvalues: {outfilename}")
    plt.close(fig)


def plot_eigenvectors(cfg, column_tag, top: int, out_root):
    os.makedirs(out_root, exist_ok=True)

    # vals = cfg.load_from_cov_folder(filename=f"eigenvalues_mm{column_tag}.parquet")
    # vecs = cfg.load_from_cov_folder(filename=f"eigenvectors_mm{column_tag}.parquet")
    vals = cfg.load_from_cov_folder(filename=f"eigenvalues_mcmc{column_tag}.parquet")
    vecs = cfg.load_from_cov_folder(filename=f"eigenvectors_mcmc{column_tag}.parquet")

    k = int(min(top, vecs.shape[1]))

    # (a) line plot across channel index
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    for i in range(k):
        v = vecs[[f"eigvec_{i}"]]
        # v = vecs[:, i]
        lam = vals["eigval"].loc[i]
        plt.plot(np.arange(v.size), v, label=f'Mode {i+1} ($\lambda$={lam:.3g})')
    
    for pos in range(0, cfg.nch_per_erx*(cfg.nerx+1), cfg.nch_per_erx):
        plt.axvline(pos, color='black', linestyle='--', linewidth=1)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylim((-0.3, 0.3))
    plt.xlabel('Channel')
    plt.ylabel('Eigenvector component')
    plt.legend(ncol=2, fontsize='small')
    fig.tight_layout()
    outfilename = os.path.join(out_root, "eigenvectors.pdf")
    fig.savefig(outfilename)
    print(f"--> Plotted top-{top} eigenvectors: {outfilename}")
    plt.close(fig)

def plot_loss(modeldir, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    # --- Load losses ---
    if not os.path.exists(f"{os.path.join(modeldir, 'train_losses.npy')}") or not os.path.exists(f"{os.path.join(modeldir, 'test_losses.npy')}"):
        raise FileNotFoundError("Missing train_losses.npy or val_losses.npy.")

    train_losses = np.load(os.path.join(modeldir, 'train_losses.npy'))
    val_losses = np.load(os.path.join(modeldir, 'test_losses.npy'))

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_plot_path = f"{plot_dir}/loss.pdf"
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot to: {loss_plot_path}")
