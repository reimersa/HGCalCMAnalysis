import os
import psutil # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
import matplotlib.colors as mcolors # type: ignore
import matplotlib.ticker as ticker # type: ignore
from typing import Union, List, Iterable, Optional, Tuple
import pandas as pd # type: ignore
import math

from scipy.optimize import curve_fit  # type: ignore

import classes

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def compute_cov_streaming(
    iter_i,
    iter_j,
) -> pd.DataFrame:
    """
    iter_i / iter_j must yield DataFrames with identical row indices per step,
    and fixed, consistent column orders across all steps.
    """
    # prime the iterators
    it_i = iter(iter_i())
    it_j = iter(iter_j())
    first_i = next(it_i)
    first_j = next(it_j)

    acc = classes.CovAccumulator(cols_i=list(first_i.columns), cols_j=list(first_j.columns))

    acc.update(first_i, first_j)

    # continue with remaining chunks
    idx=0
    for df_i, df_j in zip(it_i, it_j):
        acc.update(df_i, df_j)
        idx+=1


    return acc.finalize()


def f_corr_from_cov(cov) -> float:
    # Convert DataFrame to NumPy array if needed
    if isinstance(cov, pd.DataFrame):
        cov = cov.to_numpy()
    elif not isinstance(cov, np.ndarray):
        raise TypeError("cov must be a pandas.DataFrame or numpy.ndarray")

    total_sum2 = np.sum(cov ** 2)
    diag_sum2  = np.sum(np.diag(cov) ** 2)

    if total_sum2 == 0:
        return np.nan  # avoid division by zero for empty matrices

    f_corr = 1.0 - diag_sum2 / total_sum2
    return f_corr


# reshape measurements into event x channel DataFrame (rows sorted by eventid)
def build_target_df(values: np.ndarray, eventid: np.ndarray) -> pd.DataFrame:
    cols = [f"ch_{i:03d}" for i in range(values.shape[1])]
    df = pd.DataFrame(values, index=pd.Index(eventid, name="eventid"), columns=cols).sort_index()
    return df



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

# build inputs_df (all features) and cm_df (CM-only subset), both row-sorted by eventid
def build_input_and_cm_df(inputs: np.ndarray, eventid: np.ndarray, ncm: int, colnames_inputs: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if inputs.shape[1] < ncm:
        raise ValueError(f"Requested ncm={ncm} but inputs have only {inputs.shape[1]} columns")
    
    # Vector inputs (per-channel per-event information)
    cols = {}
    for j, name in enumerate(colnames_inputs):
        cols[name] = [inputs[i, :, j].astype(np.float32, copy=False) for i in range(inputs.shape[0])]
    inputs_df = pd.DataFrame(cols, index=pd.Index(eventid, name="eventid")).sort_index()

    # CM subset of scalar inputs
    cm_cols = [c for c in colnames_inputs if c.startswith("cm_erx")]
    if len(cm_cols) != ncm:
        raise ValueError(f"Found {len(cm_cols)} CM columns by name ({cm_cols[:5]}...), but cfg.ncmchannels={ncm}.")
    cm_df = inputs_df[cm_cols]

    return (inputs_df, cm_df)

def corr_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance to correlation (safe when diagonal has zeros)."""
    d = np.diag(cov.to_numpy())
    inv = np.zeros_like(d, dtype=float)
    pos = d > 0
    inv[pos] = 1.0 / np.sqrt(d[pos])
    D = np.diag(inv)
    R = D @ cov.to_numpy() @ D
    # numerical guard
    np.clip(R, -1.0, 1.0, out=R)
    return pd.DataFrame(R, index=cov.index, columns=cov.columns)


def compute_eig_from_cov(C):
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    return(vals[order], vecs[:, order])

def project_cm_to_eig(cm_df: pd.DataFrame, eigvecs) -> pd.DataFrame:
    V = eigvecs.to_numpy() if isinstance(eigvecs, pd.DataFrame) else np.asarray(eigvecs)
    if V.shape[0] != cm_df.shape[1]:
        raise ValueError(f"Shape mismatch: cm has {cm_df.shape[1]} columns but eigvecs has {V.shape[0]} rows.")
    Y = cm_df.to_numpy(dtype=float) @ V   # (n_events, 12)

    # Nice column names
    cols = [f"cm_eigdir{i:02d}" for i in range(V.shape[1])]
    return pd.DataFrame(Y, index=cm_df.index, columns=cols)


def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # in MB
    print(f"[{tag}] Memory usage: {mem:.2f} MB")


def overlay_profiles(vals_x, list_of_vals_y, label_x, label_y, labels_profiles, output_filename, nbins_x=20, ratio_to = None):
    if not len(list_of_vals_y) == len(labels_profiles):
        raise ValueError(f"Number of profiles to plot ({len(list_of_vals_y)}) does not match number of profile labels ({len(labels_profiles)})")
    vals_x = np.asarray(vals_x)
    list_of_vals_y = [np.asarray(v) for v in list_of_vals_y]

    bins_x = np.histogram_bin_edges(vals_x, bins=nbins_x)
    centers_x = (bins_x[:-1] + bins_x[1:]) / 2
    bin_indices_x = np.digitize(vals_x, bins_x) - 1

    list_of_means_y = [np.full_like(centers_x, np.nan, dtype=float) for v in list_of_vals_y]
    for i in range(len(centers_x)):
        for j in range(len(list_of_means_y)):
            in_bin = list_of_vals_y[j][bin_indices_x == i]
            if len(in_bin) > 0:
                list_of_means_y[j][i] = np.mean(in_bin)

    
    if ratio_to is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        axes = (ax,)
    else:
        fig = plt.figure(figsize=(6.8, 6.2))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax  = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1], sharex=ax)
        axes = (ax, axr)

    # Top panel: overlay means
    for m, lbl in zip(list_of_means_y, labels_profiles):
        valid = ~np.isnan(m)
        ax.plot(centers_x[valid], m[valid], label=lbl, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if ratio_to is not None:
        # Hide xticklabels on top panel (shared axis)
        plt.setp(ax.get_xticklabels(), visible=False)

        # Ratio panel
        base = list_of_means_y[int(ratio_to)]
        # Build ratios with safe division: mask where base is nan or ~0
        eps = 0.0
        denom_mask = (~np.isnan(base)) & (np.abs(base) > eps)

        for m, lbl in zip(list_of_means_y, labels_profiles):
            ratio = np.full_like(base, np.nan, dtype=float)
            ok = denom_mask & (~np.isnan(m))
            ratio[ok] = m[ok] / base[ok]
            valid = ~np.isnan(ratio)
            axr.plot(centers_x[valid], ratio[valid], linewidth=1.8, marker=".", markersize=3, label=f"{lbl} / {labels_profiles[int(ratio_to)]}")

        axr.axhline(1.0, color="k", lw=1, ls="--", alpha=0.6)
        axr.set_xlabel(label_x)
        axr.set_ylabel("ratio")
        axr.grid(True, alpha=0.3)
        axr.set_ylim(0.8, 1.2)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved overlay of means: {output_filename}")
    plt.close()

def plot_y_vs_x_with_marginals_hist(H: np.ndarray,
                                    x_edges: np.ndarray,
                                    y_edges: np.ndarray,
                                    x_prof_centers: np.ndarray,
                                    x_prof_mean: np.ndarray,
                                    label_x: str,
                                    label_y: str,
                                    label_profile: str,
                                    output_filename: str,
                                    make_profile_plot: bool=False):
    """
    Render a 2D heatmap of counts with a top marginal (x-profile mean as a line)
    and a right marginal (y-histogram). Closely matches plot_y_vs_x_with_marginals.
    """

    # Marginals
    x_hist = H.sum(axis=1)   # per x-bin totals
    y_hist = H.sum(axis=0)   # per y-bin totals

    # Setup figure with 3 axes
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)

    ax_main = plt.subplot(gs[1, 0])
    ax_top = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_right = plt.subplot(gs[1, 1], sharey=ax_main)

    # Show ticks on all 4 sides
    for ax in [ax_main, ax_top, ax_right]:
        ax.tick_params(
            axis='both',
            which='both',
            direction='in',
            top=True,
            bottom=True,
            left=True,
            right=True,
            labelsize=12
        )

    # 2D histogram
    cmap = plt.cm.viridis.copy()
    cmap.set_under("white")
    # norm = mcolors.Normalize(vmin=1)
    norm = mcolors.LogNorm(vmin=1)

    # Main 2D heatmap
    # X, Y = np.meshgrid(x_edges, y_edges, indexing="ij")
    im = ax_main.pcolormesh(x_edges, y_edges, H.T, cmap=cmap, norm=norm, shading="auto")  # note H.T for (x,y)
    # cb = fig.colorbar(im, ax=ax_main, pad=0.01)
    # cb.set_label("counts")

    ax_main.scatter(x_prof_centers, x_prof_mean, color='red', s=25, label=label_profile, zorder=10)
    ax_main.legend(fontsize=15)
    ax_main.tick_params(labelsize=15)

    ax_top.set_autoscalex_on(False)
    ax_right.set_autoscaley_on(False)

    # 1D histograms
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax_top.fill_between(x_centers, x_hist, step="mid", color='gray')
    ax_right.fill_betweenx(y_centers, 0, y_hist, step="mid", color='gray') #, orientation='horizontal'

    # Clean ticks and labels
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_right.tick_params(axis='y', labelleft=False)

    ax_main.set_xlabel(label_x, fontsize=19, loc='right', labelpad=10)
    ax_main.set_ylabel(label_y, fontsize=19, loc='top', labelpad=1)

    ax_main.grid(True)
    plt.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.97)
    plt.savefig(output_filename)
    print(f"Saved 2-d plot with marginals {output_filename}")
    plt.close()

    if not make_profile_plot:
        return

    root, ext = os.path.splitext(output_filename)
    output_profile = f"{root}_profile{ext}"

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        bottom=True,
        left=True,
        right=True,
        labelsize=15,
    )

    # ax2.scatter(x_prof_centers, x_prof_mean, color="red", s=25, label=label_profile, zorder=10)
    ax2.plot(x_prof_centers, x_prof_mean, marker=None, linewidth=1., label=label_profile)

    ax2.set_xlabel(label_x, fontsize=19, loc="right", labelpad=10)
    ax2.set_ylabel(label_profile if label_y is None else label_y, fontsize=19, loc="top", labelpad=1)

    ax2.grid(True)
    ax2.legend(fontsize=15)

    fig2.subplots_adjust(left=0.10, right=0.99, bottom=0.12, top=0.97)
    fig2.savefig(output_profile)
    print(f"Saved profile-only plot {output_profile}")
    plt.close(fig2)

def plot_covariance(df, nch_per_erx, title, xtitle, ytitle, ztitle, output_filename, zrange=(-1., 1.)):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot 2D heatmap
    im = ax.pcolormesh(
        df.columns,  # x bin edges
        df.index,    # y bin edges
        df.values,   # 2D values
        shading='auto',
        cmap='coolwarm',   # or 'RdBu', 'coolwarm', 'plasma', etc.,
        vmin=zrange[0],
        vmax=zrange[1]
    )

    # Draw dashed lines every `nch_per_erx` channels
    n_channels = df.shape[0]
    for i in range(nch_per_erx, n_channels, nch_per_erx):
        ax.axhline(i-0.5, color='black', linestyle='--', linewidth=0.7)
        ax.axvline(i-0.5, color='black', linestyle='--', linewidth=0.7)

    ticks = np.arange(0, n_channels, nch_per_erx)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    # Labels and styling
    ax.set_title(title)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(ztitle)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved 2-d plot {output_filename}")
    plt.close()


def plot_hist_single(x: np.ndarray, bins: Union[int, np.ndarray], color: str, xlabel: str, title: str, outpath: str, show_mean_line: bool = True) -> None:
    edges = np.histogram_bin_edges(x, bins=bins) if isinstance(bins, int) else bins
    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=edges, color=color)
    if show_mean_line:
        mu = float(np.mean(x))
        plt.axvline(mu, color="k", ls="--", label=f"mean = {mu:.3%}")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Number of channels")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_hist_single_precomputed(x: np.ndarray, mean: float, rms: float, bins: Union[int, np.ndarray], color: str, xlabel: str, title: str, outpath: str, show_mean_line: bool = True, ylabel="Number of channels", do_gauss_fit: bool=False, gauss_p0: Optional[Tuple[float, float, float]] = None, mean_window: float=None, rms_window: float=None) -> None:
    edges = np.histogram_bin_edges(x, bins=bins) if isinstance(bins, int) else bins
    plt.figure(figsize=(8, 5))
    plt.step(edges[:-1], x, where="mid", color=color)
    if show_mean_line:
        plt.axvline(mean, color="k", ls="-", label=f"mean = {mean:.3f}, rms = {rms:.3f}")
        if mean_window is not None and rms_window is not None:
            plt.axvline(mean_window, color="k", ls="--", label=f"mean [-10, 10] = {mean_window:.3f}, rms [-10, 10] = {rms_window:.3f}")

    # --- Gaussian fit (minimal add-on) ---
    if do_gauss_fit:

        # bin centers for the fit
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
            mu0 = float(mean)
            sigma0 = float(rms) if rms > 0 else 1.0
            gauss_p0 = (A0, mu0, sigma0)

        try:
            # Poisson uncertainties for fit
            sigma_y = np.sqrt(np.maximum(yc, 1.0))
            popt, _ = curve_fit(gauss, xc, yc, p0=gauss_p0, sigma=sigma_y, absolute_sigma=True, maxfev=20000)
            A_hat, mu_hat, sigma_hat = popt

            x_dense = np.linspace(edges[0], edges[-1], 1000)
            plt.plot(
                x_dense, gauss(x_dense, *popt),
                label=f"gauss: A={A_hat:.1f}, mu={mu_hat:.3f}, sigma={sigma_hat:.3f}",
            )
        except Exception as e:
            print(f"[WARNING] Gaussian fit failed: {e}")

    if show_mean_line or do_gauss_fit:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(bottom=0.)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"--> Saved 1d distribution to {outpath}")


def overlay_hists_precomputed(xs, means, rmss, bins: Union[int, np.ndarray], colors, legnames, xlabel: str, title: str, outpath: str, ylabel="Number of channels") -> None:
    edges = np.histogram_bin_edges(x, bins=bins) if isinstance(bins, int) else bins
    plt.figure(figsize=(8, 5))
    for x, mean, rms, color, legname in zip(xs, means, rmss, colors, legnames):
        # plt.step(edges[:-1], x, where="mid", label=f"{legname}: mean = {mean:.2f}, rms = {rms:.2f}", color=color)
        bin_widths = np.diff(edges)
        plt.bar(edges[:-1], x, width=bin_widths, align="edge", alpha=0.6, color=color, edgecolor=None, label=f"{legname}: mean = {mean:.2f}, rms = {rms:.2f}")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(bottom=0.)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"--> Saved 1d distribution to {outpath}")

def plot_hist_overlay_pair(a: np.ndarray, b: np.ndarray, bins: Union[int, np.ndarray], label_a: str, label_b: str, color_a: str, color_b: str, xlabel: str, title: str, outpath: str) -> None:
    # common bins from both series if bins is an int
    if isinstance(bins, int):
        edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=bins)
    else:
        edges = bins
    plt.figure(figsize=(8, 5))
    plt.hist(a, bins=edges, alpha=0.6, label=label_a, color=color_a)
    plt.hist(b, bins=edges, alpha=0.6, label=label_b, color=color_b)
    plt.xlabel(xlabel)
    plt.ylabel("Number of channels")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath); plt.close()

def truncated_rms(x: np.ndarray, fraction: float = 1.0) -> float:
    x = np.asarray(x).ravel()
    if x.size == 0: 
        return np.nan
    if not (0 < fraction <= 1): 
        raise ValueError("fraction must be in (0, 1].")
    if fraction == 1.0: 
        return np.sqrt(np.nanmean(x**2))

    lo = (1.0 - fraction) / 2.0
    hi = 1.0 - lo

    vmin, vmax = np.quantile(x, [lo, hi])
    sel = x[(x >= vmin) & (x <= vmax)]
    
    return np.sqrt(np.nanmean(sel**2)) if sel.size else np.nan

def get_input_tag(basetag: str, normalize_to_unit_area: bool, remove_disconnected: bool, standardize_std: bool = False) -> str:
    result_tag = basetag
    if normalize_to_unit_area:
        result_tag += "_normalizedarea"
    if remove_disconnected:
        result_tag += "_nodisconnected"
    if standardize_std:
        result_tag += "_unitstd"
    return result_tag

def round_nearest(n):
    return int(math.floor(n + 0.5)) if n >= 0 else int(math.ceil(n - 0.5))



def stack_vector_df(df):
    # Each column contains a 1-D array/list of length C per row.
    cols_stacked = [np.stack(df[c].to_numpy(dtype=object)) for c in df.columns]  # each (E, C)
    if len(cols_stacked) == 1:
        return cols_stacked[0]                         # (E, C)
    return np.stack(cols_stacked, axis=-1)             # (E, C, F)
