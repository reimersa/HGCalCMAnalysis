#!/usr/bin/env python3

import argparse
import json
import os
import warnings
from fnmatch import fnmatch
from functools import lru_cache
from glob import glob

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from scipy.integrate import IntegrationWarning, quad  # type: ignore
from scipy.optimize import curve_fit  # type: ignore
from scipy.signal import fftconvolve  # type: ignore
from scipy.stats import norm  # type: ignore


PARAMETER_NAMES = ["A_ped", "mu_p", "sigma_p", "A_mip", "mu_L", "c_L", "A_2mip"]


def _standard_landau_pdf_scalar(x):
    if x < -6.0:
        return 0.0

    def integrand(t):
        if t <= 0.0:
            return 0.0
        exponent = -t * np.log(t) - x * t
        if exponent > 650.0:
            return 0.0
        return np.exp(exponent) * np.sin(np.pi * t)

    value, _ = quad(integrand, 0.0, 80.0, limit=300)
    return value / np.pi


@lru_cache(maxsize=1)
def _landau_lookup():
    grid_x = np.unique(
        np.concatenate([np.linspace(-6.0, 15.0, 700), np.linspace(15.0, 400.0, 500)])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=IntegrationWarning)
        grid_y = np.asarray([_standard_landau_pdf_scalar(x) for x in grid_x])
    return grid_x, np.clip(grid_y, 0.0, None)


def landau_pdf(x, location=0.0, scale=1.0):
    if scale <= 0.0:
        return np.zeros_like(np.asarray(x, dtype=float))
    grid_x, grid_y = _landau_lookup()
    reduced = (np.asarray(x, dtype=float) - location) / scale
    return np.interp(reduced, grid_x, grid_y, left=0.0, right=0.0) / scale


def langau_pdf(x, location=0.0, landau_scale=1.0, gaussian_sigma=1.0, n_grid=4000, pad=40.0):
    x = np.asarray(x, dtype=float)
    lo = float(np.min(x)) - pad - 5.0 * gaussian_sigma
    hi = float(np.max(x)) + pad + 5.0 * gaussian_sigma
    grid = np.linspace(lo, hi, n_grid)
    dx = grid[1] - grid[0]
    landau = landau_pdf(grid, location=location, scale=landau_scale)
    half_width = max(5.0 * gaussian_sigma, 5.0 * dx)
    kernel_x = np.arange(-half_width, half_width + dx, dx)
    kernel = np.exp(-0.5 * (kernel_x / gaussian_sigma) ** 2)
    kernel /= kernel.sum() * dx
    convolution = fftconvolve(landau, kernel, mode="same") * dx
    return np.interp(x, grid, convolution, left=0.0, right=0.0)


def pedestal_component(x, A_ped, mu_p, sigma_p):
    return A_ped * norm.pdf(x, mu_p, sigma_p)


def mip_component(x, A_mip, mu_L, c_L, mu_p, sigma_p):
    return A_mip * langau_pdf(x, mu_L + mu_p, c_L, sigma_p)


def mip2_component(x, A_2mip, mu_L, c_L, mu_p, sigma_p):
    return A_2mip * langau_pdf(x, 2.0 * mu_L + mu_p, c_L, sigma_p)


def mip_model(x, A_ped, mu_p, sigma_p, A_mip, mu_L, c_L, A_2mip):
    return (
        pedestal_component(x, A_ped, mu_p, sigma_p)
        + mip_component(x, A_mip, mu_L, c_L, mu_p, sigma_p)
        + mip2_component(x, A_2mip, mu_L, c_L, mu_p, sigma_p)
    )


def fit_mip_spectrum(bin_centers, counts, fit_range=None, p0=None, bounds=None):
    bin_centers = np.asarray(bin_centers, dtype=float)
    counts = np.asarray(counts, dtype=float)
    mask = np.isfinite(bin_centers) & np.isfinite(counts)
    if fit_range is not None:
        mask &= (bin_centers >= fit_range[0]) & (bin_centers <= fit_range[1])
    x = bin_centers[mask]
    y = counts[mask]
    if x.size <= len(PARAMETER_NAMES):
        raise ValueError("Too few populated histogram bins for the seven-parameter MIP fit.")
    yerr = np.sqrt(np.maximum(y, 1.0))

    initial = {
        "A_ped": float(np.max(y)),
        "mu_p": 0.0,
        "sigma_p": 1.0,
        "A_mip": float(np.max(y)) * 0.1,
        "mu_L": 9.0,
        "c_L": 1.0,
        "A_2mip": float(np.max(y)) * 0.01,
    }
    if p0:
        initial.update(p0)
    limits = {
        "A_ped": (0.0, np.inf),
        "mu_p": (-10.0, 10.0),
        "sigma_p": (0.05, 10.0),
        "A_mip": (0.0, np.inf),
        "mu_L": (0.5, 60.0),
        "c_L": (0.05, 10.0),
        "A_2mip": (0.0, np.inf),
    }
    if bounds:
        limits.update(bounds)

    popt, pcov = curve_fit(
        mip_model,
        x,
        y,
        p0=[initial[name] for name in PARAMETER_NAMES],
        sigma=yerr,
        absolute_sigma=True,
        bounds=(
            [limits[name][0] for name in PARAMETER_NAMES],
            [limits[name][1] for name in PARAMETER_NAMES],
        ),
        maxfev=20000,
    )
    errors = np.sqrt(np.diag(pcov))
    residual = (y - mip_model(x, *popt)) / yerr
    chi2 = float(np.sum(residual * residual))
    ndof = int(x.size - len(popt))
    values = dict(zip(PARAMETER_NAMES, map(float, popt)))

    peak_grid = np.linspace(float(np.min(x)), float(np.max(x)), 4000)
    one_mip = mip_component(
        peak_grid,
        values["A_mip"],
        values["mu_L"],
        values["c_L"],
        values["mu_p"],
        values["sigma_p"],
    )
    result = {
        "values": values,
        "errors": dict(zip(PARAMETER_NAMES, map(float, errors))),
        "chi2": chi2,
        "ndof": ndof,
        "chi2_ndof": chi2 / ndof if ndof > 0 else None,
        "one_mip_peak": float(peak_grid[int(np.argmax(one_mip))]),
        "covariance": pcov.astype(float).tolist(),
    }
    quality_warnings = []
    mu_l_low, mu_l_high = limits["mu_L"]
    mu_l_tolerance = 1.0e-3 * (mu_l_high - mu_l_low)
    if values["mu_L"] <= mu_l_low + mu_l_tolerance:
        quality_warnings.append(f"mu_L is at or near its lower bound ({mu_l_low:g})")
    if values["mu_L"] >= mu_l_high - mu_l_tolerance:
        quality_warnings.append(f"mu_L is at or near its upper bound ({mu_l_high:g})")
    chi2_ndof = result["chi2_ndof"]
    if chi2_ndof is None or not np.isfinite(chi2_ndof):
        quality_warnings.append("chi2/ndof is not finite")
    elif chi2_ndof > 10.0:
        quality_warnings.append(f"chi2/ndof is very large ({chi2_ndof:.3g})")
    result["fit_quality"] = {
        "valid": not quality_warnings,
        "warnings": quality_warnings,
    }
    return result


def histogram_parquet_files(paths, column_pattern, bins, value_range):
    counts = np.zeros(bins, dtype=np.int64)
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    columns = None
    n_finite = 0
    n_outside = 0
    for path in paths:
        available = pq.ParquetFile(path).schema.names
        matched = [column for column in available if fnmatch(column, column_pattern)]
        if columns is None:
            columns = matched
        elif matched != columns:
            raise ValueError(f"Matched column order changed in {path}.")
        if not matched:
            raise KeyError(f"No columns matching {column_pattern!r} in {path}.")
        values = pd.read_parquet(path, columns=matched).to_numpy(dtype=np.float64, copy=False).ravel()
        values = values[np.isfinite(values)]
        n_finite += int(values.size)
        n_outside += int(np.count_nonzero((values < value_range[0]) | (values > value_range[1])))
        counts += np.histogram(values, bins=edges)[0]
    return counts, edges, columns or [], n_finite, n_outside


def _fit_summary_text(result):
    values = result["values"]
    errors = result["errors"]
    chi2_ndof = result["chi2_ndof"]
    chi2_text = f"chi2/ndof = {chi2_ndof:.2f}" if chi2_ndof is not None else "chi2/ndof = n/a"
    summary = (
        f"mu_p = {values['mu_p']:.3g} +/- {errors['mu_p']:.2g}\n"
        f"sigma_p = {values['sigma_p']:.3g} +/- {errors['sigma_p']:.2g}\n"
        f"Landau location = {values['mu_L']:.3g} +/- {errors['mu_L']:.2g}\n"
        f"Landau width c_L = {values['c_L']:.3g} +/- {errors['c_L']:.2g}\n"
        f"1-MIP peak = {result['one_mip_peak']:.3g}\n"
        f"{chi2_text}"
    )
    if not result.get("fit_quality", {}).get("valid", True):
        summary += "\nFIT QUALITY WARNING"
    return summary


def plot_fit(counts, edges, result, output_path, logy=False, xlabel="ADC [counts]"):
    centers = 0.5 * (edges[:-1] + edges[1:])
    dense_x = np.linspace(edges[0], edges[-1], 2000)
    values = result["values"]
    args = [values[name] for name in PARAMETER_NAMES]
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.step(centers, counts, where="mid", color="0.3", label="data")
    ax.plot(dense_x, mip_model(dense_x, *args), color="red", label="total fit")
    ax.plot(dense_x, pedestal_component(dense_x, *args[:3]), "--", label="pedestal")
    ax.plot(dense_x, mip_component(dense_x, args[3], args[4], args[5], args[1], args[2]), "--", label="1 MIP")
    ax.plot(dense_x, mip2_component(dense_x, args[6], args[4], args[5], args[1], args[2]), "--", label="2 MIP")
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    summary = _fit_summary_text(result)
    ax.text(0.98, 0.97, summary, transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.85})
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entries")
    ax.grid(ls="--", alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fit and plot a pedestal + 1-MIP + 2-MIP Landau-Gaussian spectrum.")
    parser.add_argument("inputs", nargs="+", help="Parquet files or glob patterns.")
    parser.add_argument("--columns", default="adc_ch*_pedsub", help="fnmatch pattern for columns combined in the spectrum.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--range", nargs=2, type=float, default=(-10.0, 40.0), metavar=("MIN", "MAX"))
    parser.add_argument("--fit-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--xlabel", default="ADC [counts]")
    args = parser.parse_args()

    paths = sorted({path for pattern in args.inputs for path in (glob(pattern) or [pattern]) if os.path.exists(path)})
    if not paths:
        raise FileNotFoundError("No input parquet files matched.")
    if args.bins <= 0 or args.range[0] >= args.range[1]:
        raise ValueError("Histogram bins/range are invalid.")

    counts, edges, columns, n_finite, n_outside = histogram_parquet_files(
        paths=paths,
        column_pattern=args.columns,
        bins=args.bins,
        value_range=args.range,
    )
    result = fit_mip_spectrum(
        bin_centers=0.5 * (edges[:-1] + edges[1:]),
        counts=counts,
        fit_range=args.fit_range,
    )
    result.update(
        {
            "input_files": paths,
            "columns": columns,
            "column_pattern": args.columns,
            "histogram_range": list(args.range),
            "fit_range": args.fit_range,
            "bins": args.bins,
            "finite_entries": n_finite,
            "entries_outside_histogram": n_outside,
        }
    )
    os.makedirs(args.output_dir, exist_ok=True)
    for suffix, logy in [("", False), ("_logy", True)]:
        plot_fit(
            counts=counts,
            edges=edges,
            result=result,
            output_path=os.path.join(args.output_dir, f"mip_landau_fit{suffix}.pdf"),
            logy=logy,
            xlabel=args.xlabel,
        )
    json_path = os.path.join(args.output_dir, "mip_landau_fit.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Wrote MIP/Landau fit plots and parameters to {args.output_dir}")


if __name__ == "__main__":
    main()
