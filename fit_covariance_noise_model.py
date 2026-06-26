#!/usr/bin/env python3

import argparse
import os

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.optimize import minimize  # type: ignore

import classes
import utils


def main():
    parser = argparse.ArgumentParser(description="Fit the paper-inspired diagonal + low-rank noise model to sigma_mm.")
    parser.add_argument("-r", "--run", type=str, required=True, help="Run whose sigma_mm covariance should be fit.")
    parser.add_argument("-p", "--pedestal-run", type=str, required=True, help="Pedestal run used for the inputs.")
    parser.add_argument(
        "--correction-run",
        type=str,
        required=True,
        help="Correction-run tag used in the covariance folder path.",
    )
    parser.add_argument("-c", "--column-tag", type=str, default="", help="Measurement column tag, e.g. '_resid_dnn'.")
    parser.add_argument(
        "-k",
        "--n-coherent",
        nargs="+",
        type=int,
        default=[1],
        help="One or more coherent-source counts to fit.",
    )
    parser.add_argument("--max-nfev", type=int, default=2000, help="Maximum least-squares function evaluations.")
    parser.add_argument(
        "-m",
        "--modules",
        nargs="+",
        metavar="MOD",
        default=["ML_F3WC_IH0182"],
        help="List of module names to fit.",
    )
    parser.add_argument(
        "--module-for-correction",
        type=str,
        required=True,
        help="Module whose correction context should be fit.",
    )
    parser.add_argument(
        "--selection-for-correction",
        type=str,
        default="",
        help="Optional selection tag encoded in the correction-artifact folder.",
    )
    args = parser.parse_args()

    cfgs = [
        classes.AnalysisConfig(
            modulename=x,
            run=_maybe_int(args.run),
            run_for_pedestal=_maybe_int(args.pedestal_run),
            run_for_correction=_maybe_int(args.correction_run),
            module_for_correction=args.module_for_correction,
            selection_for_correction=args.selection_for_correction,
            standardize_std=False,
            inputfoldertag="",
        )
        for x in args.modules
    ]

    for cfg in cfgs:
        for n_coherent in args.n_coherent:
            fit_covariance_noise_model(
                cfg=cfg,
                column_tag=args.column_tag,
                n_coherent=n_coherent,
                max_nfev=args.max_nfev,
            )


def fit_covariance_noise_model(cfg, column_tag: str, n_coherent: int = 1, max_nfev: int = 2000) -> None:
    print("Hello from fit_covariance_noise_model()!")
    if n_coherent < 0:
        raise ValueError(f"n_coherent must be >= 0, got {n_coherent}")

    artifact_tag = column_tag
    print(
        f"Loading sigma_mm{artifact_tag} for module={cfg.modulename}, run={cfg.run}, "
        f"n_coherent={n_coherent}, max_nfev={max_nfev}"
    )
    sigma_mm_df = cfg.load_from_cov_folder(filename=f"sigma_mm{artifact_tag}.parquet")
    sigma_mm = sigma_mm_df.to_numpy(dtype=np.float64)
    sigma_mm = 0.5 * (sigma_mm + sigma_mm.T)
    n_channels = sigma_mm.shape[0]
    n_matrix_entries = sigma_mm.size

    x0 = _initial_guess(cov=sigma_mm, n_coherent=n_coherent)
    bounds = [(0.0, None)] * n_channels + [(None, None)] * (n_channels * n_coherent)
    print(
        f"Starting covariance fit: nch={n_channels}, params={x0.size}, "
        f"matrix_entries={n_matrix_entries}"
    )

    def objective_and_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
        d, L = _unpack_params(params=params, n_channels=n_channels, n_coherent=n_coherent)
        sigma_model = _build_model(d=d, L=L)
        resid = sigma_model - sigma_mm
        mse = float(np.mean(resid * resid))

        grad_d = (2.0 / n_matrix_entries) * np.diag(resid)
        grad_L = (4.0 / n_matrix_entries) * (resid @ L)
        grad = np.concatenate([grad_d, grad_L.reshape(-1)])
        return mse, grad

    mse0, _ = objective_and_gradient(x0)
    print(f"Initial MSE = {mse0:.6g}")

    iter_state = {"iter": 0}

    def callback(xk: np.ndarray) -> None:
        iter_state["iter"] += 1
        mse, _ = objective_and_gradient(xk)
        if iter_state['iter'] % 10 == 0:
            print(f"  iter={iter_state['iter']:04d} mse={mse:.6g}")

    result = minimize(
        fun=lambda params: objective_and_gradient(params)[0],
        x0=x0,
        jac=lambda params: objective_and_gradient(params)[1],
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={"maxiter": max_nfev},
    )
    print(
        f"Finished covariance fit: success={result.success}, status={result.status}, "
        f"nit={getattr(result, 'nit', -1)}, nfev={result.nfev}, njev={getattr(result, 'njev', -1)}, "
        f"message='{result.message}'"
    )

    d_fit, L_fit = _unpack_params(params=result.x, n_channels=n_channels, n_coherent=n_coherent)
    d_fit = np.clip(d_fit, 0.0, None)
    L_fit = _canonicalize_loadings(L_fit)
    sigma_coherent = L_fit @ L_fit.T
    sigma_incoherent = np.diag(d_fit)
    sigma_model = sigma_incoherent + sigma_coherent
    sigma_residual = sigma_mm - sigma_model

    summary = _fit_summary(
        sigma_data=sigma_mm,
        sigma_model=sigma_model,
        sigma_coherent=sigma_coherent,
        d=d_fit,
        result=result,
        n_coherent=n_coherent,
    )

    os.makedirs(cfg.noise_model_fit_folder, exist_ok=True)
    fit_tag = f"{artifact_tag}_nc{n_coherent}"

    df_summary = pd.DataFrame([summary])
    df_channels = pd.DataFrame(
        {
            "data_diag": np.diag(sigma_mm),
            "incoherent_var": d_fit,
            "incoherent_sigma": np.sqrt(np.maximum(d_fit, 0.0)),
            "coherent_var": np.diag(sigma_coherent),
            "model_diag": np.diag(sigma_model),
            "residual_diag": np.diag(sigma_residual),
        },
        index=sigma_mm_df.index,
    )
    df_loadings = pd.DataFrame(
        L_fit,
        index=sigma_mm_df.index,
        columns=[f"coh_source_{idx:02d}" for idx in range(n_coherent)],
    )

    def write_df(df: pd.DataFrame, filename: str, index: bool = True) -> None:
        utils.write_via_tmpdir(
            outfilename=os.path.join(cfg.noise_model_fit_folder, filename),
            suffix=".parquet",
            writer_fn=lambda tmp, data=df, use_index=index: data.to_parquet(tmp, index=use_index, compression="zstd"),
        )

    write_df(df_summary, f"noise_model_summary_mm{fit_tag}.parquet", index=False)
    write_df(df_channels, f"noise_model_channels_mm{fit_tag}.parquet")
    write_df(df_loadings, f"noise_model_loadings_mm{fit_tag}.parquet")
    write_df(pd.DataFrame(sigma_model, index=sigma_mm_df.index, columns=sigma_mm_df.columns), f"sigma_mm_model{fit_tag}.parquet")
    write_df(pd.DataFrame(sigma_coherent, index=sigma_mm_df.index, columns=sigma_mm_df.columns), f"sigma_mm_coherent{fit_tag}.parquet")
    write_df(pd.DataFrame(sigma_incoherent, index=sigma_mm_df.index, columns=sigma_mm_df.columns), f"sigma_mm_incoherent{fit_tag}.parquet")
    write_df(pd.DataFrame(sigma_residual, index=sigma_mm_df.index, columns=sigma_mm_df.columns), f"sigma_mm_fitresidual{fit_tag}.parquet")

    print(
        f"Fitted covariance noise model with n_coherent={n_coherent} for sigma_mm{artifact_tag}. "
        f"mse={summary['mse']:.4g}, coherent_fraction={summary['coherent_fraction_trace']:.4g}"
    )
    print(f"Wrote fit artifacts to folder: {cfg.noise_model_fit_folder}")





def _maybe_int(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def _unpack_params(params: np.ndarray, n_channels: int, n_coherent: int) -> tuple[np.ndarray, np.ndarray]:
    d = params[:n_channels]
    L = params[n_channels:].reshape(n_channels, n_coherent)
    return d, L


def _build_model(d: np.ndarray, L: np.ndarray) -> np.ndarray:
    return np.diag(d) + (L @ L.T)


def _canonicalize_loadings(L: np.ndarray) -> np.ndarray:
    if L.shape[1] == 0:
        return L

    powers = np.sum(L * L, axis=0)
    order = np.argsort(powers)[::-1]
    L = L[:, order]

    for idx in range(L.shape[1]):
        col = L[:, idx]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            L[:, idx] = -col
    return L


def _initial_guess(cov: np.ndarray, n_coherent: int) -> np.ndarray:
    n_channels = cov.shape[0]
    if n_coherent == 0:
        return np.clip(np.diag(cov), 0.0, None)

    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    top_vals = np.maximum(vals[:n_coherent], 0.0)
    L0 = vecs[:, :n_coherent] * np.sqrt(top_vals)[None, :]
    d0 = np.clip(np.diag(cov - (L0 @ L0.T)), 0.0, None)
    return np.concatenate([d0, L0.reshape(-1)])


def _fit_summary(
    sigma_data: np.ndarray,
    sigma_model: np.ndarray,
    sigma_coherent: np.ndarray,
    d: np.ndarray,
    result,
    n_coherent: int,
) -> dict:
    resid = sigma_data - sigma_model
    offdiag_mask = ~np.eye(sigma_data.shape[0], dtype=bool)

    trace_total = float(np.trace(sigma_data))
    trace_incoherent = float(np.sum(d))
    trace_coherent = float(np.trace(sigma_coherent))
    mse = float(np.mean(resid * resid))
    mse_offdiag = float(np.mean((resid[offdiag_mask]) ** 2))

    return {
        "n_channels": int(sigma_data.shape[0]),
        "n_coherent": int(n_coherent),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nit": int(getattr(result, "nit", -1)),
        "nfev": int(result.nfev),
        "njev": int(getattr(result, "njev", -1)),
        "mse": mse,
        "mse_offdiag": mse_offdiag,
        "trace_total": trace_total,
        "trace_incoherent": trace_incoherent,
        "trace_coherent": trace_coherent,
        "coherent_fraction_trace": float(trace_coherent / trace_total) if trace_total != 0 else np.nan,
    }



if __name__ == "__main__":
    main()
