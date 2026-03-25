#! /eos/user/a/areimers/torch-env/bin/python

import argparse
import os

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore

from sklearn.model_selection import train_test_split # type: ignore

import classes
import dnn_models
import inferencers
import prepare_dnn_inputs
import functions_plot


"""
    adc_channel_indices = [x for x in range(cfg.nch)]
    target_columns = [f"adc_ch{idx:03}_pedsub{column_tag}" for idx in adc_channel_indices]
"""


def main():
    parser = argparse.ArgumentParser(description="Add DNN predictions/residuals to df_batch*.parquet (overwrite).")

    parser.add_argument("-r", "--run", type=int, default=112044)
    parser.add_argument("-p", "--pedestal-run", type=int, default=112044)

    parser.add_argument(
        "-m", "--modules",
        nargs="+",
        metavar="MOD",
        default=[
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
        ],
    )

    # --- DNN config / checkpoint ---
    parser.add_argument("-n", "--nodes", nargs="+", type=int, required=True, help="Nodes per hidden layer (must match training).")
    parser.add_argument("-d", "--dropout", type=float, default=0.0, help="Dropout rate (must match training).")
    parser.add_argument("-t", "--tag", type=str, default="", help="Model tag (must match training only for model_string; weights load regardless).")

    # naming
    parser.add_argument(
        "--column-tag",
        type=str,
        default="_dnn",
        help="Suffix tag used like your analytic version. Produces *_pred<column_tag> and *_resid<column_tag>.",
    )

    # inputs definition
    parser.add_argument(
        "--per-channel-cols",
        nargs="+",
        default=["channel_indices"],
        help="List-like per-channel columns in df_inputs. All other df_inputs columns are treated as per-event.",
    )

    # inference batching
    parser.add_argument(
        "--infer-batch",
        type=int,
        default=65536,
        help="Number of (event,channel) samples per forward pass when predicting (per-channel loop).",
    )


    parser.add_argument(
        "--plotfolder",
        type=str,
        required=True,
        help="Folder to save loss plots to.",
    )

    args = parser.parse_args()

    cfgs = [
        classes.AnalysisConfig(
            modulename=x,
            run=args.run,
            run_for_pedestal=args.pedestal_run,
            standardize_std=False,
            inputfoldertag="",
        )
        for x in args.modules
    ]

    for cfg in cfgs:
        inferencer = inferencers.AnalysisTruthInferencer(cfg=cfg)
        add_correction_dnn(cfg=cfg, inferencer=inferencer, nodes=args.nodes, dropout=args.dropout, tag=args.tag, column_tag=args.column_tag, per_channel_cols=args.per_channel_cols, infer_batch=args.infer_batch, plot_dir_loss=args.plotfolder)


def add_correction_dnn(cfg, inferencer, nodes: list[int], dropout: float, tag: str, column_tag: str, per_channel_cols: list[str], infer_batch: int, plot_dir_loss: str) -> None:
    print("Hello from add_correction_dnn()!")
    print(f"Loading checkpoint: {cfg.dnn_models_folder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    columns_to_predict = [f"adc_ch{i:03d}_pedsub{column_tag}" for i in range(cfg.nch)]
    adc_channel_indices = [x for x in range(cfg.nch)]

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):
        print(f"Probing input/target files from chunk {idx:03d}...")
        df_inputs = prepare_dnn_inputs.make_input_df(cfg=cfg, df=df_chunk, adc_channel_indices=adc_channel_indices, column_tag=column_tag)
        input_dim = df_inputs.shape[1]
        break



    # infer C and the base adc column names from targets
    C = cfg.nch
    print(f"Per-event cols: {input_dim-len(per_channel_cols)} | per-channel cols: {len(per_channel_cols)} | input_dim={input_dim} | C={C}")

    # --- model ---
    model = dnn_models.PerChannelDNN(input_dim=input_dim, nodes_per_layer=nodes, dropout_rate=dropout, tag=tag).to(device)
    state = torch.load(os.path.join(cfg.dnn_models_folder, model.get_model_string(), "dnn_best.pth"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    for idx, df_chunk in enumerate(inferencer.full_df_iter()):

        df_inputs = prepare_dnn_inputs.make_input_df(cfg=cfg, df=df_chunk, adc_channel_indices=adc_channel_indices, column_tag=column_tag)
        E = df_inputs.shape[0]

        x_evt = df_inputs.drop(columns=per_channel_cols).to_numpy(np.float32, copy=False)
        ch_mats = inferencers.matrices_from_per_channel_cols(per_channel_cols=per_channel_cols, df=df_inputs, nch=cfg.nch)

        # predictions [E, C]
        preds = np.full((E, C), np.nan, dtype=np.float32)

        # predict channel-by-channel (keeps memory bounded)
        with torch.no_grad():
            for ch in range(C):
                # build feature matrix for all events at this channel: [E, Fevt + Fch]
                ch_feats = [ch_mats[ccol][:, ch][:, None] for ccol in per_channel_cols]  # each [E,1]
                X = np.concatenate([x_evt] + ch_feats, axis=1).astype(np.float32, copy=False)  # [E, F]

                # torch inference in batches
                out = np.empty((E,), dtype=np.float32)
                for start in range(0, E, infer_batch):
                    stop = min(start + infer_batch, E)
                    xb = torch.from_numpy(X[start:stop]).to(device=device, dtype=torch.float32)
                    yb = model(xb).detach().float().cpu().numpy()
                    out[start:stop] = yb

                preds[:, ch] = out

        meas = df_chunk[columns_to_predict].to_numpy(np.float32, copy=False)
        resids = (meas - preds).astype(np.float32, copy=False)

        preds_df = pd.DataFrame(preds, index=df_chunk.index, columns=columns_to_predict).add_suffix(f"_pred_dnn")
        resids_df = pd.DataFrame(resids, index=df_chunk.index, columns=columns_to_predict).add_suffix(f"_resid_dnn")

        # drop old columns if present
        existing = [c for c in list(preds_df.columns) + list(resids_df.columns) if c in df_chunk.columns]
        if existing:
            df_chunk = df_chunk.drop(columns=existing)

        df_chunk = pd.concat([df_chunk, preds_df, resids_df], axis=1)

        outfilename = os.path.join(cfg.analysis_inputs_folder, f"df_batch{idx:03d}.parquet")
        df_chunk.to_parquet(outfilename, engine="pyarrow", index=True, compression="zstd")
        print(f"Wrote updated df with DNN predictions and residuals to {outfilename}, overwriting possibly existing columns in existing file.")

    print(f"Now plotting loss")
    functions_plot.plot_loss(modeldir=os.path.join(cfg.dnn_models_folder, model.get_model_string()), plot_dir=plot_dir_loss)


    print("Done.")


if __name__ == "__main__":
    main()