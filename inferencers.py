import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import Optional

from sklearn.model_selection import train_test_split # type: ignore

import classes



class AnalysisTruthInferencer:
    def __init__(self, cfg: classes.AnalysisConfig, selection=""):
        self.cfg = cfg
        self.name = "true"
        self.batches = classes.AnalysisBatchIter(cfg=cfg)
        self.selection = selection

    def full_df_iter(self):
        for batch in self.batches:
            batch_sel = batch.select_flag(flag_col=self.selection)
            yield batch_sel.full_df

    def df_cols_iter(self, colnames):
        def iterator():
            for batch in self.batches:
                batch_sel = batch.select_flag(flag_col=self.selection)
                yield batch_sel.df_cols(colnames)
        return iterator

    def pred_iter(self):
        for batch in self.batches:
            batch_sel = batch.select_flag(flag_col=self.selection)
            yield batch_sel.measurements_df

    def resid_iter(self): # same as pred, this is the truth and has no correction.
        for batch in self.batches:
            batch_sel = batch.select_flag(flag_col=self.selection)
            yield batch_sel.measurements_df

    def pred_with_cm_iter(self):
        for batch in self.batches:
            batch_sel = batch.select_flag(flag_col=self.selection)
            yield batch_sel.measurements_df_with_cm_df

    def resid_with_cm_iter(self):
        for batch in self.batches:
            batch_sel = batch.select_flag(flag_col=self.selection)
            yield batch_sel.measurements_df_with_cm_df

    def cm_iter(self):
        for batch in self.batches:
            batch_sel = batch.select_flag(flag_col=self.selection)
            yield batch_sel.cm_df

    def proj_pred_iter(self, vecs, k):
        u = vecs[:, k]
        for batch in self.pred_iter():
            X = batch.to_numpy(dtype=np.float64, copy=False)
            y = X @ u
            df = pd.DataFrame(y, index=batch.index, columns=[f"proj_mode_{k}"])
            yield df

class AnalysisDNNInferencer:
    """
    DNN analogue of AnalysisTruthInferencer.

    - Reads the written parquet chunks via DNNBatchIter
    - Applies event-level selection flag (optional, using a column in df_inputs)
    - Applies train/test split (event-level, using event_id_global + splitfile)
    - Provides iterators similar to TruthInferencer
    - Provides a sample iterator yielding per-(event,channel) samples for PerChannelDNN
    """

    def __init__(self, cfg, split: str = "train", per_channel_cols: Optional[list[str]] = None):
        self.cfg = cfg
        self.name = "dnn"
        self.split = split  # "train" or "test" or ""/None for no split filtering
        self.per_channel_cols = per_channel_cols or ["channel_indices"]
        self.batches = classes.DNNBatchIter(cfg=cfg)
        self.per_event_cols = None  # to be set on first sample_iter call

        split_path = os.path.join(cfg.dnn_training_input_folder, "event_split_train_test.parquet")
        df_split = pd.read_parquet(split_path)
        if "event_id_global" not in df_split.columns or "split" not in df_split.columns:
            raise KeyError(f"Split file {split_path} must have columns: event_id_global, split")

        # Map: event_id_global -> "train"/"test"
        self.split_map = dict(zip(df_split["event_id_global"].to_numpy(dtype=np.int64), df_split["split"].astype(str).to_numpy()))

    # def apply_split(self, batch: classes.DNNBatch) -> classes.DNNBatch:
    #     df_inputs  = batch.df_inputs
    #     df_targets = batch.df_targets
    #     df_shuffle = batch.df_shuffle

    #     if "event_id_global" not in df_shuffle.columns:
    #         raise KeyError("Shuffle df missing required column 'event_id_global'")

    #     # --- strict unknown-ID check on shuffle
    #     ev_shuffle = df_shuffle["event_id_global"].to_numpy(np.int64, copy=False)
    #     unknown = [e for e in set(map(int, np.unique(ev_shuffle))) if e not in self.split_map]
    #     if unknown:
    #         raise KeyError(f"Unknown event_id_global(s) not found in split map (showing up to 10): {unknown[:10]}")

    #     ev_inputs = df_inputs.index.to_numpy(np.int64, copy=False)
    #     unknown_inputs = [e for e in set(map(int, np.unique(ev_inputs))) if e not in self.split_map]
    #     if unknown_inputs:
    #         raise KeyError(f"Unknown event_id_global(s) in inputs index (showing up to 10): {unknown_inputs[:10]}")

    #     ev_targets = df_targets.index.to_numpy(np.int64, copy=False)
    #     unknown_targets = [e for e in set(map(int, np.unique(ev_targets))) if e not in self.split_map]
    #     if unknown_targets:
    #         raise KeyError(f"Unknown event_id_global(s) in targets index (showing up to 10): {unknown_targets[:10]}")

    #     # no split filtering requested -> return as-is
    #     if not self.split:
    #         return classes.DNNBatch(cfg=batch.cfg, df_inputs=df_inputs, df_targets=df_targets, df_shuffle=df_shuffle)

    #     # --- 1) filter events in inputs/targets by split ---
    #     keep_mask = np.asarray(pd.Index(ev_inputs).map(self.split_map) == self.split, dtype=bool)

    #     # indices (old row numbers) of kept events in this chunk
    #     kept_old_rows = np.nonzero(keep_mask)[0]               # e.g. [0,2,5,...] in 0..Nevt-1
    #     df_inputs_f   = df_inputs.iloc[keep_mask]
    #     df_targets_f  = df_targets.loc[df_inputs_f.index]
    #     df_shuffle   = df_shuffle[df_shuffle["event_id_global"].map(self.split_map) == self.split]

    #     # --- 2) remap shuffle's row_in_chunk to the new compressed row numbering ---
    #     # old -> new mapping; size ~ Nevt, cheap
    #     old_to_new = {int(old): int(new) for new, old in enumerate(kept_old_rows)}

    #     # keep only shuffle samples whose row_in_chunk survived, AND remap row_in_chunk
    #     rows = df_shuffle["row_in_chunk"].to_numpy(np.int64, copy=False)
    #     keep_samples = np.fromiter((int(r) in old_to_new for r in rows), dtype=bool, count=len(rows))
    #     df_shuffle_f = df_shuffle.loc[keep_samples].copy()

    #     # remap row_in_chunk in-place
    #     df_shuffle_f["row_in_chunk"] = df_shuffle_f["row_in_chunk"].map(old_to_new).astype(np.int64)

    #     return classes.DNNBatch(cfg=batch.cfg, df_inputs=df_inputs_f, df_targets=df_targets_f, df_shuffle=df_shuffle_f)

    def apply_split(self, batch: classes.DNNBatch) -> classes.DNNBatch:
        df_inputs  = batch.df_inputs
        df_targets = batch.df_targets

        ev_inputs = df_inputs.index.to_numpy(np.int64, copy=False)
        
        # no split filtering requested -> return as-is
        if not self.split:
            return classes.DNNBatch(cfg=batch.cfg, df_inputs=df_inputs, df_targets=df_targets)

        # --- filter events by split (based on inputs index) ---
        split_labels = pd.Index(ev_inputs).map(self.split_map)
        keep_mask = np.asarray(split_labels == self.split, dtype=bool)

        df_inputs_f  = df_inputs.iloc[keep_mask]
        df_targets_f = df_targets.loc[df_inputs_f.index]

        return classes.DNNBatch(cfg=batch.cfg, df_inputs=df_inputs_f, df_targets=df_targets_f)

    # --- iterators in the "TruthInferencer style" ---
    def full_inputs_iter(self):
        for batch in self.batches:
            yield self.apply_split(batch).full_inputs_df

    # def full_targets_iter(self):
    #     for batch in self.batches:
    #         yield self.apply_split(batch).full_targets_df

    # def shuffle_iter(self):
    #     for batch in self.batches:
    #         yield self.apply_split(batch).full_shuffle_df

    # --- iterator used for training ---
    # def sample_iter(self, batch_samples: int=8192, include_targets: bool=True, epoch_seed=None):
    #     """
    #     Yield per-(event,channel) batches compatible with PerChannelDNN:

    #         x: [N, F] where F = ncmchannels + 1 (channel index feature)
    #         y: [N]    target adc value for that sample

    #     Channel index feature is read from df_inputs["channel_indices"] per row
    #     """
    #     for batch in self.batches:
    #         b = self.apply_split(batch)
    #         # b = batch

    #         if len(b.df_inputs) == 0 or len(b.df_shuffle) == 0:
    #             continue
            
    #         if self.per_event_cols is None:
    #             self.per_event_cols = [c for c in b.df_inputs.columns if c not in self.per_channel_cols]

    #         # per-event inputs
    #         X_evt = b.df_inputs[self.per_event_cols].to_numpy(np.float32, copy=False)  # [Nevt, 12]

    #         # materialize each per-channel list-column to dense [Nevt, C] once per chunk
    #         ch_mats = matrices_from_per_channel_cols(per_channel_cols=self.per_channel_cols, df=b.df_inputs, nch=self.cfg.nch)

    #         # per-event targets (optional)
    #         if include_targets:
    #             Y_evt = b.df_targets.to_numpy(np.float32, copy=False)  # [Nevt, C]

    #         ### HACK: shuffle now, not using external file
    #         # all_indices = np.arange(b.df_targets.shape[0] * b.df_targets.shape[1])
    #         # train_indices, val_indices = train_test_split(
    #         #     all_indices,
    #         #     test_size=0.2,
    #         #     random_state=42,
    #         #     shuffle=True
    #         # )

    #         # if self.split == "train":
    #         #     rows = (train_indices // 222).astype(np.int32)
    #         #     chs = (train_indices % 222).astype(np.int32)
    #         # elif self.split == "test":
    #         #     rows = (val_indices // 222).astype(np.int32)
    #         #     chs = (val_indices % 222).astype(np.int32)
    #         # else:
    #         #     raise ValueError("???")

    #         # forced_channels = np.array([8, 17, 19, 28, 45, 54, 56, 65, 82, 91, 93, 102, 119, 128, 130, 139, 156, 165, 167, 176, 193, 202, 204, 213], dtype=np.int64)  
    #         # rows = np.arange(b.df_inputs.shape[0], dtype=np.int64)[:, None]          # (nrows, 1)
    #         # chs = forced_channels[None, :]                            # (1, nforced)
    #         # forced_indices = (rows * b.df_targets.shape[1] + chs).ravel()                # (nrows * nforced,)

    #         # # ------------------------------
    #         # # Remove forced channels from val/test
    #         # # ------------------------------
    #         # val_chs = val_indices % b.df_targets.shape[1]
    #         # val_indices = val_indices[~np.isin(val_chs, forced_channels)]

    #         # # ------------------------------
    #         # # Add forced channels to train
    #         # # ------------------------------
    #         # train_indices = np.unique(
    #         #     np.concatenate([train_indices, forced_indices])
    #         # )

    #         # # ------------------------------
    #         # # Final row / channel arrays
    #         # # ------------------------------
    #         # if self.split == "train":
    #         #     idx = train_indices
    #         # elif self.split == "test":
    #         #     idx = val_indices
    #         # else:
    #         #     raise ValueError("split must be 'train' or 'test'")

    #         # rows = (idx // b.df_targets.shape[1]).astype(np.int32)
    #         # chs  = (idx %  b.df_targets.shape[1]).astype(np.int32)

    #         #### HACK End

    #         # # shuffle-defined samples
    #         # rows = b.df_shuffle["row_in_chunk"].to_numpy(np.int64, copy=False)
    #         # chs  = b.df_shuffle["channel_id"].to_numpy(np.int64, copy=False)

    #         # # different shuffling every epoch when setting epoch_seed
    #         # if epoch_seed is not None:
    #         #     rng  = np.random.default_rng(epoch_seed)
    #         #     perm = rng.permutation(len(rows))
    #         #     rows = rows[perm]
    #         #     chs  = chs[perm]

    #         # n = len(rows)
    #         # for start in range(0, n, batch_samples):
    #         #     sl = slice(start, min(start + batch_samples, n))
    #         #     rr = rows[sl]
    #         #     cc = chs[sl]
    #             # print("picking row/channel:\n", rr, "\n", cc)

    #         # different shuffling every epoch when setting epoch_seed
    #         n = b.df_inputs.shape[0]
    #         c = b.df_targets.shape[1]
    #         chs = np.asarray([x for x in range(c)], dtype=np.int64)
    #         rows = np.asarray([x for x in range(n)], dtype=np.int64)

    #         if epoch_seed is not None:
    #             rng  = np.random.default_rng(epoch_seed)
    #             perm = rng.permutation(n)
    #             rows = rows[perm]

    #         for start in range(0, n, batch_samples):
    #             sl = slice(start, min(start + batch_samples, n))
    #             rows_thisslice = rows[sl]

    #             cc = np.tile(chs, len(rows_thisslice))
    #             rr = np.repeat(rows_thisslice, c)
    #             # print("picking row/channel:\n", rr, "\n", cc)

    #             # build per-channel features for these samples: each -> [N,1]
    #             ch_feats = [ch_mats[c][rr, cc][:, None] for c in self.per_channel_cols]

    #             # final x: [N, Fevt + Fch]
    #             x = np.concatenate([X_evt[rr]] + ch_feats, axis=1).astype(np.float32, copy=False)

    #             if include_targets:
    #                 y = Y_evt[rr, cc].astype(np.float32, copy=False)
    #                 yield x, y
    #             else:
    #                 yield x

    def sample_iter(self, batch_samples: int=8192, include_targets: bool=True, epoch_seed=None):
        for batch in self.batches:
            b = batch

            # if len(b.df_inputs) == 0 or len(b.df_shuffle) == 0:
            if len(b.df_inputs) == 0:
                continue
            
            if self.per_event_cols is None:
                self.per_event_cols = [c for c in b.df_inputs.columns if c not in self.per_channel_cols]

            # per-event inputs
            X_evt = b.df_inputs[self.per_event_cols].to_numpy(np.float32, copy=False)  # [Nevt, 12]

            # materialize each per-channel list-column to dense [Nevt, C] once per chunk
            ch_mats = matrices_from_per_channel_cols(per_channel_cols=self.per_channel_cols, df=b.df_inputs, nch=self.cfg.nch)

            # per-event targets (optional)
            if include_targets:
                Y_evt = b.df_targets.to_numpy(np.float32, copy=False)  # [Nevt, C]

            # ------------------------------
            # define special channels
            # ------------------------------
            special_channels = np.array(
                # [8, 17, 19, 28, 45, 54, 56, 65, 82, 91, 93, 102, 119, 128, 130, 139, 156, 165, 167, 176, 193, 202, 204, 213],
                [],
                dtype=np.int64,
            )
            special_set = set(map(int, special_channels.tolist()))
            
            c = b.df_targets.shape[1]
            all_chs = np.arange(c, dtype=np.int64)
            
            non_special_channels = np.array([ch for ch in all_chs if int(ch) not in special_set], dtype=np.int64)
            
            # ------------------------------
            # event split mask for this chunk
            # ------------------------------
            # df_inputs index is event_id_global
            ev_ids = b.df_inputs.index.to_numpy(np.int64, copy=False)
            
            # map to "train"/"test"
            ev_split = pd.Index(ev_ids).map(self.split_map).to_numpy()   # dtype=object/str
            is_train_evt = (ev_split == "train")
            is_test_evt  = (ev_split == "test")
            
            # sanity: you probably want strictness here
            if not np.all(is_train_evt | is_test_evt):
                bad = ev_ids[~(is_train_evt | is_test_evt)]
                raise KeyError(f"Unknown split label for events (showing up to 10): {bad[:10]}")
            
            # ------------------------------
            # epoch-level shuffle over EVENTS (keep your current semantics)
            # ------------------------------
            n = b.df_inputs.shape[0]
            rows_all = np.arange(n, dtype=np.int64)
            
            if epoch_seed is not None:
                rng = np.random.default_rng(epoch_seed)
                rows_all = rng.permutation(rows_all)
            
            # ------------------------------
            # iterate by event batches, but emit different channels depending on split & event-type
            # ------------------------------
            for start in range(0, n, batch_samples):
                sl = slice(start, min(start + batch_samples, n))
                rows_this = rows_all[sl]
            
                rows_train = rows_this[is_train_evt[rows_this]]
                rows_test  = rows_this[is_test_evt[rows_this]]
            
                if self.split == "train":
                    # 1) all channels for train events
                    rr1 = np.repeat(rows_train, c)
                    cc1 = np.tile(all_chs, len(rows_train))
                    
                    # 2) ONLY special channels for test events (migrated into train)
                    rr2 = np.repeat(rows_test, len(special_channels))
                    cc2 = np.tile(special_channels, len(rows_test))
                    
                    rr = np.concatenate([rr1, rr2], axis=0)
                    cc = np.concatenate([cc1, cc2], axis=0)
    
                    # # ONLY non-special channels for train events
                    # rr = np.repeat(rows_train, len(non_special_channels))
                    # cc = np.tile(non_special_channels, len(rows_train))
            
                elif self.split == "test":
                    # ONLY non-special channels for test events
                    rr = np.repeat(rows_test, len(non_special_channels))
                    cc = np.tile(non_special_channels, len(rows_test))
            
                else:
                    raise ValueError("split must be 'train' or 'test'")
            
                if rr.size == 0:
                    continue
                
                # per-channel features: each -> [N, 1]
                ch_feats = [ch_mats[col][rr, cc][:, None] for col in self.per_channel_cols]
            
                # x: [N, Fevt + Fch]
                x = np.concatenate([X_evt[rr]] + ch_feats, axis=1).astype(np.float32, copy=False)
            
                if include_targets:
                    y = Y_evt[rr, cc].astype(np.float32, copy=False)
                    yield x, y
                else:
                    yield x



    def sample_iter_allchannels(self, batch_samples: int = 8192, include_targets: bool = True, epoch_seed=None):
        for batch in self.batches:
            b = batch
            if len(b.df_inputs) == 0:
                continue
    
            if self.per_event_cols is None:
                self.per_event_cols = [c for c in b.df_inputs.columns if c not in self.per_channel_cols]
    
            # per-event inputs: [Nevt, Fevt]
            X_evt = b.df_inputs[self.per_event_cols].to_numpy(np.float32, copy=False)
    
            # per-channel list-cols materialized: dict(col -> [Nevt, C])
            ch_mats = matrices_from_per_channel_cols(per_channel_cols=self.per_channel_cols, df=b.df_inputs, nch=self.cfg.nch)
    
            # per-event targets (optional): [Nevt, C]
            if include_targets:
                Y_evt = b.df_targets.to_numpy(np.float32, copy=False)
    
    
            if b.df_targets.shape[1] != self.cfg.nch:
                raise ValueError(f"Got inconsistent number of channels: (b.df_targets.shape[1], self.cfg.nch) = ({b.df_targets.shape[1]}, {self.cfg.nch})")
    
            # ------------------------------
            # event split mask for this chunk
            # ------------------------------
            ev_ids = b.df_inputs.index.to_numpy(np.int64, copy=False)
            ev_split = pd.Index(ev_ids).map(self.split_map).to_numpy()
            is_train_evt = (ev_split == "train")
            is_test_evt  = (ev_split == "test")
    
            if not np.all(is_train_evt | is_test_evt):
                bad = ev_ids[~(is_train_evt | is_test_evt)]
                raise KeyError(f"Unknown split label for events (showing up to 10): {bad[:10]}")
    
            # ------------------------------
            # epoch-level shuffle over EVENTS
            # ------------------------------
            n = b.df_inputs.shape[0]
            rows_all = np.arange(n, dtype=np.int64)
            if epoch_seed is not None:
                rng = np.random.default_rng(epoch_seed)
                rows_all = rng.permutation(rows_all)
    
            # ------------------------------
            # iterate by event batches, emit full [B,C,F]
            # ------------------------------
            Fevt = X_evt.shape[1]
            Fch = len(self.per_channel_cols)
    
            for start in range(0, n, batch_samples):
                sl = slice(start, min(start + batch_samples, n))
                rows_this = rows_all[sl]
    
                rows_train = rows_this[is_train_evt[rows_this]]
                rows_test  = rows_this[is_test_evt[rows_this]]
    
                if self.split == "train":
                    rows = rows_train
                elif self.split == "test":
                    rows = rows_test
                else:
                    raise ValueError("split must be 'train' or 'test'")
    
                if rows.size == 0:
                    continue
    
                B = rows.size
    
                # x_evt: [B, Fevt]
                x_evt = X_evt[rows]
    
                # broadcast event features: [B, C, Fevt]
                x_evt_bc = np.broadcast_to(x_evt[:, None, :], (B, self.cfg.nch, Fevt)).astype(np.float32, copy=False)
    
                # stack per-channel features: [B, C, Fch]
                x_ch = np.stack([ch_mats[col][rows, :] for col in self.per_channel_cols], axis=2).astype(np.float32, copy=False)
    
                # concat: [B, C, F]
                x_cf = np.concatenate([x_evt_bc, x_ch], axis=2).astype(np.float32, copy=False)
    
                if include_targets:
                    y = Y_evt[rows, :]  # [B, C]
                    yield x_cf, y
                else:
                    yield x_cf

    

def matrices_from_per_channel_cols(per_channel_cols, df, nch):

    mats = {}
    for c in per_channel_cols:
        if c not in df.columns:
            raise KeyError(f"Missing per-channel column '{c}' in inputs df.")
        mat = np.vstack(df[c].to_numpy()).astype(np.float32, copy=False)
        # if mat.shape[1] != nch:
        #     raise ValueError(f"Column '{c}' has wrong length: expected {nch}, got {mat.shape[1]}")
        mats[c] = mat
    return mats