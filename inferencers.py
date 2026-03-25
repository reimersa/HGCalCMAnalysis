import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import Optional

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

            c = b.df_targets.shape[1]
            all_chs = np.arange(c, dtype=np.int64)

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
                    rr = np.repeat(rows_train, c)
                    cc = np.tile(all_chs, len(rows_train))
                elif self.split == "test":
                    rr = np.repeat(rows_test, c)
                    cc = np.tile(all_chs, len(rows_test))
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
        mats[c] = mat
    return mats
