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

    def __init__(self, cfg, split: str = "train", per_channel_cols: Optional[list[str]] = None, require_weights: bool = False):
        self.cfg = cfg
        self.name = "dnn"
        self.split = split  # "train" or "test" or ""/None for no split filtering
        self.per_channel_cols = per_channel_cols or ["channel_indices"]
        self.metadata_cols = ["source_run", "source_is_pedestal"]
        self.batches = classes.DNNBatchIter(cfg=cfg, require_weights=require_weights)
        self.require_weights = require_weights
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
        df_weights = batch.df_weights

        ev_inputs = df_inputs.index.to_numpy(np.int64, copy=False)
        
        # no split filtering requested -> return as-is
        if not self.split:
            return classes.DNNBatch(cfg=batch.cfg, df_inputs=df_inputs, df_targets=df_targets, df_weights=df_weights)

        # --- filter events by split (based on inputs index) ---
        split_labels = pd.Index(ev_inputs).map(self.split_map)
        keep_mask = np.asarray(split_labels == self.split, dtype=bool)

        df_inputs_f  = df_inputs.iloc[keep_mask]
        df_targets_f = df_targets.loc[df_inputs_f.index]
        df_weights_f = None if df_weights is None else df_weights.loc[df_inputs_f.index]

        return classes.DNNBatch(cfg=batch.cfg, df_inputs=df_inputs_f, df_targets=df_targets_f, df_weights=df_weights_f)

    # --- iterators in the "TruthInferencer style" ---
    def full_inputs_iter(self):
        for batch in self.batches:
            yield self.apply_split(batch).full_inputs_df

    def sample_iter(
        self,
        batch_samples: int=8192,
        include_targets: bool=True,
        epoch_seed=None,
        shuffle_mode: str = "chunk_events",
        shuffle_buffer_samples: int = 200_000,
        shuffle_buffer_chunks: int = 1,
        exclude_target_channels: Optional[list[int]] = None,
        include_weights: bool = False,
        include_channel_indices: bool = False,
    ):
        if include_weights and not self.require_weights:
            raise RuntimeError("sample_iter(include_weights=True) requires AnalysisDNNInferencer(require_weights=True).")
        if shuffle_mode == "chunk_events":
            yield from self._sample_iter_chunk_events(
                batch_samples=batch_samples,
                include_targets=include_targets,
                epoch_seed=epoch_seed,
                exclude_target_channels=exclude_target_channels,
                include_weights=include_weights,
                include_channel_indices=include_channel_indices,
            )
            return
        if shuffle_mode == "buffered_chunk_events":
            yield from self._sample_iter_buffered_chunk_events(
                batch_samples=batch_samples,
                include_targets=include_targets,
                epoch_seed=epoch_seed,
                shuffle_buffer_chunks=shuffle_buffer_chunks,
                exclude_target_channels=exclude_target_channels,
                include_weights=include_weights,
                include_channel_indices=include_channel_indices,
            )
            return
        if shuffle_mode == "global_samples":
            yield from self._sample_iter_global_samples(
                batch_samples=batch_samples,
                include_targets=include_targets,
                epoch_seed=epoch_seed,
                shuffle_buffer_samples=shuffle_buffer_samples,
                exclude_target_channels=exclude_target_channels,
                include_weights=include_weights,
                include_channel_indices=include_channel_indices,
            )
            return
        raise ValueError("shuffle_mode must be 'chunk_events', 'buffered_chunk_events', or 'global_samples'")

    def _target_channels(self, n_channels: int, exclude_target_channels: Optional[list[int]] = None) -> np.ndarray:
        if not exclude_target_channels:
            return np.arange(n_channels, dtype=np.int64)

        excluded = {int(ch) for ch in exclude_target_channels}
        bad = [ch for ch in excluded if ch < 0 or ch >= n_channels]
        if bad:
            raise ValueError(f"Excluded target channels are outside [0, {n_channels}): {bad[:10]}")

        target_chs = np.asarray([ch for ch in range(n_channels) if ch not in excluded], dtype=np.int64)
        if target_chs.size == 0:
            raise ValueError("All target channels were excluded.")
        return target_chs

    def _sample_iter_chunk_events(self, batch_samples: int=8192, include_targets: bool=True, epoch_seed=None, exclude_target_channels: Optional[list[int]] = None, include_weights: bool = False, include_channel_indices: bool = False):
        for batch in self.batches:
            b = batch

            # if len(b.df_inputs) == 0 or len(b.df_shuffle) == 0:
            if len(b.df_inputs) == 0:
                continue
            
            if self.per_event_cols is None:
                self.per_event_cols = [c for c in b.df_inputs.columns if c not in self.per_channel_cols and c not in self.metadata_cols]

            # per-event inputs
            X_evt = b.df_inputs[self.per_event_cols].to_numpy(np.float32, copy=False)  # [Nevt, 12]

            # materialize each per-channel list-column to dense [Nevt, C] once per chunk
            ch_mats = matrices_from_per_channel_cols(per_channel_cols=self.per_channel_cols, df=b.df_inputs, nch=self.cfg.nch)

            # per-event targets (optional)
            if include_targets:
                Y_evt = b.df_targets.to_numpy(np.float32, copy=False)  # [Nevt, C]
                if include_weights:
                    if b.df_weights is None:
                        raise RuntimeError("Requested DNN sample weights, but this batch has no weights.")
                    if b.df_weights.shape != b.df_targets.shape:
                        raise ValueError(f"DNN weight shape mismatch: weights={b.df_weights.shape}, targets={b.df_targets.shape}")
                    W_evt = b.df_weights.to_numpy(np.float32, copy=False)

            c = b.df_targets.shape[1]
            target_chs = self._target_channels(n_channels=c, exclude_target_channels=exclude_target_channels)

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
                    rr = np.repeat(rows_train, target_chs.size)
                    cc = np.tile(target_chs, len(rows_train))
                elif self.split == "test":
                    rr = np.repeat(rows_test, target_chs.size)
                    cc = np.tile(target_chs, len(rows_test))
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
                    if include_weights:
                        w = W_evt[rr, cc].astype(np.float32, copy=False)
                        if include_channel_indices:
                            yield x, y, w, cc.astype(np.int64, copy=False)
                        else:
                            yield x, y, w
                    else:
                        if include_channel_indices:
                            yield x, y, cc.astype(np.int64, copy=False)
                        else:
                            yield x, y
                else:
                    if include_channel_indices:
                        yield x, cc.astype(np.int64, copy=False)
                    else:
                        yield x


    def _sample_iter_buffered_chunk_events(
        self,
        batch_samples: int = 8192,
        include_targets: bool = True,
        epoch_seed=None,
        shuffle_buffer_chunks: int = 1,
        exclude_target_channels: Optional[list[int]] = None,
        include_weights: bool = False,
        include_channel_indices: bool = False,
    ):
        if self.split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test' for buffered_chunk_events shuffling")

        batch_samples = int(batch_samples)
        shuffle_buffer_chunks = int(shuffle_buffer_chunks)
        if batch_samples <= 0:
            raise ValueError("batch_samples must be positive")
        if shuffle_buffer_chunks <= 0:
            raise ValueError("shuffle_buffer_chunks must be positive")

        rng = np.random.default_rng(0 if epoch_seed is None else epoch_seed)
        n_chunks = len(self.batches.inputfiles)
        chunk_order = rng.permutation(n_chunks) if epoch_seed is not None else np.arange(n_chunks, dtype=np.int64)

        for start_chunk in range(0, n_chunks, shuffle_buffer_chunks):
            chunk_group = chunk_order[start_chunk:start_chunk + shuffle_buffer_chunks]
            input_frames = []
            target_frames = []
            weight_frames = []

            for chunk_idx in chunk_group:
                input_frames.append(pd.read_parquet(self.batches.inputfiles[int(chunk_idx)]))
                target_frames.append(pd.read_parquet(self.batches.targetfiles[int(chunk_idx)]))
                if include_weights:
                    if len(self.batches.weightfiles) != len(self.batches.inputfiles):
                        raise RuntimeError(
                            "Requested DNN sample weights, but weights_chunk*.parquet sidecars are missing. "
                            "Rerun prepare_dnn_inputs.py."
                        )
                    df_weights = pd.read_parquet(self.batches.weightfiles[int(chunk_idx)])
                    if df_weights.shape != target_frames[-1].shape:
                        raise ValueError(f"DNN weight shape mismatch for {self.batches.weightfiles[int(chunk_idx)]}: weights={df_weights.shape} targets={target_frames[-1].shape}")
                    if not df_weights.index.equals(target_frames[-1].index) or list(df_weights.columns) != list(target_frames[-1].columns):
                        raise ValueError(f"DNN weights in {self.batches.weightfiles[int(chunk_idx)]} do not match target index/columns.")
                    weight_frames.append(df_weights)

            df_inputs = pd.concat(input_frames, axis=0)
            df_targets = pd.concat(target_frames, axis=0)
            df_weights = pd.concat(weight_frames, axis=0) if include_weights else None
            if len(df_inputs) == 0:
                continue

            if self.per_event_cols is None:
                self.per_event_cols = [c for c in df_inputs.columns if c not in self.per_channel_cols and c not in self.metadata_cols]

            ev_ids = df_inputs.index.to_numpy(np.int64, copy=False)
            ev_split = pd.Index(ev_ids).map(self.split_map).to_numpy()
            is_train_evt = (ev_split == "train")
            is_test_evt = (ev_split == "test")
            if not np.all(is_train_evt | is_test_evt):
                bad = ev_ids[~(is_train_evt | is_test_evt)]
                raise KeyError(f"Unknown split label for events (showing up to 10): {bad[:10]}")

            if self.split == "train":
                rows_selected = np.flatnonzero(is_train_evt)
            else:
                rows_selected = np.flatnonzero(is_test_evt)

            if rows_selected.size == 0:
                continue

            rows_selected = rng.permutation(rows_selected)
            X_evt = df_inputs[self.per_event_cols].to_numpy(np.float32, copy=False)
            ch_mats = matrices_from_per_channel_cols(per_channel_cols=self.per_channel_cols, df=df_inputs, nch=self.cfg.nch)

            if include_targets:
                Y_evt = df_targets.to_numpy(np.float32, copy=False)
                if include_weights:
                    W_evt = df_weights.to_numpy(np.float32, copy=False)

            c = df_targets.shape[1]
            target_chs = self._target_channels(n_channels=c, exclude_target_channels=exclude_target_channels)

            for start_evt in range(0, rows_selected.size, batch_samples):
                rows_this = rows_selected[start_evt:start_evt + batch_samples]
                rr = np.repeat(rows_this, target_chs.size)
                cc = np.tile(target_chs, len(rows_this))
                if rr.size == 0:
                    continue

                ch_feats = [ch_mats[col][rr, cc][:, None] for col in self.per_channel_cols]
                x = np.concatenate([X_evt[rr]] + ch_feats, axis=1).astype(np.float32, copy=False)

                if include_targets:
                    y = Y_evt[rr, cc].astype(np.float32, copy=False)
                    if include_weights:
                        w = W_evt[rr, cc].astype(np.float32, copy=False)
                        if include_channel_indices:
                            yield x, y, w, cc.astype(np.int64, copy=False)
                        else:
                            yield x, y, w
                    else:
                        if include_channel_indices:
                            yield x, y, cc.astype(np.int64, copy=False)
                        else:
                            yield x, y
                else:
                    if include_channel_indices:
                        yield x, cc.astype(np.int64, copy=False)
                    else:
                        yield x


    def _sample_iter_global_samples(
        self,
        batch_samples: int = 8192,
        include_targets: bool = True,
        epoch_seed=None,
        shuffle_buffer_samples: int = 200_000,
        exclude_target_channels: Optional[list[int]] = None,
        include_weights: bool = False,
        include_channel_indices: bool = False,
    ):
        if self.split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test' for global_samples shuffling")

        rng = np.random.default_rng(0 if epoch_seed is None else epoch_seed)
        batch_samples = int(batch_samples)
        shuffle_buffer_samples = int(shuffle_buffer_samples)
        if batch_samples <= 0:
            raise ValueError("batch_samples must be positive")
        if shuffle_buffer_samples <= 0:
            raise ValueError("shuffle_buffer_samples must be positive")

        x_buf = []
        y_buf = []
        w_buf = []
        c_buf = []
        n_buffered = 0

        def emit_buffer(final_flush: bool = False):
            nonlocal x_buf, y_buf, w_buf, c_buf, n_buffered
            if n_buffered == 0:
                return
            if (not final_flush) and n_buffered < shuffle_buffer_samples:
                return

            x_all = np.concatenate(x_buf, axis=0)
            rows = rng.permutation(x_all.shape[0])
            x_all = x_all[rows]
            if include_targets:
                y_all = np.concatenate(y_buf, axis=0)[rows]
                if include_weights:
                    w_all = np.concatenate(w_buf, axis=0)[rows]
            if include_channel_indices:
                c_all = np.concatenate(c_buf, axis=0)[rows]

            for start in range(0, x_all.shape[0], batch_samples):
                stop = min(start + batch_samples, x_all.shape[0])
                if include_targets:
                    if include_weights:
                        if include_channel_indices:
                            yield x_all[start:stop], y_all[start:stop], w_all[start:stop], c_all[start:stop]
                        else:
                            yield x_all[start:stop], y_all[start:stop], w_all[start:stop]
                    else:
                        if include_channel_indices:
                            yield x_all[start:stop], y_all[start:stop], c_all[start:stop]
                        else:
                            yield x_all[start:stop], y_all[start:stop]
                else:
                    if include_channel_indices:
                        yield x_all[start:stop], c_all[start:stop]
                    else:
                        yield x_all[start:stop]

            x_buf = []
            y_buf = []
            w_buf = []
            c_buf = []
            n_buffered = 0

        n_chunks = len(self.batches.inputfiles)
        chunk_order = rng.permutation(n_chunks)
        for chunk_idx in chunk_order:
            df_inputs = pd.read_parquet(self.batches.inputfiles[int(chunk_idx)])
            df_targets = pd.read_parquet(self.batches.targetfiles[int(chunk_idx)])
            df_weights = None
            if include_weights:
                if len(self.batches.weightfiles) != len(self.batches.inputfiles):
                    raise RuntimeError(
                        "Requested DNN sample weights, but weights_chunk*.parquet sidecars are missing. "
                        "Rerun prepare_dnn_inputs.py."
                    )
                df_weights = pd.read_parquet(self.batches.weightfiles[int(chunk_idx)])
                if df_weights.shape != df_targets.shape:
                    raise ValueError(f"DNN weight shape mismatch for {self.batches.weightfiles[int(chunk_idx)]}: weights={df_weights.shape}, targets={df_targets.shape}")
                if not df_weights.index.equals(df_targets.index) or list(df_weights.columns) != list(df_targets.columns):
                    raise ValueError(f"DNN weights in {self.batches.weightfiles[int(chunk_idx)]} do not match target index/columns.")

            if len(df_inputs) == 0:
                continue

            if self.per_event_cols is None:
                self.per_event_cols = [c for c in df_inputs.columns if c not in self.per_channel_cols and c not in self.metadata_cols]

            ev_ids = df_inputs.index.to_numpy(np.int64, copy=False)
            ev_split = pd.Index(ev_ids).map(self.split_map).to_numpy()
            is_train_evt = (ev_split == "train")
            is_test_evt = (ev_split == "test")
            if not np.all(is_train_evt | is_test_evt):
                bad = ev_ids[~(is_train_evt | is_test_evt)]
                raise KeyError(f"Unknown split label for events (showing up to 10): {bad[:10]}")

            if self.split == "train":
                rows_selected = np.flatnonzero(is_train_evt)
            else:
                rows_selected = np.flatnonzero(is_test_evt)

            if rows_selected.size == 0:
                continue

            rows_selected = rng.permutation(rows_selected)
            X_evt = df_inputs[self.per_event_cols].to_numpy(np.float32, copy=False)
            ch_mats = matrices_from_per_channel_cols(per_channel_cols=self.per_channel_cols, df=df_inputs, nch=self.cfg.nch)

            if include_targets:
                Y_evt = df_targets.to_numpy(np.float32, copy=False)
                if include_weights:
                    W_evt = df_weights.to_numpy(np.float32, copy=False)

            c = df_targets.shape[1]
            target_chs = self._target_channels(n_channels=c, exclude_target_channels=exclude_target_channels)
            max_events_per_block = max(1, shuffle_buffer_samples // target_chs.size)

            for start_evt in range(0, rows_selected.size, max_events_per_block):
                rows_block = rows_selected[start_evt:start_evt + max_events_per_block]
                rr = np.repeat(rows_block, target_chs.size)
                cc = np.tile(target_chs, len(rows_block))

                pair_order = rng.permutation(rr.size)
                rr = rr[pair_order]
                cc = cc[pair_order]

                ch_feats = [ch_mats[col][rr, cc][:, None] for col in self.per_channel_cols]
                x = np.concatenate([X_evt[rr]] + ch_feats, axis=1).astype(np.float32, copy=False)
                x_buf.append(x)
                if include_targets:
                    y = Y_evt[rr, cc].astype(np.float32, copy=False)
                    y_buf.append(y)
                    if include_weights:
                        w = W_evt[rr, cc].astype(np.float32, copy=False)
                        w_buf.append(w)
                if include_channel_indices:
                    c_buf.append(cc.astype(np.int64, copy=False))
                n_buffered += int(x.shape[0])

                yield from emit_buffer(final_flush=False)

        yield from emit_buffer(final_flush=True)



    def sample_iter_allchannels(self, batch_samples: int = 8192, include_targets: bool = True, epoch_seed=None):
        for batch in self.batches:
            b = batch
            if len(b.df_inputs) == 0:
                continue
    
            if self.per_event_cols is None:
                self.per_event_cols = [c for c in b.df_inputs.columns if c not in self.per_channel_cols and c not in self.metadata_cols]
    
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
