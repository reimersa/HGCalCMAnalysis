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

    def __init__(self, cfg, split: str = "train", per_channel_cols: Optional[list[str]] = None, require_weights: bool = False, event_fraction: float = 1.0):
        self.cfg = cfg
        self.name = "dnn"
        self.split = split  # "train" or "test" or ""/None for no split filtering
        self.per_channel_cols = per_channel_cols or ["channel_indices"]
        self.metadata_cols = ["source_run", "source_is_pedestal"]
        self.batches = classes.DNNBatchIter(cfg=cfg, require_weights=require_weights)
        self.require_weights = require_weights
        self.event_fraction = validate_event_fraction(event_fraction)
        self.per_event_cols = None  # to be set on first sample_iter call
        self.allch_col_idx = None

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
            df_inputs = self.apply_split(batch).full_inputs_df
            if self.event_fraction < 1.0:
                keep = event_keep_mask(df_inputs.index.to_numpy(np.int64), self.event_fraction)
                df_inputs = df_inputs.iloc[keep]
            yield df_inputs

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
        subsample_frac: Optional[float] = None,
    ):
        subsample_frac = self.event_fraction if subsample_frac is None else validate_event_fraction(subsample_frac)
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
                subsample_frac=subsample_frac,
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
                subsample_frac=subsample_frac,
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
                subsample_frac=subsample_frac,
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

    def _event_features_for_targets(self, X_evt, rows, target_channels):
        values = X_evt[rows]
        if self.allch_col_idx is None:
            self.allch_col_idx = adc_allch_col_indices(self.per_event_cols, self.cfg.nch)
        if self.allch_col_idx is None:
            return values

        values = values.copy()
        values[
            np.arange(values.shape[0], dtype=np.int64),
            self.allch_col_idx[target_channels],
        ] = np.float32(0.0)
        return values

    def _sample_iter_chunk_events(self, batch_samples: int=8192, include_targets: bool=True, epoch_seed=None, exclude_target_channels: Optional[list[int]] = None, include_weights: bool = False, include_channel_indices: bool = False, subsample_frac: float = 1.0):
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
                    rows_split = rows_train
                elif self.split == "test":
                    rows_split = rows_test
                else:
                    raise ValueError("split must be 'train' or 'test'")

                if subsample_frac < 1.0 and rows_split.size:
                    rows_split = rows_split[event_keep_mask(ev_ids[rows_split], subsample_frac)]

                rr = np.repeat(rows_split, target_chs.size)
                cc = np.tile(target_chs, len(rows_split))
            
                if rr.size == 0:
                    continue
                
                # per-channel features: each -> [N, 1]
                ch_feats = [ch_mats[col][rr, cc][:, None] for col in self.per_channel_cols]
            
                # x: [N, Fevt + Fch]
                x_evt = self._event_features_for_targets(X_evt, rr, cc)
                x = np.concatenate([x_evt] + ch_feats, axis=1).astype(np.float32, copy=False)
            
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
        subsample_frac: float = 1.0,
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

            if subsample_frac < 1.0:
                rows_selected = rows_selected[event_keep_mask(ev_ids[rows_selected], subsample_frac)]

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
                x_evt = self._event_features_for_targets(X_evt, rr, cc)
                x = np.concatenate([x_evt] + ch_feats, axis=1).astype(np.float32, copy=False)

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
        subsample_frac: float = 1.0,
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

            if subsample_frac < 1.0:
                rows_selected = rows_selected[event_keep_mask(ev_ids[rows_selected], subsample_frac)]

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
                x_evt = self._event_features_for_targets(X_evt, rr, cc)
                x = np.concatenate([x_evt] + ch_feats, axis=1).astype(np.float32, copy=False)
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
                if self.allch_col_idx is None:
                    self.allch_col_idx = adc_allch_col_indices(self.per_event_cols, self.cfg.nch)
                if self.allch_col_idx is not None:
                    x_evt_bc = x_evt_bc.copy()
                    channel_indices = np.arange(self.cfg.nch, dtype=np.int64)
                    x_evt_bc[:, channel_indices, self.allch_col_idx] = np.float32(0.0)
    
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


def adc_allch_col_indices(per_event_cols, nch):
    """Return target-channel -> all-channel feature position, or None for legacy inputs."""
    expected = [f"adc_allch_{channel:03d}" for channel in range(nch)]
    present = [column for column in expected if column in per_event_cols]
    if not present:
        return None
    if len(present) != nch:
        missing = [column for column in expected if column not in per_event_cols]
        raise KeyError(
            f"Incomplete all-channel DNN input block: found {len(present)}/{nch}; "
            f"missing starts with {missing[:3]}."
        )
    return np.asarray([per_event_cols.index(column) for column in expected], dtype=np.int64)


class CombinedAnalysisDNNInferencer:
    """Stream independently prepared module datasets into one training iterator."""

    def __init__(self, cfgs, split="train", per_channel_cols=None, require_weights=False, event_fractions=None):
        self.cfgs = list(cfgs)
        if not self.cfgs:
            raise ValueError("CombinedAnalysisDNNInferencer requires at least one module.")
        self.cfg = self.cfgs[0]
        for cfg in self.cfgs[1:]:
            if cfg.nch != self.cfg.nch:
                raise ValueError(
                    "All modules in one DNN must have the same channel count: "
                    f"{self.cfg.modulename}={self.cfg.nch}, {cfg.modulename}={cfg.nch}."
                )
        self.per_channel_cols = per_channel_cols or ["channel_indices"]
        event_fractions = event_fractions or {}
        unknown_modules = sorted(set(event_fractions) - {cfg.modulename for cfg in self.cfgs})
        if unknown_modules:
            raise KeyError(f"Event fractions were provided for unknown modules: {unknown_modules}")
        self.event_fractions = {
            cfg.modulename: validate_event_fraction(event_fractions.get(cfg.modulename, 1.0))
            for cfg in self.cfgs
        }
        self.sources = [
            AnalysisDNNInferencer(
                cfg=cfg,
                split=split,
                per_channel_cols=self.per_channel_cols,
                require_weights=require_weights,
                event_fraction=self.event_fractions[cfg.modulename],
            )
            for cfg in self.cfgs
        ]
        self.per_event_cols = None
        self.allch_col_idx = None
        self._validate_shared_event_splits()

    def _validate_shared_event_splits(self):
        labels_by_event = {}
        for source in self.sources:
            for event_id, label in source.split_map.items():
                previous = labels_by_event.setdefault(event_id, label)
                if previous != label:
                    raise ValueError(
                        "The same event ID is assigned to different DNN splits across modules: "
                        f"event={event_id}, labels={previous!r}/{label!r}."
                    )

    def _capture_feature_order(self, source):
        if source.per_event_cols is None:
            return
        if self.per_event_cols is None:
            self.per_event_cols = list(source.per_event_cols)
        elif self.per_event_cols != list(source.per_event_cols):
            raise ValueError(
                "DNN per-event feature order differs between modules: "
                f"expected {self.per_event_cols}, got {source.per_event_cols}."
            )

    def full_inputs_iter(self):
        for source in self.sources:
            yield from source.full_inputs_iter()

    def sample_iter(self, epoch_seed=None, **kwargs):
        shuffle_mode = kwargs.get("shuffle_mode", "chunk_events")
        if shuffle_mode == "buffered_chunk_events":
            buffered_kwargs = dict(kwargs)
            buffered_kwargs.pop("shuffle_mode", None)
            buffered_kwargs.pop("shuffle_buffer_samples", None)
            yield from self._sample_iter_buffered_chunk_events(
                epoch_seed=epoch_seed,
                **buffered_kwargs,
            )
            return

        generators = []
        for source_index, source in enumerate(self.sources):
            source_seed = None if epoch_seed is None else int(epoch_seed) + source_index * 1000003
            generators.append(source.sample_iter(epoch_seed=source_seed, **kwargs))

        active = list(range(len(generators)))
        rng = np.random.default_rng(0 if epoch_seed is None else epoch_seed)
        while active:
            order = rng.permutation(active) if epoch_seed is not None else np.asarray(active)
            exhausted = []
            for source_index in order:
                source_index = int(source_index)
                try:
                    batch = next(generators[source_index])
                except StopIteration:
                    exhausted.append(source_index)
                    continue
                self._capture_feature_order(self.sources[source_index])
                yield batch
            if exhausted:
                exhausted_set = set(exhausted)
                active = [index for index in active if index not in exhausted_set]

    def _sample_iter_buffered_chunk_events(
        self,
        batch_samples=8192,
        include_targets=True,
        epoch_seed=None,
        shuffle_buffer_chunks=1,
        exclude_target_channels=None,
        include_weights=False,
        include_channel_indices=False,
        subsample_frac=None,
    ):
        common_event_fraction = (
            None
            if subsample_frac is None
            else validate_event_fraction(subsample_frac)
        )
        if self.sources[0].split not in ("train", "test"):
            raise ValueError(
                "split must be 'train' or 'test' for buffered_chunk_events shuffling"
            )

        batch_samples = int(batch_samples)
        shuffle_buffer_chunks = int(shuffle_buffer_chunks)
        if batch_samples <= 0:
            raise ValueError("batch_samples must be positive")
        if shuffle_buffer_chunks <= 0:
            raise ValueError("shuffle_buffer_chunks must be positive")
        if include_weights and not all(source.require_weights for source in self.sources):
            raise RuntimeError(
                "sample_iter(include_weights=True) requires weights for every module."
            )

        records = [
            (source_index, chunk_index)
            for source_index, source in enumerate(self.sources)
            for chunk_index in range(len(source.batches.inputfiles))
        ]
        rng = np.random.default_rng(0 if epoch_seed is None else epoch_seed)
        record_order = (
            rng.permutation(len(records))
            if epoch_seed is not None
            else np.arange(len(records), dtype=np.int64)
        )

        for start_chunk in range(0, len(records), shuffle_buffer_chunks):
            group_indices = record_order[start_chunk:start_chunk + shuffle_buffer_chunks]
            input_frames = []
            target_frames = []
            weight_frames = []

            for group_index in group_indices:
                source_index, chunk_index = records[int(group_index)]
                source = self.sources[source_index]
                df_inputs = pd.read_parquet(source.batches.inputfiles[chunk_index])
                df_targets = pd.read_parquet(source.batches.targetfiles[chunk_index])
                if not df_inputs.index.equals(df_targets.index):
                    raise ValueError(
                        f"DNN inputs and targets have different event indices for "
                        f"module {source.cfg.modulename!r}, chunk {chunk_index}."
                    )
                target_index = df_targets.index

                per_event_cols = [
                    column
                    for column in df_inputs.columns
                    if column not in self.per_channel_cols
                    and column not in source.metadata_cols
                ]
                source.per_event_cols = per_event_cols
                self._capture_feature_order(source)

                event_ids = df_inputs.index.to_numpy(np.int64, copy=False)
                split_labels = pd.Index(event_ids).map(source.split_map).to_numpy()
                valid_split = (split_labels == "train") | (split_labels == "test")
                if not np.all(valid_split):
                    bad = event_ids[~valid_split]
                    raise KeyError(
                        f"Unknown split label for module {source.cfg.modulename!r} "
                        f"(showing up to 10): {bad[:10]}"
                    )
                keep = split_labels == source.split
                event_fraction = (
                    source.event_fraction
                    if common_event_fraction is None
                    else common_event_fraction
                )
                if event_fraction < 1.0:
                    keep &= event_keep_mask(event_ids, event_fraction)
                if not np.any(keep):
                    continue

                df_inputs = df_inputs.iloc[keep]
                df_targets = df_targets.iloc[keep]
                input_frames.append(df_inputs)
                target_frames.append(df_targets)

                if include_weights:
                    if len(source.batches.weightfiles) != len(source.batches.inputfiles):
                        raise RuntimeError(
                            f"DNN weights are missing for module {source.cfg.modulename!r}. "
                            "Rerun prepare_dnn_inputs.py."
                        )
                    df_weights = pd.read_parquet(source.batches.weightfiles[chunk_index])
                    if not df_weights.index.equals(target_index):
                        raise ValueError(
                            f"DNN weights and targets have different event indices for "
                            f"module {source.cfg.modulename!r}, chunk {chunk_index}."
                        )
                    if list(df_weights.columns) != list(df_targets.columns):
                        raise ValueError(
                            f"DNN weight columns do not match targets for "
                            f"module {source.cfg.modulename!r}, chunk {chunk_index}."
                        )
                    weight_frames.append(df_weights.iloc[keep])

            if not input_frames:
                continue

            df_inputs = pd.concat(input_frames, axis=0)
            df_targets = pd.concat(target_frames, axis=0)
            df_weights = pd.concat(weight_frames, axis=0) if include_weights else None

            rows_selected = rng.permutation(len(df_inputs))
            X_evt = df_inputs[self.per_event_cols].to_numpy(np.float32, copy=False)
            ch_mats = matrices_from_per_channel_cols(
                per_channel_cols=self.per_channel_cols,
                df=df_inputs,
                nch=self.cfg.nch,
            )
            Y_evt = df_targets.to_numpy(np.float32, copy=False) if include_targets else None
            W_evt = df_weights.to_numpy(np.float32, copy=False) if include_weights else None
            target_channels = self.sources[0]._target_channels(
                n_channels=df_targets.shape[1],
                exclude_target_channels=exclude_target_channels,
            )
            if self.allch_col_idx is None:
                self.allch_col_idx = adc_allch_col_indices(
                    self.per_event_cols,
                    self.cfg.nch,
                )

            for start_event in range(0, rows_selected.size, batch_samples):
                event_rows = rows_selected[start_event:start_event + batch_samples]
                rows = np.repeat(event_rows, target_channels.size)
                channels = np.tile(target_channels, event_rows.size)
                if rows.size == 0:
                    continue

                event_features = X_evt[rows]
                if self.allch_col_idx is not None:
                    event_features = event_features.copy()
                    event_features[
                        np.arange(rows.size, dtype=np.int64),
                        self.allch_col_idx[channels],
                    ] = np.float32(0.0)
                channel_features = [
                    ch_mats[column][rows, channels][:, None]
                    for column in self.per_channel_cols
                ]
                x = np.concatenate(
                    [event_features] + channel_features,
                    axis=1,
                ).astype(np.float32, copy=False)

                if include_targets:
                    y = Y_evt[rows, channels].astype(np.float32, copy=False)
                    if include_weights:
                        weights = W_evt[rows, channels].astype(np.float32, copy=False)
                        if include_channel_indices:
                            yield x, y, weights, channels.astype(np.int64, copy=False)
                        else:
                            yield x, y, weights
                    elif include_channel_indices:
                        yield x, y, channels.astype(np.int64, copy=False)
                    else:
                        yield x, y
                elif include_channel_indices:
                    yield x, channels.astype(np.int64, copy=False)
                else:
                    yield x


def validate_event_fraction(value):
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"DNN event fractions must be in (0, 1], got {value}.")
    return value


def event_keep_mask(event_ids, fraction):
    """Deterministically retain a fraction of physical event IDs."""
    fraction = validate_event_fraction(fraction)
    if fraction >= 1.0:
        return np.ones(np.asarray(event_ids).shape, dtype=bool)
    values = np.asarray(event_ids, dtype=np.uint64)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values = values ^ (values >> np.uint64(31))
    uniform = (values >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return uniform < fraction
