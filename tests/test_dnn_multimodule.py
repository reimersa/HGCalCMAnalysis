import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import add_correction_dnn
import inferencers
import prepare_dnn_inputs
import submit_train


class DNNFeatureTests(unittest.TestCase):
    def setUp(self):
        self.spec = prepare_dnn_inputs.make_feature_spec(
            feature_version=prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS,
            module_vocabulary=["module_a", "module_b"],
        )

    def test_reserved_unknown_index(self):
        self.assertEqual(prepare_dnn_inputs.module_onehot_index("module_a", self.spec), 0)
        self.assertEqual(prepare_dnn_inputs.module_onehot_index("module_b", self.spec), 1)
        self.assertEqual(prepare_dnn_inputs.module_onehot_index("unseen", self.spec), 2)
        self.assertEqual(len(prepare_dnn_inputs.module_onehot_columns(self.spec)), 3)

    def test_materialized_input_schemas_have_distinct_folders(self):
        first_spec = prepare_dnn_inputs.make_feature_spec(
            feature_version=prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS,
            module_vocabulary=["module_a01", "module_b02"],
        )
        larger_spec = prepare_dnn_inputs.make_feature_spec(
            feature_version=prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS,
            module_vocabulary=["module_a01", "module_b02", "module_c03"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            first_cfg = SimpleNamespace(dnn_training_input_folder=tmp)
            second_cfg = SimpleNamespace(dnn_training_input_folder=tmp)
            first = prepare_dnn_inputs.configure_dnn_input_folder(first_cfg, first_spec)
            second = prepare_dnn_inputs.configure_dnn_input_folder(second_cfg, larger_spec)

        self.assertNotEqual(first, second)
        self.assertEqual(os.path.dirname(first), tmp)
        self.assertEqual(os.path.dirname(second), tmp)

    def test_schema_folder_compacts_module_families(self):
        spec = prepare_dnn_inputs.make_feature_spec(
            feature_version=prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS,
            module_vocabulary=[
                "ML_F3WC_IH0180",
                "ML_F3WC_IH0182",
                "ML_F3WC_IH0190",
                "MH_F3WC_IH099",
                "MH_F3WC_IH100",
            ],
        )
        schema_id = prepare_dnn_inputs.dnn_input_schema_id(spec)
        self.assertEqual(
            schema_id,
            "all_channels_multimodule_v1__ML_F3WC_IH0180-82-90_MH_F3WC_IH099-100",
        )

    def test_schema_folder_rejects_unexpected_module_names(self):
        for modulename in ["module A01", "module/A01", "module_without_number"]:
            spec = prepare_dnn_inputs.make_feature_spec(
                feature_version=prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS,
                module_vocabulary=[modulename],
            )
            with self.subTest(modulename=modulename), self.assertRaisesRegex(
                ValueError,
                "DNN input folder",
            ):
                prepare_dnn_inputs.dnn_input_schema_id(spec)

    def test_multimodule_condor_name_stays_within_filename_limit(self):
        modules = [
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
        ]
        name = submit_train.make_job_name(
            modules=modules,
            run="112044_112050_112060_112073_adcmax10",
            selection_for_correction="selection_trigtime",
            nodes=[256, 256, 256, 32],
            dropout=0.0,
            epochs=200,
            weight_decay=0.0,
            modeltag="allchannels_multimodule",
            preprocess_inputs=False,
        )
        self.assertTrue(name.startswith("ML_F3WC_IH0180-82-90-91-92-94-96-97-98-99_"))
        self.assertLessEqual(len(name + ".sub"), 255)

    def test_event_fraction_is_deterministic(self):
        event_ids = np.arange(10000, dtype=np.int64)
        first = inferencers.event_keep_mask(event_ids, 0.25)
        second = inferencers.event_keep_mask(event_ids, 0.25)
        np.testing.assert_array_equal(first, second)
        self.assertLess(abs(float(first.mean()) - 0.25), 0.02)

    def test_all_channel_builder_and_target_mask(self):
        cfg = SimpleNamespace(modulename="unseen", run=1, ncmchannels=2, nch=3, nch_per_erx=1)
        df = pd.DataFrame(
            {
                "cm_erx00_pedsub": [0.1, 0.2],
                "cm_erx01_pedsub": [0.3, 0.4],
                "nchtoa": [0, 1],
                "nchtot": [0, 0],
                "nchadcgt10": [2, 2],
                "nchadcgt50": [0, 0],
                "nchadcgt200": [0, 0],
                "nchadcgt500": [0, 0],
                "adc_ch000_pedsub_nocut": [10.0, 11.0],
                "adc_ch001_pedsub_nocut": [20.0, 21.0],
                "adc_ch002_pedsub_nocut": [30.0, 31.0],
            }
        )
        with patch.object(
            prepare_dnn_inputs,
            "_load_cell_area_fractions",
            return_value=np.ones(3, dtype=np.float32),
        ):
            inputs = prepare_dnn_inputs.make_input_df(
                cfg=cfg,
                df=df,
                adc_channel_indices=[0, 1, 2],
                column_tag="",
                feature_spec=self.spec,
            )

        self.assertTrue(np.all(inputs["module_onehot_002"].to_numpy() == 1.0))
        per_channel = ["channel_indices", "erx_indices", "cell_area_fraction"]
        per_event = [
            column
            for column in inputs.columns
            if column not in per_channel
            and column not in prepare_dnn_inputs.INPUT_METADATA_COLUMNS
        ]
        matrices = list(
            add_correction_dnn.iter_feature_matrices(
                df_inputs=inputs,
                per_event_cols=per_event,
                per_channel_cols=per_channel,
                nch=3,
            )
        )
        allch_positions = inferencers.adc_allch_col_indices(per_event, 3)
        for target_channel, values in matrices:
            self.assertTrue(np.all(values[:, allch_positions[target_channel]] == 0.0))
            other = (target_channel + 1) % 3
            expected = df[f"adc_ch{other:03d}_pedsub_nocut"].to_numpy()
            np.testing.assert_allclose(values[:, allch_positions[other]], expected)

    def test_all_channel_builder_requires_uncut_columns(self):
        cfg = SimpleNamespace(modulename="module_a", run=1, ncmchannels=1, nch=1, nch_per_erx=1)
        df = pd.DataFrame(
            {
                "cm_erx00_pedsub": [0.0],
                "nchtoa": [0],
                "nchtot": [0],
                "nchadcgt10": [0],
                "nchadcgt50": [0],
                "nchadcgt200": [0],
                "nchadcgt500": [0],
            }
        )
        with patch.object(
            prepare_dnn_inputs,
            "_load_cell_area_fractions",
            return_value=np.ones(1, dtype=np.float32),
        ):
            with self.assertRaisesRegex(KeyError, "requires unmasked"):
                prepare_dnn_inputs.make_input_df(
                    cfg=cfg,
                    df=df,
                    adc_channel_indices=[0],
                    column_tag="",
                    feature_spec=self.spec,
                )


class CombinedInferencerTests(unittest.TestCase):
    def _write_source(self, base, module_index):
        os.makedirs(base)
        index = pd.Index([10, 11], dtype=np.int64)
        inputs = pd.DataFrame(
            {
                "module_onehot_000": np.float32(module_index == 0),
                "module_onehot_001": np.float32(module_index == 1),
                "module_onehot_002": np.float32(0.0),
                "adc_allch_000": [1.0, 2.0],
                "adc_allch_001": [3.0, 4.0],
                "adc_allch_002": [5.0, 6.0],
                "channel_indices": [[0.0, 1.0, 2.0]] * 2,
            },
            index=index,
        )
        targets = pd.DataFrame(np.ones((2, 3), dtype=np.float32), index=index)
        inputs.to_parquet(os.path.join(base, "inputs_chunk000.parquet"))
        targets.to_parquet(os.path.join(base, "targets_chunk000.parquet"))
        pd.DataFrame(
            {"event_id_global": index, "split": ["train", "train"]}
        ).to_parquet(os.path.join(base, "event_split_train_test.parquet"), index=False)

    def test_combines_modules_and_masks_each_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [os.path.join(tmp, "a"), os.path.join(tmp, "b")]
            for module_index, path in enumerate(paths):
                self._write_source(path, module_index)
            cfgs = [
                SimpleNamespace(
                    modulename=name,
                    nch=3,
                    dnn_training_input_folder=path,
                )
                for name, path in zip(["module_a", "module_b"], paths)
            ]
            for shuffle_mode in ["chunk_events", "buffered_chunk_events", "global_samples"]:
                combined = inferencers.CombinedAnalysisDNNInferencer(
                    cfgs=cfgs,
                    split="train",
                    per_channel_cols=["channel_indices"],
                )
                batches = list(
                    combined.sample_iter(
                        batch_samples=8,
                        include_targets=True,
                        shuffle_mode=shuffle_mode,
                        shuffle_buffer_chunks=2,
                    )
                )
                self.assertEqual(sum(len(x) for x, _ in batches), 12)
                allch_positions = inferencers.adc_allch_col_indices(combined.per_event_cols, 3)
                for x, _ in batches:
                    self.assertTrue(np.all(np.count_nonzero(x[:, allch_positions] == 0.0, axis=1) == 1))
                if shuffle_mode == "buffered_chunk_events":
                    self.assertEqual(len(batches), 1)
                    x = batches[0][0]
                    self.assertTrue(np.any(x[:, 0] == 1.0))
                    self.assertTrue(np.any(x[:, 1] == 1.0))


if __name__ == "__main__":
    unittest.main()
