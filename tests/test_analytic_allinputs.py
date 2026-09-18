import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import add_correction_analytic_allinputs
import compute_predictor_analytic_allinputs
import prepare_dnn_inputs


class AnalyticAllInputsTests(unittest.TestCase):
    def _fit_synthetic_predictor(self):
        rng = np.random.default_rng(12345)
        n_events = 4000
        common = rng.normal(size=n_events)
        adc0 = rng.normal(size=n_events)
        adc1 = rng.normal(size=n_events)
        x = np.column_stack(
            [
                common,
                np.ones(n_events),
                np.zeros(n_events),
                adc0,
                adc1,
            ]
        )
        y = np.column_stack(
            [
                1.5 + 2.0 * common + 3.0 * adc1,
                -0.75 - common + 4.0 * adc0,
            ]
        )
        valid = np.isfinite(y)
        result = compute_predictor_analytic_allinputs.solve_analytic_allinputs(
            feature_names=[
                "common",
                "module_onehot_000",
                "module_onehot_001",
                "adc_allch_000",
                "adc_allch_001",
            ],
            target_columns=["adc_ch000_pedsub", "adc_ch001_pedsub"],
            n_events=n_events,
            sum_x=x.sum(axis=0),
            sum_xx=x.T @ x,
            target_counts=valid.sum(axis=0),
            sum_y=y.sum(axis=0),
            sum_xy=x.T @ y,
            sum_x_target_valid=x.T @ valid.astype(np.float64),
        )
        return x, y, result

    def test_solver_fits_each_target_and_excludes_own_adc(self):
        x, y, result = self._fit_synthetic_predictor()
        weights = result["weights"]
        predictions = x @ weights.T + result["intercepts"][None, :]

        self.assertEqual(weights[0, 3], 0.0)
        self.assertEqual(weights[1, 4], 0.0)
        np.testing.assert_allclose(predictions, y, atol=1.0e-10)
        self.assertIn("module_onehot_001", result["constant_features"])

    def test_prediction_is_insensitive_to_target_adc(self):
        x, _, result = self._fit_synthetic_predictor()
        feature_names = [
            "common",
            "module_onehot_000",
            "module_onehot_001",
            "adc_allch_000",
            "adc_allch_001",
        ]
        target_columns = ["adc_ch000_pedsub", "adc_ch001_pedsub"]
        weights = pd.DataFrame(result["weights"], index=target_columns, columns=feature_names)
        intercepts = pd.DataFrame(
            {"intercept": result["intercepts"]}, index=target_columns
        )
        inputs = pd.DataFrame(x, columns=feature_names)
        nominal = add_correction_analytic_allinputs.predict_analytic_allinputs(
            inputs, feature_names, weights, intercepts
        )

        changed = inputs.copy()
        changed["adc_allch_000"] += 10000.0
        changed_predictions = add_correction_analytic_allinputs.predict_analytic_allinputs(
            changed, feature_names, weights, intercepts
        )
        np.testing.assert_allclose(changed_predictions[:, 0], nominal[:, 0])
        self.assertGreater(
            float(np.max(np.abs(changed_predictions[:, 1] - nominal[:, 1]))),
            1000.0,
        )

    def test_application_rejects_feature_reordering(self):
        x, _, result = self._fit_synthetic_predictor()
        feature_names = [
            "common",
            "module_onehot_000",
            "module_onehot_001",
            "adc_allch_000",
            "adc_allch_001",
        ]
        target_columns = ["adc_ch000_pedsub", "adc_ch001_pedsub"]
        weights = pd.DataFrame(result["weights"], index=target_columns, columns=feature_names)
        intercepts = pd.DataFrame(
            {"intercept": result["intercepts"]}, index=target_columns
        )
        reordered = pd.DataFrame(x, columns=feature_names)[feature_names[::-1]]
        with self.assertRaisesRegex(ValueError, "feature order"):
            add_correction_analytic_allinputs.predict_analytic_allinputs(
                reordered, feature_names, weights, intercepts
            )

    def test_streaming_derivation_writes_manifest_and_artifacts(self):
        feature_spec = prepare_dnn_inputs.make_feature_spec(
            feature_version=prepare_dnn_inputs.FEATURE_VERSION_ALL_CHANNELS,
            module_vocabulary=["module_a01"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                modulename="module_a01",
                nch=2,
                dnn_training_input_base_folder=os.path.join(tmp, "inputs"),
                dnn_training_input_folder=os.path.join(tmp, "inputs"),
            )
            input_folder = prepare_dnn_inputs.configure_dnn_input_folder(
                cfg, feature_spec
            )
            os.makedirs(input_folder)
            index = pd.Index(np.arange(100, 200, dtype=np.int64))
            common = np.linspace(-2.0, 2.0, len(index))
            adc0 = np.sin(common)
            adc1 = np.cos(common)
            inputs = pd.DataFrame(
                {
                    "common": common,
                    "module_onehot_000": np.ones(len(index)),
                    "module_onehot_001": np.zeros(len(index)),
                    "adc_allch_000": adc0,
                    "adc_allch_001": adc1,
                    "channel_indices": [[-0.5, 0.5]] * len(index),
                    "erx_indices": [[0.0, 0.0]] * len(index),
                    "cell_area_fraction": [[1.0, 1.0]] * len(index),
                },
                index=index,
            )
            targets = pd.DataFrame(
                {
                    "adc_ch000_pedsub": 0.5 + 2.0 * common + 3.0 * adc1,
                    "adc_ch001_pedsub": -0.2 - common + 4.0 * adc0,
                },
                index=index,
            )
            inputs.to_parquet(os.path.join(input_folder, "inputs_chunk000.parquet"))
            targets.to_parquet(os.path.join(input_folder, "targets_chunk000.parquet"))
            pd.DataFrame(
                {"event_id_global": index, "split": ["train"] * len(index)}
            ).to_parquet(
                os.path.join(input_folder, "event_split_train_test.parquet"), index=False
            )
            manifest = dict(feature_spec)
            manifest.update(
                {
                    "module": "module_a01",
                    "nch": 2,
                    "column_tag": "",
                    "per_channel_columns": [
                        "channel_indices",
                        "erx_indices",
                        "cell_area_fraction",
                    ],
                }
            )
            with open(
                os.path.join(
                    input_folder, prepare_dnn_inputs.DNN_INPUT_MANIFEST_FILENAME
                ),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(manifest, handle)

            output_folder = os.path.join(tmp, "predictors")
            cfg_out = SimpleNamespace(analytic_predictor_folder=output_folder)
            compute_predictor_analytic_allinputs.compute_predictor_analytic_allinputs(
                cfgs=[cfg],
                cfg_out=cfg_out,
                feature_spec=feature_spec,
                train_event_fractions={"module_a01": 1.0},
            )

            saved_manifest = compute_predictor_analytic_allinputs._read_json(
                os.path.join(
                    output_folder, compute_predictor_analytic_allinputs.MANIFEST_FILENAME
                )
            )
            saved_weights = pd.read_parquet(
                os.path.join(
                    output_folder, compute_predictor_analytic_allinputs.WEIGHTS_FILENAME
                )
            )
            self.assertEqual(saved_manifest["n_train_events"], len(index))
            self.assertEqual(saved_manifest["training_modules"], ["module_a01"])
            self.assertEqual(saved_weights.loc["adc_ch000_pedsub", "adc_allch_000"], 0.0)
            self.assertEqual(saved_weights.loc["adc_ch001_pedsub", "adc_allch_001"], 0.0)


if __name__ == "__main__":
    unittest.main()
