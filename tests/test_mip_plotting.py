import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import functions_plot
import mip_landau


def sample_fit_result():
    values = {
        "A_ped": 100.0,
        "mu_p": 0.1,
        "sigma_p": 1.2,
        "A_mip": 20.0,
        "mu_L": 9.5,
        "c_L": 1.7,
        "A_2mip": 2.0,
    }
    return {
        "values": values,
        "errors": {name: 0.1 for name in values},
        "chi2": 10.0,
        "ndof": 10,
        "chi2_ndof": 1.0,
        "one_mip_peak": 10.0,
        "covariance": [],
        "fit_quality": {"valid": True, "warnings": []},
    }


class MIPPlottingTests(unittest.TestCase):
    def test_summary_displays_landau_width(self):
        summary = mip_landau._fit_summary_text(sample_fit_result())
        self.assertIn("Landau width c_L = 1.7 +/- 0.1", summary)

    def test_normal_plotting_fits_the_streamed_pooled_histogram(self):
        frame = pd.DataFrame(
            {
                "adc_ch000_pedsub": [0.0, 9.0, 10.0],
                "adc_ch001_pedsub": [1.0, 11.0, 12.0],
                "unrelated": [100.0, 100.0, 100.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            functions_plot.mip_landau,
            "fit_mip_spectrum",
            return_value=sample_fit_result(),
        ) as fit_mock, patch.object(
            functions_plot.mip_landau,
            "plot_fit",
        ) as plot_mock:
            functions_plot.plot_1d_multicol(
                varname_template="adc_ch*_pedsub",
                value_iterator=lambda: iter([frame]),
                out_root=tmp,
                nbins_x=50,
                x_range=(-10.0, 40.0),
                do_mip_fit=True,
                mip_fit_range=(-5.0, 35.0),
                make_logy=True,
            )

            fit_kwargs = fit_mock.call_args.kwargs
            self.assertEqual(int(fit_kwargs["counts"].sum()), 6)
            self.assertEqual(fit_kwargs["fit_range"], (-5.0, 35.0))
            self.assertEqual(plot_mock.call_count, 2)

            json_path = os.path.join(tmp, "adc_chall_pedsub_mipfit_1d.json")
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(
                payload["columns"],
                ["adc_ch000_pedsub", "adc_ch001_pedsub"],
            )
            self.assertEqual(payload["entries_in_histogram"], 6)


if __name__ == "__main__":
    unittest.main()
