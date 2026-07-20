#! /usr/bin/env python3

import argparse
import json
import os
import importlib.util
from array import array

import ROOT


def _load_wafer_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    wafer_path = os.path.join(repo_root, "data", "wafer.py")
    spec = importlib.util.spec_from_file_location("localcalibration_wafer", wafer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load wafer module from {wafer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_wafer_centers(nchans: int, moduletype: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    fname = os.path.join(repo_root, "data", f"geometry_{moduletype}_wafer.root")
    file = ROOT.TFile.Open(fname, "READ")
    if not file:
        raise RuntimeError(f"Could not open wafer geometry file {fname}")

    xcenters = []
    ycenters = []
    iobj = 0
    for key in file.GetListOfKeys():
        obj = key.ReadObj()
        if not obj.InheritsFrom("TGraph"):
            continue
        is_cm = (iobj % 39 == 37) or (iobj % 39 == 38)
        if is_cm:
            iobj += 1
            continue
        xcenters.append(sum(obj.GetX()) / obj.GetN())
        ycenters.append(sum(obj.GetY()) / obj.GetN())
        iobj += 1

    file.Close()
    if len(xcenters) != nchans:
        raise RuntimeError(f"Expected {nchans} wafer cells, got {len(xcenters)} from {fname}")
    return xcenters, ycenters


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a wafer histogram to PDF from a JSON payload.")
    parser.add_argument("--payload", required=True, help="Path to the JSON payload describing the wafer plot.")
    args = parser.parse_args()

    with open(args.payload, "r") as infile:
        payload = json.load(infile)

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(False)
    ROOT.gStyle.SetOptTitle(False)

    palette = payload["palette"]
    ROOT.TColor.CreateGradientColorTable(
        len(palette["stops"]),
        array("d", palette["stops"]),
        array("d", palette["red"]),
        array("d", palette["green"]),
        array("d", palette["blue"]),
        255,
    )
    ROOT.gStyle.SetNumberContours(255)

    wafer = _load_wafer_module()
    hist = wafer.fill_wafer_hist(payload["values"], moduletype=payload["module_type"])
    hist.SetDirectory(0)
    hist.SetContour(255)
    hist.SetLineColor(ROOT.kBlack)
    hist.SetLineWidth(1)
    hist.GetXaxis().SetTitleOffset(1.1)
    hist.GetYaxis().SetTitleOffset(1.6)
    hist.GetZaxis().SetTitleOffset(1.45)
    hist.GetXaxis().SetTitleSize(0.04)
    hist.GetYaxis().SetTitleSize(0.04)
    hist.GetZaxis().SetTitleSize(0.04)
    hist.GetXaxis().SetLabelSize(0.032)
    hist.GetYaxis().SetLabelSize(0.032)
    hist.GetZaxis().SetLabelSize(0.032)
    hist.GetZaxis().SetTitle(payload["ztitle"])
    if payload["zrange"] is not None:
        hist.SetMinimum(float(payload["zrange"][0]))
        hist.SetMaximum(float(payload["zrange"][1]))

    canvas = ROOT.TCanvas(f"c_{abs(hash(payload['output_filename']))}", "", 900, 850)
    canvas.SetRightMargin(0.18)
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.10)
    canvas.SetTopMargin(0.06)
    hist.Draw("COLZ L")

    if payload["title"]:
        title_latex = ROOT.TLatex()
        title_latex.SetNDC(True)
        title_latex.SetTextAlign(13)
        title_latex.SetTextSize(0.028)
        title_latex.DrawLatex(0.10, 0.965, payload["title"])

    if payload["labels"] is not None:
        xcenters, ycenters = _get_wafer_centers(len(payload["values"]), payload["module_type"])
        latex = ROOT.TLatex()
        latex.SetTextAlign(22)
        latex.SetTextSize(0.010)
        latex.SetTextColor(ROOT.kBlack)
        for idx, label in enumerate(payload["labels"]):
            latex.DrawLatex(float(xcenters[idx]), float(ycenters[idx]), str(label))

    canvas.SaveAs(payload["output_filename"])
    canvas.Close()


if __name__ == "__main__":
    main()
