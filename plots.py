"""
plots.py
========
ROOT plots for the Q(V, phi) parametrization. Each function takes the loaded
`sensors` and a production `result` (from model.fit_production). Objects are
kept alive in `_KEEP` so PyROOT does not garbage-collect them.
"""

from array import array
import numpy as np
import ROOT

import config
import data_io
import model

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)
_KEEP = []


def _phi_label(phi):
    return "new (#phi=0)" if phi == 0 else "#phi = %.1e n_{eq}/cm^{2}" % phi


def plot_QV(sensors, result, show_tb=True, save=None):
    prod = result['production']
    flus = list(result['fluences'])
    n = len(flus)
    c = ROOT.TCanvas("cQV_" + model._short(prod), "Q(V) - " + prod,
                     360 * max(n, 1), 430)
    c.Divide(n, 1)
    for i, phi in enumerate(flus):
        c.cd(i + 1)
        ROOT.gPad.SetLeftMargin(0.15)
        ROOT.gPad.SetBottomMargin(0.13)
        fit = result['perpoint'][phi]
        gr = fit['graph']
        gr.SetTitle(f"{prod} - {_phi_label(phi)};V [V];Q [fC]")
        gr.SetMarkerStyle(20); gr.SetMarkerSize(0.8)
        gr.SetMarkerColor(ROOT.kBlack); gr.SetLineColor(ROOT.kBlack)
        gr.Draw("ap")
        f = fit['tf1']; f.SetLineColor(ROOT.kAzure + 1); f.SetLineWidth(2)
        f.Draw("same")
        leg = ROOT.TLegend(0.17, 0.72, 0.55, 0.88)
        leg.SetBorderSize(0); leg.SetFillStyle(0)
        leg.AddEntry(gr, "%s (pooled)" % result['setup'], "pe")
        leg.AddEntry(f, "fit #chi^{2}/ndf=%.1f/%d" % (fit['chi2'], fit['ndf']), "l")
        if show_tb:
            Vtb, Qtb = data_io.pooled(sensors, prod, phi, setup='TB')
            if len(Vtb):
                gtb = ROOT.TGraph(len(Vtb), array('d', Vtb), array('d', Qtb))
                gtb.SetMarkerStyle(24); gtb.SetMarkerSize(0.8)
                gtb.SetMarkerColor(ROOT.kRed + 1); gtb.Draw("p same")
                leg.AddEntry(gtb, "test beam (check)", "p")
                _KEEP.append(gtb)
        leg.Draw(); _KEEP.append(leg)
    c.Update(); _KEEP.append(c)
    if save:
        c.SaveAs(save)
    return c


def plot_params(result, save=None):
    prod = result['production']
    phi = np.asarray(result['fluences'])
    phi_err = array('d', [config.RELERR_PHI * p for p in phi])
    c = ROOT.TCanvas("cPar_" + model._short(prod), "params - " + prod, 900, 720)
    c.Divide(2, 2)
    keep = []
    for i in range(4):
        c.cd(i + 1); ROOT.gPad.SetLeftMargin(0.17)
        p = np.asarray(result['P'][i]); perr = np.asarray(result['Perr'][i])
        g = ROOT.TGraphErrors(len(phi), array('d', phi), array('d', p),
                              phi_err, array('d', perr))
        g.SetTitle(f"{model.PARAM_NAMES[i]} - {prod};"
                   f"#phi [n_{{eq}}/cm^{{2}}];{model.PARAM_NAMES[i]}")
        g.SetMarkerStyle(21); g.SetMarkerSize(1.0); g.Draw("ap")
        coeff = result['coeffs'][i]
        xlo, xhi = -0.05e15, max(phi.max() * 1.15, 1e15)
        xs = np.linspace(xlo, xhi, 250)
        ys = model.eval_param(coeff, xs)
        gl = ROOT.TGraph(len(xs), array('d', xs), array('d', ys))
        gl.SetLineColor(ROOT.kAzure + 1); gl.SetLineWidth(2); gl.Draw("l same")
        # grey dashed = linear trend (the compact closed form)
        t = result['trend'][i]
        gt = ROOT.TGraph(len(xs), array('d', xs),
                         array('d', t['m'] * xs + t['q']))
        gt.SetLineColor(ROOT.kGray + 2); gt.SetLineStyle(2); gt.Draw("l same")
        keep += [g, gl, gt]
    c.Update(); _KEEP.extend(keep + [c])
    if save:
        c.SaveAs(save)
    return c


def plot_prediction(result, save=None):
    prod = result['production']
    flus = list(result['fluences'])
    n = len(flus)
    coeffs = result['coeffs']
    c = ROOT.TCanvas("cPred_" + model._short(prod), "prediction - " + prod,
                     360 * max(n, 1), 430)
    c.Divide(n, 1)
    for i, phi in enumerate(flus):
        c.cd(i + 1); ROOT.gPad.SetLeftMargin(0.15)
        gr = result['perpoint'][phi]['graph']
        gr.SetTitle(f"{prod} - {_phi_label(phi)} (pred.);V [V];Q [fC]")
        gr.SetMarkerStyle(20); gr.SetMarkerSize(0.8); gr.Draw("ap")
        vmin = min(gr.GetX()[k] for k in range(gr.GetN()))
        vmax = max(gr.GetX()[k] for k in range(gr.GetN()))
        pv = [model.eval_param(coeffs[j], phi) for j in range(4)]
        fp = ROOT.TF1(f"fpred_{model._short(prod)}_{i}", model.QV_FORMULA, vmin, vmax)
        fp.SetParameters(*pv)
        fp.SetLineColor(ROOT.kGreen + 2); fp.SetLineWidth(2); fp.SetLineStyle(2)
        fp.Draw("same")
        _KEEP.append(fp)
    c.Update(); _KEEP.append(c)
    if save:
        c.SaveAs(save)
    return c


def plot_bias_for_target(results, Q_target=None, save=None):
    Q_target = config.Q_TARGET if Q_target is None else Q_target
    if isinstance(results, dict) and 'production' in results:
        results = {results['production']: results}
    c = ROOT.TCanvas(f"cBias_{int(Q_target)}", f"Bias Q={Q_target}", 800, 600)
    ROOT.gPad.SetLeftMargin(0.13)
    palette = [ROOT.kAzure + 1, ROOT.kOrange + 7, ROOT.kGreen + 2, ROOT.kMagenta + 1]
    leg = ROOT.TLegend(0.15, 0.72, 0.5, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
    first = True
    for k, (name, res) in enumerate(results.items()):
        phi = np.linspace(0, float(res['fluences'].max()), 80)
        V = np.array([model.V_for_Q(Q_target, ph, res['coeffs']) for ph in phi])
        g = ROOT.TGraph(len(phi), array('d', phi), array('d', np.nan_to_num(V)))
        g.SetLineColor(palette[k % len(palette)]); g.SetLineWidth(3)
        g.SetTitle(f"Bias to reach Q = {Q_target:.0f} fC;"
                   f"#phi [n_{{eq}}/cm^{{2}}];V_{{{Q_target:.0f}fC}} [V]")
        if first:
            g.GetYaxis().SetRangeUser(0, 900); g.Draw("al"); first = False
        else:
            g.Draw("l same")
        leg.AddEntry(g, name, "l"); _KEEP.append(g)
    leg.Draw(); c.Update(); _KEEP.extend([leg, c])
    if save:
        c.SaveAs(save)
    return c
