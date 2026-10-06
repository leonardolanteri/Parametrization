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
# dashed grid lines at the axis ticks on every pad -- kept thin and light
ROOT.gStyle.SetPadGridX(True)
ROOT.gStyle.SetPadGridY(True)
ROOT.gStyle.SetGridStyle(2)        # 2 = dashed (3 = dotted)
ROOT.gStyle.SetGridColor(ROOT.kGray)
ROOT.gStyle.SetGridWidth(1)
# thinner lines in the PDF/PS output (1 is the minimum integer line width, so
# scale all exported line widths down for fine grid/curves)
ROOT.gStyle.SetLineScalePS(1.3)
_KEEP = []


def _grid(pad=None):
    (pad or ROOT.gPad).SetGridx(True)
    (pad or ROOT.gPad).SetGridy(True)
_CN = [0]


def _cname(base):
    """Unique canvas name (avoids ROOT 'same name' double-free at teardown)."""
    _CN[0] += 1
    return f"{base}_{_CN[0]}"


def _phi_label(phi):
    return "new (#phi=0)" if phi == 0 else "#phi = %.1e n_{eq}/cm^{2}" % phi


def _panel_layout(n):
    """(ncol, nrow) for n per-fluence panels: up to 3 columns, wraps to more rows."""
    ncol = min(max(n, 1), 3)
    nrow = int(np.ceil(n / ncol)) if n else 1
    return ncol, nrow


def _cms(canvas=None):
    """Draw 'CMS ETL Preliminary' top-left of the current single-pad figure."""
    t = ROOT.TLatex(); t.SetNDC(True); t.SetTextFont(42); t.SetTextAlign(11)
    t.SetTextSize(0.038)
    t.DrawLatex(ROOT.gPad.GetLeftMargin(), 1.0 - ROOT.gPad.GetTopMargin() + 0.012,
                "#bf{CMS} #it{ETL Preliminary}")
    _KEEP.append(t)


def _pad_header(label, cms=False):
    """In the current pad's top margin: condition `label` right-aligned, and (on the
    first panel) 'CMS ETL Preliminary' left-aligned. Used instead of a ROOT pad
    title so nothing overlaps the plot or the CMS tag."""
    y = 1.0 - ROOT.gPad.GetTopMargin() + 0.012
    r = ROOT.TLatex(); r.SetNDC(True); r.SetTextFont(42); r.SetTextAlign(31)
    r.SetTextSize(0.050)
    r.DrawLatex(1.0 - ROOT.gPad.GetRightMargin(), y, label)
    _KEEP.append(r)
    if cms:
        t = ROOT.TLatex(); t.SetNDC(True); t.SetTextFont(42); t.SetTextAlign(11)
        t.SetTextSize(0.052)
        t.DrawLatex(ROOT.gPad.GetLeftMargin(), y, "#bf{CMS} #it{ETL Preliminary}")
        _KEEP.append(t)


def plot_QV(sensors, result, show_tb=True, save=None):
    prod = result['production']
    flus = list(result['fluences'])
    n = len(flus)
    ncol, nrow = _panel_layout(n)
    c = ROOT.TCanvas(_cname("cQV_" + model._short(prod)), "Q(V) - " + prod,
                     460 * ncol, 400 * nrow)
    c.Divide(ncol, nrow)
    for i, phi in enumerate(flus):
        c.cd(i + 1)
        ROOT.gPad.SetLeftMargin(0.15)
        ROOT.gPad.SetBottomMargin(0.13)
        fit = result['perpoint'][phi]
        gr = fit['graph']
        gr.GetListOfFunctions().Clear()      # drop the red auto-drawn fit curve
        gr.SetTitle(";V [V];Q [fC]")
        gr.SetMarkerStyle(20); gr.SetMarkerSize(0.6)
        gr.SetMarkerColor(ROOT.kBlack); gr.SetLineColor(ROOT.kBlack)
        gr.Draw("ap")
        f = fit['tf1']; f.SetLineColor(ROOT.kAzure + 1); f.SetLineWidth(1)
        f.Draw("same")
        leg = ROOT.TLegend(0.17, 0.64, 0.55, 0.80)
        leg.SetBorderSize(0); leg.SetFillStyle(0)
        leg.AddEntry(gr, "%s (pooled)" % result['setup'], "pe")
        leg.AddEntry(f, "fit #chi^{2}/ndf=%.1f/%d" % (fit['chi2'], fit['ndf']), "l")
        if show_tb:
            Vtb, Qtb = data_io.pooled(sensors, prod, phi, setup='TB')
            if len(Vtb):
                gtb = ROOT.TGraph(len(Vtb), array('d', Vtb), array('d', Qtb))
                gtb.SetMarkerStyle(24); gtb.SetMarkerSize(0.6)
                gtb.SetMarkerColor(ROOT.kRed + 1); gtb.Draw("p same")
                leg.AddEntry(gtb, "test beam (check)", "p")
                _KEEP.append(gtb)
        leg.Draw(); _KEEP.append(leg)
        _pad_header(f"{prod}  {_phi_label(phi)}", cms=(i == 0))
    c.Update(); _KEEP.append(c)
    if save:
        c.SaveAs(save)
    return c


def plot_params(result, save=None):
    prod = result['production']
    phi = np.asarray(result['fluences'])
    phi_err = array('d', [config.RELERR_PHI * p for p in phi])
    c = ROOT.TCanvas(_cname("cPar_" + model._short(prod)), "params - " + prod, 900, 720)
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
    c = ROOT.TCanvas(_cname("cPred_" + model._short(prod)), "prediction - " + prod,
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


def plot_param_laws(surface, save=None):
    """
    The four parameters of the global surface
    Q(V,phi)=p0(phi)+p1(phi)*V+exp(p2(phi)+p3(phi)*V): p0,p1,p2,p3 vs fluence, each
    with its law (exponential, or a flat line when p0 is a constant) and the
    uncertainty band propagated from the global fit. Markers lie on the law by
    construction -- it is one simultaneous fit.
    """
    prod = surface.production
    phi = surface.fluences
    phi_err = array('d', [config.RELERR_PHI * p for p in phi])
    # grid from 0 (physical fluences only): the exp laws blow up at phi < 0
    xs = np.linspace(0.0, max(phi.max() * 1.15, 1e15), 250)
    c = ROOT.TCanvas(_cname("cPL_" + model._short(prod)),
                     "parameter laws - " + prod, 900, 720)
    c.Divide(2, 2)
    keep = []
    specs = [(surface.p0_nodes, surface.p0_err, surface.law0, surface.p0, 'p0'),
             (surface.p1_nodes, surface.p1_err, surface.law1, surface.p1, 'p1'),
             (surface.p2_nodes, surface.p2_err, surface.law2, surface.p2, 'p2'),
             (surface.p3_nodes, surface.p3_err, surface.law3, surface.p3, 'p3')]
    for j, (nodes, errs, law, fn, nm) in enumerate(specs):
        c.cd(j + 1); ROOT.gPad.SetLeftMargin(0.19); ROOT.gPad.SetBottomMargin(0.14)
        nodes = np.asarray(nodes, dtype=float); errs = np.asarray(errs, dtype=float)
        kind = 'const' if law['kind'] == 'lin' else law['kind']
        g = ROOT.TGraphErrors(len(phi), array('d', phi), array('d', nodes),
                              phi_err, array('d', errs))
        unit = ' [fC]' if nm == 'p0' else ''
        g.SetTitle(f";#phi [n_{{eq}}/cm^{{2}}];{nm}{unit}")
        g.SetMarkerStyle(21); g.SetMarkerSize(0.9); g.SetLineWidth(1)
        ys = np.asarray(fn(xs), dtype=float)
        # y-range adapts to the points (+/- band) and the law over the grid
        lo = min(float((nodes - errs).min()), float(ys.min()))
        hi = max(float((nodes + errs).max()), float(ys.max()))
        pad = 0.12 * (hi - lo) + (0.1 if hi == lo else 0.0)
        g.Draw("ap")
        g.GetXaxis().SetLimits(xs[0], xs[-1])
        g.GetYaxis().SetRangeUser(lo - pad, hi + pad)
        gl = ROOT.TGraph(len(xs), array('d', xs), array('d', ys))
        gl.SetLineColor(ROOT.kAzure + 1); gl.SetLineWidth(1); gl.Draw("l same")
        _pad_header(f"{prod}  {nm}(#phi) [{kind}]", cms=(j == 0))
        keep += [g, gl]
    c.Update(); _KEEP.extend(keep + [c])
    if save:
        c.SaveAs(save)
    return c


def plot_surface_vs_data(sensors, surface, setup=None, save=None):
    """
    Overlay the global surface prediction Q(V,phi) on the pooled data at each
    measured fluence (one pad per fluence). The prediction is a dashed green line.
    Saved as param_pred_<production> by run.py.
    """
    setup = setup or config.FIT_SETUP
    prod = surface.production
    flus = list(surface.fluences)
    n = len(flus)
    ncol, nrow = _panel_layout(n)
    c = ROOT.TCanvas(_cname("cParPred_" + model._short(prod)),
                     "param. prediction vs data - " + prod, 460 * ncol, 400 * nrow)
    c.Divide(ncol, nrow)
    for i, phi in enumerate(flus):
        c.cd(i + 1)
        ROOT.gPad.SetLeftMargin(0.15)
        ROOT.gPad.SetBottomMargin(0.13)
        V, Q = data_io.pooled(sensors, prod, phi, setup)
        Verr = array('d', np.full(len(V), config.ERR_V))
        Qerr = array('d', np.full(len(Q), config.ERR_Q))
        gr = ROOT.TGraphErrors(len(V), array('d', V), array('d', Q), Verr, Qerr)
        gr.SetTitle(";V [V];Q [fC]")
        gr.SetMarkerStyle(20); gr.SetMarkerSize(0.6); gr.Draw("ap")
        vmin, vmax_ = float(V.min()), float(V.max())
        xs = np.linspace(vmin, vmax_, 200)
        ys = surface.Q(xs, phi)               # full global surface prediction
        gl = ROOT.TGraph(len(xs), array('d', xs), array('d', ys))
        gl.SetLineColor(ROOT.kGreen + 2); gl.SetLineWidth(1); gl.SetLineStyle(2)
        gl.Draw("l same")
        leg = ROOT.TLegend(0.17, 0.70, 0.55, 0.82)
        leg.SetBorderSize(0); leg.SetFillStyle(0)
        leg.AddEntry(gr, "%s (pooled)" % setup, "pe")
        leg.AddEntry(gl, "param. prediction", "l")
        leg.Draw()
        _pad_header(f"{prod}  {_phi_label(phi)}", cms=(i == 0))
        _KEEP.extend([gr, gl, leg])
    c.Update(); _KEEP.append(c)
    if save:
        c.SaveAs(save)
    return c


def _vmax_for(name, vmax):
    if isinstance(vmax, (int, float)):
        return float(vmax)
    if vmax and name in vmax:
        return float(vmax[name])
    d = config.V_SEB
    if name in d:
        return float(d[name])
    u = name.upper()
    if 'FBK' in u:
        return float(d.get('FBK', config.V_SEB_DEFAULT))
    if 'HPK' in u:
        return float(d.get('HPK', config.V_SEB_DEFAULT))
    return float(config.V_SEB_DEFAULT)


def _color_for(name):
    u = name.upper()
    if 'FBK' in u:
        return ROOT.kRed + 1
    if 'HPK' in u:
        return ROOT.kAzure + 2
    return ROOT.kBlack


def _sim_colors():
    """(charge, bias) colors for the simulation overlay -- green & amber, chosen to
    stand out from the azure/red data-driven curves."""
    return ROOT.kGreen + 2, ROOT.TColor.GetColor("#e0a000")   # green, amber/yellow


def _load_sim(name):
    """Read simulation/<production>.csv -> (phi[n_eq], charge[fC], bias[V]) or None."""
    import os
    path = os.path.join(config.SIM_DIR, "%s.csv" % model._short(name))
    if not os.path.exists(path):
        path = os.path.join(config.SIM_DIR, "%s.csv" % name)
    if not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path, comment="#")
    return (df["phi14"].to_numpy() * 1e14,
            df["charge"].to_numpy(), df["bias"].to_numpy())


def plot_charge_bias_vs_fluence(surfaces, target=None, vmax=None,
                                phi_lo=1e14, phi_hi=1e16,
                                charge_max=22.0, bias_max=660.0,
                                overlay=None, save=None):
    """
    Charge obtained and bias needed vs fluence, with the SEB cap.

    `surfaces`: dict {name: SimulSurface} (or a single surface). The global
    surface is one smooth trend over the whole range; its free p0 term makes the
    charge decay on its own to the bulk value (~0.3-0.5 fC), not to 0.

    For each production, over a fluence grid:
      * dashed (right axis, Bias): V needed to reach `target` charge, capped at
        V_SEB; flat at V_SEB once the target is out of reach.
      * solid (left axis, Charge): charge actually obtainable =
        min(target, Q(V_SEB, phi)).
    """
    target = config.PLATEAU_CHARGE if target is None else target
    if hasattr(surfaces, 'production'):                          # single GlobalSurface
        surfaces = {surfaces.production: surfaces}
    elif isinstance(surfaces, dict) and 'coeffs' in surfaces:    # single fit result
        surfaces = {surfaces['production']: surfaces}
    surfaces = {k: model.as_surface(v) for k, v in surfaces.items()}

    c = ROOT.TCanvas(_cname("cQB_%d" % int(target)), "charge & bias vs fluence", 820, 640)
    c.SetLogx(True)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.13)
    c.SetTopMargin(0.08)
    _grid(c)

    frame = c.DrawFrame(phi_lo / 1e14, 0.0, phi_hi / 1e14, charge_max)
    frame.GetXaxis().SetTitle("Fluence  [10^{14} n_{eq} cm^{-2}]")
    frame.GetYaxis().SetTitle("Charge [fC]")
    frame.GetYaxis().SetTitleOffset(1.1)

    scale = charge_max / bias_max            # map bias [V] -> charge-axis units
    leg = ROOT.TLegend(0.15, 0.13, 0.52, 0.34)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    phi = np.logspace(np.log10(phi_lo), np.log10(phi_hi), 400)
    x = phi / 1e14
    for name, surf in surfaces.items():
        Vm = _vmax_for(name, vmax)
        col = _color_for(name)

        Qvm = surf.Q(Vm, phi)                        # charge available at V_SEB
        Qobt = np.minimum(target, Qvm)               # obtainable = min(target, avail.)
        Vneed = np.empty_like(phi)
        for i, ph in enumerate(phi):
            if Qvm[i] >= target:
                v = surf.V_for_Q(target, ph, v_hi=Vm)
                Vneed[i] = Vm if v != v else v
            else:
                Vneed[i] = Vm

        g_q = ROOT.TGraph(len(x), array('d', x), array('d', Qobt))
        g_q.SetLineColor(col); g_q.SetLineWidth(2); g_q.Draw("l same")
        g_v = ROOT.TGraph(len(x), array('d', x), array('d', Vneed * scale))
        g_v.SetLineColor(col); g_v.SetLineWidth(1); g_v.SetLineStyle(2)
        g_v.Draw("l same")

        ln = ROOT.TLine(x[0], Vm * scale, x[-1], Vm * scale)
        ln.SetLineColor(col); ln.SetLineStyle(3); ln.SetLineWidth(1); ln.Draw("same")
        txt = ROOT.TLatex(x[0] * 1.2, Vm * scale + 0.3,
                          "V_{max}^{%s} = %.0f V" % (name, Vm))
        txt.SetTextColor(col); txt.SetTextSize(0.030); txt.Draw("same")

        leg.AddEntry(g_q, "%s  Charge (solid)" % name, "l")
        leg.AddEntry(g_v, "%s  Bias (dashed)" % name, "l")
        _KEEP.extend([g_q, g_v, ln, txt])

        # --- optional simulation overlay (deep blue), read from simulation/*.csv
        sim = _load_sim(name) if overlay == "auto" else (
            overlay.get(name) if isinstance(overlay, dict) else None)
        if sim is not None:
            sphi, schg, sbias = sim
            sx = np.asarray(sphi, dtype=float) / 1e14
            scol_q, scol_v = _sim_colors()
            sg_q = ROOT.TGraph(len(sx), array('d', sx), array('d', np.asarray(schg, float)))
            sg_q.SetLineColor(scol_q); sg_q.SetLineWidth(2)
            sg_q.Draw("l same")
            sg_v = ROOT.TGraph(len(sx), array('d', sx),
                               array('d', np.asarray(sbias, float) * scale))
            sg_v.SetLineColor(scol_v); sg_v.SetLineWidth(2); sg_v.SetLineStyle(2)
            sg_v.Draw("l same")
            leg.AddEntry(sg_q, "%s  sim. Charge (solid)" % name, "l")
            leg.AddEntry(sg_v, "%s  sim. Bias (dashed)" % name, "l")
            _KEEP.extend([sg_q, sg_v])

    raxis = ROOT.TGaxis(phi_hi / 1e14, 0.0, phi_hi / 1e14, charge_max,
                        0.0, bias_max, 510, "+L")
    raxis.SetTitle("Bias [V]")
    raxis.SetTitleFont(42); raxis.SetLabelFont(42)
    raxis.SetTitleOffset(1.25)
    raxis.Draw()

    leg.Draw()
    _cms()
    c.Update()
    _KEEP.extend([frame, raxis, leg, c])
    if save:
        c.SaveAs(save)
    return c


def plot_bias_for_target(surfaces, Q_target=None, phi_hi=None, save=None):
    Q_target = config.Q_TARGET if Q_target is None else Q_target
    if hasattr(surfaces, 'production'):
        surfaces = {surfaces.production: surfaces}
    elif isinstance(surfaces, dict) and 'coeffs' in surfaces:
        surfaces = {surfaces['production']: surfaces}
    surfaces = {k: model.as_surface(v) for k, v in surfaces.items()}
    c = ROOT.TCanvas(_cname(f"cBias_{int(Q_target)}"), f"Bias Q={Q_target}", 800, 600)
    ROOT.gPad.SetLeftMargin(0.13)
    _grid()
    palette = [ROOT.kAzure + 1, ROOT.kOrange + 7, ROOT.kGreen + 2, ROOT.kMagenta + 1]
    leg = ROOT.TLegend(0.15, 0.72, 0.5, 0.88); leg.SetBorderSize(0); leg.SetFillStyle(0)
    first = True
    for k, (name, surf) in enumerate(surfaces.items()):
        hi = phi_hi if phi_hi else float(surf.fluences.max())
        phi = np.linspace(0, hi, 120)
        V = np.array([surf.V_for_Q(Q_target, ph) for ph in phi])
        g = ROOT.TGraph(len(phi), array('d', phi), array('d', np.nan_to_num(V)))
        g.SetLineColor(palette[k % len(palette)]); g.SetLineWidth(2)
        g.SetTitle(f";#phi [n_{{eq}}/cm^{{2}}];V_{{{Q_target:.0f}fC}} [V]")
        if first:
            g.GetYaxis().SetRangeUser(0, 900); g.Draw("al"); first = False
        else:
            g.Draw("l same")
        leg.AddEntry(g, name, "l"); _KEEP.append(g)
    leg.Draw(); _pad_header("Q = %.0f fC target" % Q_target, cms=True)
    c.Update(); _KEEP.extend([leg, c])
    if save:
        c.SaveAs(save)
    return c
