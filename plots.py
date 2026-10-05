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
# dashed grid lines at the axis ticks on every pad
ROOT.gStyle.SetPadGridX(True)
ROOT.gStyle.SetPadGridY(True)
ROOT.gStyle.SetGridStyle(2)        # 2 = dashed (3 = dotted)
ROOT.gStyle.SetGridColor(ROOT.kGray + 1)
ROOT.gStyle.SetGridWidth(1)
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


def plot_QV(sensors, result, show_tb=True, save=None):
    prod = result['production']
    flus = list(result['fluences'])
    n = len(flus)
    c = ROOT.TCanvas(_cname("cQV_" + model._short(prod)), "Q(V) - " + prod,
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
    All four parameters of Q(V)=p0+p1*V+exp(p2+p3*V) vs fluence:
      p0 = Q_FLOOR (fixed), p1 = 0 (fixed; unconstrained by data),
      p2(phi), p3(phi) with their analytic laws + reduced per-fluence points.
    """
    prod = surface.production
    phi = surface.fluences
    phi_err = array('d', [config.RELERR_PHI * p for p in phi])
    xs = np.linspace(-0.05e15, max(phi.max() * 1.15, 1e15), 250)
    c = ROOT.TCanvas(_cname("cPL_" + model._short(prod)),
                     "parameter laws - " + prod, 900, 720)
    c.Divide(2, 2)
    keep = []

    # p0 and p1: fixed by physics (shown as flat lines)
    fixed = [('p0', surface.qfloor, 'fixed = Q_{FLOOR}'),
             ('p1', 0.0, 'fixed = 0 (unconstrained)')]
    for i, (nm, val, note) in enumerate(fixed):
        c.cd(i + 1); ROOT.gPad.SetLeftMargin(0.17); ROOT.gPad.SetBottomMargin(0.14)
        gl = ROOT.TGraph(len(xs), array('d', xs), array('d', np.full(len(xs), val)))
        gl.SetTitle(f"{nm}(#phi)  [{note}] - {prod};#phi [n_{{eq}}/cm^{{2}}];{nm}")
        gl.SetLineColor(ROOT.kAzure + 1); gl.SetLineWidth(2)
        gl.GetYaxis().SetRangeUser(val - 1.0, val + 1.0)
        gl.Draw("al")
        keep.append(gl)

    # p2 and p3: fitted analytic laws + per-fluence points
    specs = [(surface.p2_nodes, surface.p2_err, surface.law2, surface.p2, 'p2'),
             (surface.p3_nodes, surface.p3_err, surface.law3, surface.p3, 'p3')]
    for j, (nodes, errs, law, fn, nm) in enumerate(specs):
        c.cd(j + 3); ROOT.gPad.SetLeftMargin(0.17); ROOT.gPad.SetBottomMargin(0.14)
        g = ROOT.TGraphErrors(len(phi), array('d', phi), array('d', nodes),
                              phi_err, array('d', errs))
        g.SetTitle(f"{nm}(#phi)  [{law['kind']}] - {prod};"
                   f"#phi [n_{{eq}}/cm^{{2}}];{nm}")
        g.SetMarkerStyle(21); g.SetMarkerSize(1.1); g.Draw("ap")
        gl = ROOT.TGraph(len(xs), array('d', xs), array('d', fn(xs)))
        gl.SetLineColor(ROOT.kAzure + 1); gl.SetLineWidth(2); gl.Draw("l same")
        keep += [g, gl]

    c.Update(); _KEEP.extend(keep + [c])
    if save:
        c.SaveAs(save)
    return c


def plot_surface_vs_data(sensors, surface, setup=None, save=None):
    """
    Overlay the single analytic global surface Q(V,phi) on the pooled data at
    each measured fluence (one pad per fluence). Shows how well one analytic law
    describes every fluence.
    """
    setup = setup or config.FIT_SETUP
    prod = surface.production
    flus = list(surface.fluences)
    n = len(flus)
    c = ROOT.TCanvas(_cname("cSurfData_" + model._short(prod)),
                     "surface vs data - " + prod, 360 * max(n, 1), 430)
    c.Divide(n, 1)
    for i, phi in enumerate(flus):
        c.cd(i + 1)
        ROOT.gPad.SetLeftMargin(0.15)
        ROOT.gPad.SetBottomMargin(0.13)
        V, Q = data_io.pooled(sensors, prod, phi, setup)
        Verr = array('d', np.full(len(V), config.ERR_V))
        Qerr = array('d', np.full(len(Q), config.ERR_Q))
        gr = ROOT.TGraphErrors(len(V), array('d', V), array('d', Q), Verr, Qerr)
        gr.SetTitle(f"{prod} - {_phi_label(phi)};V [V];Q [fC]")
        gr.SetMarkerStyle(20); gr.SetMarkerSize(0.8); gr.Draw("ap")
        vmin, vmax_ = float(V.min()), float(V.max())
        xs = np.linspace(vmin, vmax_, 200)
        ys = surface.Q(xs, phi)
        gl = ROOT.TGraph(len(xs), array('d', xs), array('d', ys))
        gl.SetLineColor(ROOT.kAzure + 1); gl.SetLineWidth(2); gl.Draw("l same")
        _KEEP.extend([gr, gl])
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


def plot_charge_bias_vs_fluence(surfaces, target=None, vmax=None,
                                phi_lo=1e14, phi_hi=1e16,
                                charge_max=22.0, bias_max=660.0, save=None):
    """
    Charge obtained and bias needed vs fluence, with the SEB cap.

    `surfaces`: dict {name: GlobalSurface} (or a single GlobalSurface). The
    global surface is one smooth trend over the whole range and asymptotes to the
    no-gain floor (config.Q_FLOOR), so the charge decays to ~0.55 fC, not to 0.

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
        Qobt = np.minimum(target, Qvm)               # surf.Q already >= Q_floor
        Vneed = np.empty_like(phi)
        for i, ph in enumerate(phi):
            if Qvm[i] >= target:
                v = surf.V_for_Q(target, ph, v_hi=Vm)
                Vneed[i] = Vm if v != v else v
            else:
                Vneed[i] = Vm

        g_q = ROOT.TGraph(len(x), array('d', x), array('d', Qobt))
        g_q.SetLineColor(col); g_q.SetLineWidth(3); g_q.Draw("l same")
        g_v = ROOT.TGraph(len(x), array('d', x), array('d', Vneed * scale))
        g_v.SetLineColor(col); g_v.SetLineWidth(2); g_v.SetLineStyle(2)
        g_v.Draw("l same")

        ln = ROOT.TLine(x[0], Vm * scale, x[-1], Vm * scale)
        ln.SetLineColor(col); ln.SetLineStyle(3); ln.SetLineWidth(1); ln.Draw("same")
        txt = ROOT.TLatex(x[0] * 1.2, Vm * scale + 0.3,
                          "V_{max}^{%s} = %.0f V" % (name, Vm))
        txt.SetTextColor(col); txt.SetTextSize(0.030); txt.Draw("same")

        leg.AddEntry(g_q, "%s  Charge (solid)" % name, "l")
        leg.AddEntry(g_v, "%s  Bias (dashed)" % name, "l")
        _KEEP.extend([g_q, g_v, ln, txt])

    raxis = ROOT.TGaxis(phi_hi / 1e14, 0.0, phi_hi / 1e14, charge_max,
                        0.0, bias_max, 510, "+L")
    raxis.SetTitle("Bias [V]")
    raxis.SetTitleFont(42); raxis.SetLabelFont(42)
    raxis.SetTitleOffset(1.25)
    raxis.Draw()

    leg.Draw()
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
