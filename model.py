"""
model.py
========
Fitting engine for Q(V, phi).

Step 1 (per fluence):   Q(V) = p0 + p1*V + exp(p2 + p3*V)
  Parameters are seeded analytically (auto_seed_QV) so the fit converges on its
  own -- no per-point manual tuning. Only uniform physical bounds are applied
  (p1 >= 0 and p3 >= 0: charge does not decrease with bias).

Step 2 (vs fluence):    each p_k modelled vs phi
  config.PARAM_MODE = 'auto' : linear if < MIN_PTS_FOR_EXP points, otherwise the
                               better of exp / linear per parameter (by AIC)
                 'lin' | 'exp' | 'interp' force a specific law.

Combined:  Q(V,phi) = P0(phi) + P1(phi)*V + exp( P2(phi) + P3(phi)*V )

Fits are done in ROOT; scipy is an optional fallback for the Q(V) fit.
"""

from array import array
import numpy as np
import ROOT

import config
import data_io

ROOT.gErrorIgnoreLevel = ROOT.kWarning

try:
    from scipy.optimize import curve_fit as _curve_fit
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

QV_FORMULA = "[0] + [1]*x + exp([2] + [3]*x)"   # x = V
PARAM_NAMES = ['p0', 'p1', 'p2', 'p3']


# ===========================================================================
# step 1 -- Q(V)
# ===========================================================================
def charge_np(V, p0, p1, p2, p3):
    V = np.asarray(V, dtype=float)
    return p0 + p1 * V + np.exp(p2 + p3 * V)


def auto_seed_QV(V, Q):
    """Analytic initial guess [p0, p1, p2, p3] for Q(V)=p0+p1*V+exp(p2+p3*V)."""
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    order = np.argsort(V)
    V, Q = V[order], Q[order]
    n = len(V)

    k = max(2, int(np.ceil(0.4 * n)))
    try:
        p1, p0 = np.polyfit(V[:k], Q[:k], 1)
    except Exception:
        p1, p0 = 0.0, float(Q[0])

    resid = Q - (p0 + p1 * V)
    thr = max(0.05 * np.ptp(Q), 1e-3)
    mask = resid > thr
    if mask.sum() >= 2:
        try:
            p3, p2 = np.polyfit(V[mask], np.log(resid[mask]), 1)
        except Exception:
            p3, p2 = 0.01, np.log(max(0.1 * np.mean(Q), 1e-2))
    else:
        p3, p2 = 0.01, np.log(max(0.1 * np.mean(Q), 1e-2))

    if not np.isfinite(p3) or p3 <= 0:
        p3 = 0.01
    if not np.isfinite(p2):
        p2 = np.log(max(0.1 * np.mean(Q), 1e-2))
    if not np.isfinite(p1):
        p1 = 0.0
    if not np.isfinite(p0):
        p0 = float(Q[0])
    return [float(p0), float(p1), float(p2), float(p3)]


def _fit_QV_scipy(V, Q, Qerr, seed):
    if not HAVE_SCIPY:
        return list(seed), [0.0] * 4, False
    try:
        popt, pcov = _curve_fit(
            lambda x, a, b, c, d: a + b * x + np.exp(c + d * x),
            V, Q, p0=seed, sigma=Qerr, absolute_sigma=True, maxfev=40000)
        return list(popt), list(np.sqrt(np.diag(pcov))), True
    except Exception:
        return list(seed), [0.0] * 4, False


def fit_QV(V, Q, name="qv", verbose=False):
    """Fit pooled Q(V). Returns params/errors, ROOT TF1 & graph, chi2/ndf."""
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Verr = np.full(len(V), config.ERR_V)
    Qerr = np.full(len(Q), config.ERR_Q)

    gr = ROOT.TGraphErrors(len(V), array('d', V), array('d', Q),
                           array('d', Verr), array('d', Qerr))
    gr.SetName("g_" + name)

    seed = auto_seed_QV(V, Q)
    f = ROOT.TF1("f_" + name, QV_FORMULA, float(V.min()), float(V.max()))
    f.SetParameters(*seed)
    f.SetParLimits(1, 0.0, 5.0)     # p1 >= 0  (uniform physical bound)
    f.SetParLimits(3, 0.0, 0.2)     # p3 >= 0

    backend = 'root'
    res = gr.Fit(f, "SQ")
    valid = bool(res.Get()) and res.IsValid()
    params = [f.GetParameter(i) for i in range(4)]
    errors = [f.GetParError(i) for i in range(4)]
    if (not valid) or (not np.all(np.isfinite(params))):
        sp, spe, ok = _fit_QV_scipy(V, Q, Qerr, seed)
        if ok:
            params, errors, backend = sp, spe, 'scipy'
            for i, p in enumerate(params):
                f.SetParameter(i, p)

    chi2, ndf = float(f.GetChisquare()), int(f.GetNDF())
    if verbose:
        print(f"    {name:<26s} n={len(V):2d} chi2/ndf={chi2:6.2f}/{ndf:<2d} "
              f"[{backend}] p={['%.4g' % x for x in params]}")
    return dict(params=params, errors=errors, tf1=f, graph=gr,
                chi2=chi2, ndf=ndf, backend=backend, n=len(V))


# ===========================================================================
# step 2 -- parameter vs fluence
# ===========================================================================
def eval_param(coeff, phi):
    if coeff['kind'] == 'lin':
        return coeff['m'] * phi + coeff['q']
    if coeff['kind'] == 'exp':
        return coeff['a'] + np.exp(coeff['b'] * phi + coeff['c'])
    xp, fp = coeff['phi'], coeff['p']               # interp
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
    out = np.interp(phi_arr, xp, fp)
    if len(xp) >= 2:
        lo, hi = phi_arr < xp[0], phi_arr > xp[-1]
        if lo.any():
            s = (fp[1] - fp[0]) / (xp[1] - xp[0])
            out[lo] = fp[0] + s * (phi_arr[lo] - xp[0])
        if hi.any():
            s = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            out[hi] = fp[-1] + s * (phi_arr[hi] - xp[-1])
    return out if np.ndim(phi) else float(out[0])


def _chi2(phi, p, sigma, coeff):
    r = (p - eval_param(coeff, phi)) / sigma
    return float(np.sum(r * r))


def _fit_lin(phi, p, sigma):
    m, q = np.polyfit(phi, p, 1, w=1.0 / sigma)
    c = dict(kind='lin', m=float(m), q=float(q))
    return c, _chi2(phi, p, sigma, c), 2


def _fit_exp_root(phi, p, sigma):
    """Fit a + exp(b*phi + c) in ROOT. Returns (dict, chi2, n_par) or None."""
    n = len(phi)
    if n < 3:
        return None
    rng = max(np.ptp(p), 1e-9)
    m, _ = np.polyfit(phi, p, 1)
    a0 = float(p.min() - 0.05 * rng) if m >= 0 else float(p.max() + 0.05 * rng)
    excess = np.where((p - a0) > 0, p - a0, 0.05 * rng)
    try:
        b0, c0 = np.polyfit(phi, np.log(excess), 1)
    except Exception:
        b0, c0 = np.log(rng) / max(np.ptp(phi), 1.0), np.log(rng)
    gr = ROOT.TGraphErrors(n, array('d', phi), array('d', p),
                           array('d', np.zeros(n)), array('d', sigma))
    f = ROOT.TF1("fexp_param", "[0]+exp([1]*x+[2])",
                 float(phi.min()), float(phi.max()))
    f.SetParameters(a0, b0, c0)
    res = gr.Fit(f, "SQN")
    if not (bool(res.Get()) and res.IsValid()):
        return None
    a, b, cc = (f.GetParameter(i) for i in range(3))
    if not np.all(np.isfinite([a, b, cc])):
        return None
    c = dict(kind='exp', a=float(a), b=float(b), c=float(cc))
    return c, _chi2(phi, p, sigma, c), 3


def fit_param_vs_fluence(phi, p, perr, force=None):
    """Model p(phi) as lin / exp / interp. See config.PARAM_MODE."""
    phi = np.asarray(phi, dtype=float)
    p = np.asarray(p, dtype=float)
    perr = np.asarray(perr, dtype=float)
    order = np.argsort(phi)
    phi, p, perr = phi[order], p[order], perr[order]
    sigma = np.where(perr > 0, perr, np.nanmax([np.abs(p).max() * 1e-3, 1e-9]))
    n = len(phi)

    if force == 'interp':
        return dict(kind='interp', phi=phi.copy(), p=p.copy())

    lin, chi2_lin, k_lin = _fit_lin(phi, p, sigma)
    if n < config.MIN_PTS_FOR_EXP and force != 'exp':
        return lin

    exp = _fit_exp_root(phi, p, sigma)
    if exp is None:
        return lin
    ce, chi2_exp, k_exp = exp
    if force == 'exp':
        return ce
    return ce if (chi2_exp + 2 * k_exp) < (chi2_lin + 2 * k_lin) else lin


def linear_trend(phi, p, perr=None):
    phi = np.asarray(phi, dtype=float)
    p = np.asarray(p, dtype=float)
    w = None if perr is None else 1.0 / np.where(np.asarray(perr) > 0, perr, 1.0)
    m, q = np.polyfit(phi, p, 1, w=w)
    return dict(kind='lin', m=float(m), q=float(q))


# ===========================================================================
# high-level driver
# ===========================================================================
def fit_production(sensors, production, setup=None, verbose=True, param_mode=None):
    setup = setup or config.FIT_SETUP
    mode = param_mode or config.PARAM_MODE
    force = None if mode in (None, 'auto') else mode

    flus = data_io.fluences_of(sensors, production, setup)
    if verbose:
        print(f"\n=== {production} ({setup}) — {len(flus)} fluence points "
              f"[param_mode={mode}] ===")

    perpoint = {}
    P = {i: [] for i in range(4)}
    Perr = {i: [] for i in range(4)}
    for phi in flus:
        V, Q = data_io.pooled(sensors, production, phi, setup)
        fit = fit_QV(V, Q, name=f"{_short(production)}_{phi:.1e}", verbose=verbose)
        perpoint[phi] = fit
        for i in range(4):
            P[i].append(fit['params'][i])
            Perr[i].append(fit['errors'][i])

    phi_arr = np.asarray(flus, dtype=float)
    coeffs, trend = {}, {}
    for i in range(4):
        coeffs[i] = fit_param_vs_fluence(phi_arr, np.asarray(P[i]),
                                         np.asarray(Perr[i]), force=force)
        trend[i] = linear_trend(phi_arr, np.asarray(P[i]), np.asarray(Perr[i]))
    if verbose:
        _print_coeffs(production, coeffs)

    return dict(production=production, setup=setup, fluences=phi_arr,
                perpoint=perpoint, P=P, Perr=Perr, coeffs=coeffs, trend=trend)


def fit_all(sensors=None, setup=None, verbose=True, param_mode=None):
    sensors = sensors if sensors is not None else data_io.load_sensors()
    return {prod: fit_production(sensors, prod, setup, verbose, param_mode)
            for prod in data_io.productions(sensors)}


# ===========================================================================
# surface + inversion
# ===========================================================================
def Q_of_V_phi(V, phi, coeffs):
    return charge_np(V, *(eval_param(coeffs[i], phi) for i in range(4)))


def V_for_Q(Q_target, phi, coeffs, v_lo=0.0, v_hi=1200.0, n_scan=1200):
    """Highest bias where Q crosses Q_target (robust to non-monotonic fits)."""
    try:
        Vg = np.linspace(v_lo, v_hi, n_scan)
        g = Q_of_V_phi(Vg, phi, coeffs) - Q_target
        idx = np.where(np.diff(np.sign(g)) != 0)[0]
        if len(idx) == 0:
            return np.nan
        j = idx[-1]
        a, b, ga, gb = Vg[j], Vg[j + 1], g[j], g[j + 1]
        return float(a if ga == gb else a - ga * (b - a) / (gb - ga))
    except Exception:
        return np.nan


def _param_expr(coeff, idx):
    if coeff['kind'] == 'lin':
        return f"([{idx}]*x+[{idx+1}])", [coeff['m'], coeff['q']], idx + 2
    if coeff['kind'] == 'exp':
        return (f"([{idx}]+exp([{idx+1}]*x+[{idx+2}]))",
                [coeff['a'], coeff['b'], coeff['c']], idx + 3)
    raise NotImplementedError(
        "closed-form TF2 supports 'lin'/'exp' only; use Q_of_V_phi for 'interp'.")


def build_QVphi_TF2(law, name, phi_max=1.6e15, V_max=650.0):
    """Closed-form TF2 Q(phi, V), x=phi y=V, from lin/exp laws (any mix)."""
    e0, v0, i1 = _param_expr(law[0], 0)
    e1, v1, i2 = _param_expr(law[1], i1)
    e2, v2, i3 = _param_expr(law[2], i2)
    e3, v3, i4 = _param_expr(law[3], i3)
    f = ROOT.TF2(name, f"{e0} + {e1}*y + exp({e2} + {e3}*y)",
                 0.0, phi_max, 0.0, V_max)
    for i, v in enumerate(v0 + v1 + v2 + v3):
        f.SetParameter(i, v)
    f.GetXaxis().SetTitle("Fluence [n_{eq}/cm^{2}]")
    f.GetYaxis().SetTitle("V [V]")
    f.SetTitle(f"Q(V,#phi) - {name}")
    return f


# ===========================================================================
# helpers
# ===========================================================================
# ===========================================================================
# GLOBAL surface: one smooth trend over the whole fluence range, with a
# physical no-gain floor. Fits all pooled (V, Q) points simultaneously with
#     Q(V, phi) = Q_floor + exp( a0 + a1*phi + (b0 + b1*phi)*V )
# -> single continuous law (no node-to-node kinks) that asymptotes to Q_floor
#    (the no-gain diode charge) at high fluence.
# ===========================================================================
class GlobalSurface:
    def __init__(self, production, popt, qfloor, fluences):
        self.production = production
        self.popt = tuple(float(x) for x in popt)      # a0, a1, b0, b1
        self.qfloor = float(qfloor)
        self.fluences = np.asarray(fluences, dtype=float)

    def Q(self, V, phi):
        a0, a1, b0, b1 = self.popt
        V = np.asarray(V, dtype=float)
        arg = np.clip(a0 + a1 * phi + (b0 + b1 * phi) * V, -700.0, 700.0)
        return self.qfloor + np.exp(arg)

    def V_for_Q(self, Q_target, phi, v_lo=0.0, v_hi=1200.0, n_scan=1200):
        """Lowest bias reaching Q_target at phi (the surface is monotonic in V)."""
        if Q_target <= self.qfloor:
            return v_lo
        a0, a1, b0, b1 = self.popt
        b = b0 + b1 * phi
        if b <= 0:
            return np.nan
        arg = np.log(Q_target - self.qfloor) - (a0 + a1 * phi)
        V = arg / b
        return float(V) if v_lo <= V <= v_hi else (np.nan if V > v_hi else v_lo)

    def law_string(self):
        a0, a1, b0, b1 = self.popt
        return ("Q(V,phi) = %.3g + exp[ (%.4e %+.4e*phi) + (%.4e %+.4e*phi)*V ]"
                % (self.qfloor, a0, a1, b0, b1))


def fit_global(sensors, production, setup=None, qfloor=None):
    """Fit the global smooth surface (with floor) to all pooled points."""
    if not HAVE_SCIPY:
        raise RuntimeError("fit_global needs scipy")
    from scipy.optimize import curve_fit
    setup = setup or config.FIT_SETUP
    qfloor = config.Q_FLOOR if qfloor is None else qfloor
    flus = data_io.fluences_of(sensors, production, setup)
    V, Q, PH = [], [], []
    for phi in flus:
        v, q = data_io.pooled(sensors, production, phi, setup)
        V += list(v); Q += list(q); PH += [phi] * len(v)
    V, Q, PH = np.array(V), np.array(Q), np.array(PH)

    def surf(X, a0, a1, b0, b1):
        v, ph = X
        return qfloor + np.exp(a0 + a1 * ph + (b0 + b1 * ph) * v)

    sig = np.full(len(Q), config.ERR_Q)
    best = None
    for seed in ([-2.0, -1e-15, 0.035, -1.5e-17],
                 [0.0, -5e-15, 0.02, 0.0],
                 [1.0, -1e-14, 0.015, 1e-18]):
        try:
            popt, _ = curve_fit(surf, (V, PH), Q, p0=seed, sigma=sig,
                                absolute_sigma=True, maxfev=200000)
            rms = float(np.sqrt(np.mean((Q - surf((V, PH), *popt)) ** 2)))
            if best is None or rms < best[1]:
                best = (popt, rms)
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"global fit failed for {production}")
    return GlobalSurface(production, best[0], qfloor, flus)


def fit_all_global(sensors=None, setup=None, qfloor=None):
    sensors = sensors if sensors is not None else data_io.load_sensors()
    return {prod: fit_global(sensors, prod, setup, qfloor)
            for prod in data_io.productions(sensors)}


class _CoeffSurface:
    """Adapter so a per-fluence fit result exposes the GlobalSurface interface."""
    def __init__(self, result):
        self.production = result['production']
        self.coeffs = result['coeffs']
        self.fluences = np.asarray(result['fluences'], dtype=float)

    def Q(self, V, phi):
        return Q_of_V_phi(V, phi, self.coeffs)

    def V_for_Q(self, Q_target, phi, v_lo=0.0, v_hi=1200.0, n_scan=1200):
        return V_for_Q(Q_target, phi, self.coeffs, v_lo, v_hi, n_scan)


def as_surface(obj):
    """Return obj if it already is a surface, else wrap a fit-result dict."""
    if hasattr(obj, 'Q') and hasattr(obj, 'V_for_Q'):
        return obj
    return _CoeffSurface(obj)


def law_summary(result):
    """List of human-readable strings describing each p_k(phi) law."""
    out = []
    for i, name in enumerate(PARAM_NAMES):
        c = result['coeffs'][i]
        if c['kind'] == 'lin':
            out.append(f"{name}(phi) = {c['m']:+.4e}*phi {c['q']:+.4e}")
        elif c['kind'] == 'exp':
            out.append(f"{name}(phi) = {c['a']:+.4e} + exp({c['b']:+.4e}*phi "
                       f"{c['c']:+.4e})")
        else:
            nodes = ", ".join("%.4g @ %.2e" % (v, x)
                              for x, v in zip(c['phi'], c['p']))
            out.append(f"{name}(phi) = interp[{nodes}]")
    return out


def param_table(result):
    """
    pandas DataFrame with the per-fluence Q(V) fit parameters of one production:
    fluence, n points, p0..p3 (+ errors), chi2/ndf, fit backend.
    """
    import pandas as pd
    rows = []
    for phi in result['fluences']:
        f = result['perpoint'][phi]
        p, e = f['params'], f['errors']
        rows.append({
            'production': result['production'],
            'fluence': phi, 'n': f['n'],
            'p0': p[0], 'p0_err': e[0], 'p1': p[1], 'p1_err': e[1],
            'p2': p[2], 'p2_err': e[2], 'p3': p[3], 'p3_err': e[3],
            'chi2': f['chi2'], 'ndf': f['ndf'], 'backend': f['backend'],
        })
    return pd.DataFrame(rows)


def params_all(results):
    """Concatenated parameter table for several productions."""
    import pandas as pd
    return pd.concat([param_table(r) for r in results.values()],
                     ignore_index=True)


def _short(production):
    return (str(production).replace(' ', '').replace('-', '')
            .replace('PreSeries', 'PreSer'))


def _print_coeffs(production, coeffs):
    print(f"  parametrization for {production}:")
    for i, name in enumerate(PARAM_NAMES):
        c = coeffs[i]
        if c['kind'] == 'lin':
            print(f"    {name}(phi) = {c['m']:+.4e}*phi {c['q']:+.4e}")
        elif c['kind'] == 'exp':
            print(f"    {name}(phi) = {c['a']:+.4e} + exp({c['b']:+.4e}*phi "
                  f"{c['c']:+.4e})")
        else:
            nodes = ", ".join("%.3g@%.1e" % (v, x)
                              for x, v in zip(c['phi'], c['p']))
            print(f"    {name}(phi) = interp[{nodes}]")


if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)
    fit_all(verbose=True)
