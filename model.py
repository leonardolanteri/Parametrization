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
    # model a + exp(b*phi + c): since exp > 0, the asymptote a is BELOW the data
    # for any sign of the slope -> seed a0 just below the minimum, then the
    # positive excess (p - a0) gives (b, c) from a log fit.
    a0 = float(p.min() - 0.1 * rng)
    excess = np.clip(p - a0, 1e-9, None)
    try:
        b0, c0 = np.polyfit(phi, np.log(excess), 1)
    except Exception:
        b0, c0 = -1.0 / max(np.ptp(phi), 1.0), np.log(rng)
    best = None
    for a_seed in (a0, float(p.min() - 0.5 * rng), float(p.min() - 2.0 * rng)):
        ex = np.clip(p - a_seed, 1e-9, None)
        try:
            b_s, c_s = np.polyfit(phi, np.log(ex), 1)
        except Exception:
            b_s, c_s = b0, c0
        gr = ROOT.TGraphErrors(n, array('d', phi), array('d', p),
                               array('d', np.zeros(n)), array('d', sigma))
        f = ROOT.TF1("fexp_param", "[0]+exp([1]*x+[2])",
                     float(phi.min()), float(phi.max()))
        f.SetParameters(a_seed, b_s, c_s)
        res = gr.Fit(f, "SQN")
        if not (bool(res.Get()) and res.IsValid()):
            continue
        a, b, cc = (f.GetParameter(i) for i in range(3))
        if not np.all(np.isfinite([a, b, cc])):
            continue
        c = dict(kind='exp', a=float(a), b=float(b), c=float(cc))
        chi2 = _chi2(phi, p, sigma, c)
        if best is None or chi2 < best[1]:
            best = (c, chi2, 3)
    return best


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
    if force == 'lin':
        return lin
    if n < config.MIN_PTS_FOR_EXP and force != 'exp':
        return lin

    exp = _fit_exp_root(phi, p, sigma)
    if exp is None:
        return lin
    ce, chi2_exp, k_exp = exp
    if force == 'exp':
        return ce
    # auto: Akaike information criterion, lower is better
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
# ANALYTIC 2-step model (the recommended parametrization)
#   Step A: at each fluence fit the REDUCED curve   Q(V) = Q_floor + exp(p2 + p3*V)
#           (p0 fixed to the no-gain floor, p1 dropped -> removes the degeneracy,
#            so p2(phi) and p3(phi) come out smooth and monotonic)
#   Step B: give p2 and p3 an ANALYTIC law vs fluence  (lin or exp, auto by AIC)
#   =>      Q(V,phi) = Q_floor + exp( p2(phi) + p3(phi)*V ),  -> Q_floor at high phi
# ===========================================================================
def fit_QV_reduced(V, Q, qfloor, name="qvr"):
    """Fit Q(V) = qfloor + exp(p2 + p3*V). Returns p2, p3, err2, err3, chi2, ndf."""
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Qerr = np.full(len(Q), config.ERR_Q)
    Verr = np.full(len(V), config.ERR_V)
    seed = auto_seed_QV(V, Q)
    s2, s3 = seed[2], (seed[3] if seed[3] > 0 else 0.01)
    gr = ROOT.TGraphErrors(len(V), array('d', V), array('d', Q),
                           array('d', Verr), array('d', Qerr))
    f = ROOT.TF1("fr_" + name, f"{qfloor}+exp([0]+[1]*x)",
                 float(V.min()), float(V.max()))
    f.SetParameters(s2, s3)
    f.SetParLimits(1, 0.0, 0.2)                 # p3 >= 0
    res = gr.Fit(f, "SQ")
    p2, p3 = f.GetParameter(0), f.GetParameter(1)
    e2, e3 = f.GetParError(0), f.GetParError(1)
    ok = bool(res.Get()) and res.IsValid() and np.all(np.isfinite([p2, p3]))
    if not ok and HAVE_SCIPY:
        from scipy.optimize import curve_fit
        try:
            popt, pcov = curve_fit(lambda x, a, b: qfloor + np.exp(a + b * x),
                                   V, Q, p0=[s2, s3], sigma=Qerr,
                                   absolute_sigma=True, maxfev=40000)
            p2, p3 = popt
            e2, e3 = np.sqrt(np.diag(pcov))
        except Exception:
            pass
    return (float(p2), float(p3), float(e2), float(e3),
            float(f.GetChisquare()), int(f.GetNDF()))


class GlobalSurface:
    """Q(V,phi) = qfloor + exp( p2(phi) + p3(phi)*V ) with analytic p2, p3 laws."""
    def __init__(self, production, qfloor, law2, law3, fluences,
                 p2_nodes, p3_nodes, p2_err, p3_err):
        self.production = production
        self.qfloor = float(qfloor)
        self.law2, self.law3 = law2, law3
        self.fluences = np.asarray(fluences, dtype=float)
        self.p2_nodes = np.asarray(p2_nodes, dtype=float)
        self.p3_nodes = np.asarray(p3_nodes, dtype=float)
        self.p2_err = np.asarray(p2_err, dtype=float)
        self.p3_err = np.asarray(p3_err, dtype=float)

    def p2(self, phi):
        return eval_param(self.law2, phi)

    def p3(self, phi):
        return eval_param(self.law3, phi)

    def Q(self, V, phi):
        V = np.asarray(V, dtype=float)
        arg = np.clip(self.p2(phi) + self.p3(phi) * V, -700.0, 700.0)
        return self.qfloor + np.exp(arg)

    def V_for_Q(self, Q_target, phi, v_lo=0.0, v_hi=1200.0, n_scan=1200):
        """Bias to reach Q_target at phi (analytic inverse; monotonic in V)."""
        if Q_target <= self.qfloor:
            return v_lo
        p3 = float(self.p3(phi))
        if p3 <= 0:
            return np.nan
        V = (np.log(Q_target - self.qfloor) - float(self.p2(phi))) / p3
        return float(V) if v_lo <= V <= v_hi else (np.nan if V > v_hi else v_lo)

    @staticmethod
    def _law_str(law, nm):
        if law['kind'] == 'lin':
            return f"{nm}(phi) = {law['m']:+.4e}*phi {law['q']:+.4e}"
        if law['kind'] == 'exp':
            return (f"{nm}(phi) = {law['a']:+.4e} + exp({law['b']:+.4e}*phi "
                    f"{law['c']:+.4e})")
        return f"{nm}(phi) = interp(nodes)"

    def law_string(self):
        return (f"Q(V,phi) = {self.qfloor:.3g} + exp( p2(phi) + p3(phi)*V )\n"
                f"       {self._law_str(self.law2, 'p2')}\n"
                f"       {self._law_str(self.law3, 'p3')}")

    def tf2(self, name=None, phi_max=1.6e15, V_max=650.0):
        """The analytic surface as a ROOT TF2 (x = phi, y = V)."""
        name = name or ("gsurf_" + _short(self.production))
        e2, v2, i1 = _param_expr(self.law2, 0)
        e3, v3, i2 = _param_expr(self.law3, i1)
        f = ROOT.TF2(name, f"{self.qfloor} + exp( {e2} + ({e3})*y )",
                     0.0, phi_max, 0.0, V_max)
        for i, v in enumerate(v2 + v3):
            f.SetParameter(i, v)
        f.GetXaxis().SetTitle("Fluence [n_{eq}/cm^{2}]")
        f.GetYaxis().SetTitle("V [V]")
        f.SetTitle(f"Q(V,#phi) - {self.production}")
        return f


def fit_global(sensors, production, setup=None, qfloor=None, mode=None):
    """
    Build the analytic surface for one production:
      per-fluence reduced fits -> p2(phi_j), p3(phi_j) -> analytic laws (lin/exp).
    `mode` in {'auto','lin','exp'} selects the vs-fluence law (default 'auto').
    """
    setup = setup or config.FIT_SETUP
    qfloor = config.Q_FLOOR if qfloor is None else qfloor
    mode = mode or config.LAW_MODE
    force = None if mode in (None, 'auto', 'interp') else mode   # never interp
    flus = data_io.fluences_of(sensors, production, setup)
    p2n, p3n, e2n, e3n = [], [], [], []
    for phi in flus:
        V, Q = data_io.pooled(sensors, production, phi, setup)
        p2, p3, e2, e3, _, _ = fit_QV_reduced(
            V, Q, qfloor, name=f"{_short(production)}_{phi:.1e}")
        p2n.append(p2); p3n.append(p3); e2n.append(e2); e3n.append(e3)
    phi = np.asarray(flus, dtype=float)
    law2 = fit_param_vs_fluence(phi, np.array(p2n), np.array(e2n), force=force)
    law3 = fit_param_vs_fluence(phi, np.array(p3n), np.array(e3n), force=force)
    return GlobalSurface(production, qfloor, law2, law3, phi,
                         p2n, p3n, e2n, e3n)


def fit_all_global(sensors=None, setup=None, qfloor=None, mode=None):
    sensors = sensors if sensors is not None else data_io.load_sensors()
    return {prod: fit_global(sensors, prod, setup, qfloor, mode)
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
