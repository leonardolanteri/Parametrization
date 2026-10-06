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
def fit_QV_reduced(V, Q, name="qvr"):
    """
    Fit Q(V) = exp(p2 + p3*V) by a weighted linear regression of ln(Q) vs V.
    This is the natural, bound-free, seed-free fit for an exponential and is
    numerically stable (the direct exp fit can diverge at some fluences). The
    weight Q^2 reproduces the sigma_Q = const error in log space.
    Returns p2, p3, err2, err3, chi2, ndf.
    """
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    good = Q > 0
    V, Q = V[good], Q[good]
    w = Q                                   # ~ 1/sigma(lnQ) = Q/sigma_Q
    (p3, p2), cov = np.polyfit(V, np.log(Q), 1, w=w, cov=True)
    e3, e2 = np.sqrt(np.diag(cov))
    pred = np.exp(p2 + p3 * V)
    chi2 = float(np.sum(((Q - pred) / config.ERR_Q) ** 2))
    ndf = max(len(V) - 2, 1)
    return float(p2), float(p3), float(e2), float(e3), chi2, ndf


# ===========================================================================
# GLOBAL SIMULTANEOUS FIT (the recommended parametrization)
#   Instead of fitting each fluence on its own (4 degenerate parameters that do
#   NOT follow any smooth trend) and then fitting those, we fit ONE closed-form
#   surface to ALL (V, phi, Q) points at once:
#
#       Q(V,phi) = p0 + p1(phi)*V + exp( p2(phi) + p3(phi)*V )
#
#   with an exponential law for each fluence-dependent parameter
#       p_k(phi) = a_k + exp( b_k * phi + c_k )        (k = 1,2,3)
#   and p0 a free global constant (NOT imposed): it comes out ~0.3-0.5 fC, the
#   no-gain / bulk charge the surface decays to at high fluence. The smoothness of
#   the laws breaks the per-fluence degeneracy, so there is no high-fluence
#   outlier and the charge converges on its own (no clamp / no imposed floor).
# ===========================================================================
PHI_SCALE = 1e15        # fluence scale used inside the simultaneous fit (numerics)


def _simul_charge(VPHI, p0, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    """Q(V,phi) = p0 + p1(phi)*V + exp(p2(phi)+p3(phi)*V), laws p_k=a+exp(b*s+c)."""
    V, PHI = VPHI
    s = np.asarray(PHI, dtype=float) / PHI_SCALE
    V = np.asarray(V, dtype=float)
    p1 = a1 + np.exp(np.clip(b1 * s + c1, -50.0, 50.0))
    p2 = a2 + np.exp(np.clip(b2 * s + c2, -50.0, 50.0))
    p3 = a3 + np.exp(np.clip(b3 * s + c3, -50.0, 50.0))
    return p0 + p1 * V + np.exp(np.clip(p2 + p3 * V, -50.0, 50.0))


class SimulSurface:
    """
    Charge surface  Q(V,phi) = p0 + p1(phi)*V + exp(p2(phi)+p3(phi)*V)  from one
    global simultaneous fit. p0 is a free global constant (the bulk charge the
    surface decays to at high fluence); there is no imposed floor / clamp.
    `coeffs` = {0: p0 (const), 1: p1 law, 2: p2 law, 3: p3 law} in the standard
    eval_param format, so the whole existing machinery (Q_of_V_phi, V_for_Q,
    build_QVphi_TF2) applies unchanged.
    """
    def __init__(self, production, coeffs, fluences, p1_err=None, p2_err=None,
                 p3_err=None, chi2=None, ndf=None, rms=None, p0_err=None):
        self.production = production
        self.coeffs = coeffs
        self.law1, self.law2, self.law3 = coeffs[1], coeffs[2], coeffs[3]
        self.p0 = float(coeffs[0]['q'])
        self.p0_err = p0_err
        self.fluences = np.asarray(fluences, dtype=float)
        self.chi2, self.ndf, self.rms = chi2, ndf, rms
        # parameter values at the measured fluences (they lie on the laws) + band
        self.p1_nodes = np.atleast_1d(self.p1(self.fluences))
        self.p2_nodes = np.atleast_1d(self.p2(self.fluences))
        self.p3_nodes = np.atleast_1d(self.p3(self.fluences))
        z = np.zeros_like(self.fluences)
        self.p1_err = z if p1_err is None else np.asarray(p1_err, dtype=float)
        self.p2_err = z if p2_err is None else np.asarray(p2_err, dtype=float)
        self.p3_err = z if p3_err is None else np.asarray(p3_err, dtype=float)

    def p1(self, phi):
        return eval_param(self.law1, phi)

    def p2(self, phi):
        return eval_param(self.law2, phi)

    def p3(self, phi):
        return eval_param(self.law3, phi)

    def Q(self, V, phi):
        """Collected charge from the full surface (no floor / clamp)."""
        return Q_of_V_phi(V, phi, self.coeffs)

    def Q_gain(self, V, phi):
        """Only the gain term exp(p2+p3*V) (diagnostic, without p0 + p1*V)."""
        V = np.asarray(V, dtype=float)
        return np.exp(np.clip(self.p2(phi) + self.p3(phi) * V, -700.0, 700.0))

    def V_for_Q(self, Q_target, phi, v_lo=0.0, v_hi=1200.0, n_scan=1200):
        """Bias to reach Q_target at phi (robust grid inversion of the surface)."""
        return V_for_Q(Q_target, phi, self.coeffs, v_lo, v_hi, n_scan)

    @staticmethod
    def _law_str(law, nm):
        if law['kind'] == 'lin':
            return f"{nm}(phi) = {law['m']:+.4e}*phi {law['q']:+.4e}"
        return (f"{nm}(phi) = {law['a']:+.4e} + exp({law['b']:+.4e}*phi "
                f"{law['c']:+.4e})")

    def law_string(self):
        head = "Q(V,phi) = p0 + p1(phi)*V + exp( p2(phi) + p3(phi)*V )"
        body = (f"       p0       = {self.p0:+.4e} fC  (free, no floor imposed)\n"
                f"       {self._law_str(self.law1, 'p1')}\n"
                f"       {self._law_str(self.law2, 'p2')}\n"
                f"       {self._law_str(self.law3, 'p3')}")
        tail = ""
        if self.chi2 is not None and self.ndf:
            tail = f"\n       chi2/ndf = {self.chi2:.1f}/{self.ndf}   RMS = {self.rms:.2f} fC"
        return head + "\n" + body + tail

    def tf2(self, name=None, phi_max=1.6e15, V_max=650.0):
        """The full surface Q(phi, V) as a closed-form ROOT TF2 (x = phi, y = V)."""
        name = name or ("gsurf_" + _short(self.production))
        return build_QVphi_TF2(self.coeffs, name, phi_max=phi_max, V_max=V_max)


def _node_errors(popt, pcov, phi):
    """Propagate the fit covariance to sigma[p_k(phi)] for k = 1,2,3 at each phi."""
    s = np.asarray(phi, dtype=float) / PHI_SCALE
    out = []
    for idx in (1, 4, 7):                       # start of (a,b,c) for p1,p2,p3
        a, b, c = popt[idx:idx + 3]
        cov = pcov[idx:idx + 3, idx:idx + 3]
        E = np.exp(np.clip(b * s + c, -50.0, 50.0))
        errs = []
        for si, Ei in zip(s, E):
            J = np.array([1.0, si * Ei, Ei])    # d p_k / d(a,b,c)
            var = float(J @ cov @ J)
            errs.append(np.sqrt(var) if np.isfinite(var) and var > 0 else 0.0)
        out.append(np.array(errs))
    return out                                  # [p1_err, p2_err, p3_err]


def fit_global(sensors, production, setup=None, verbose=True, **_):
    """
    One global simultaneous fit of the whole surface for a production:
        Q(V,phi) = p0 + p1(phi)*V + exp( p2(phi) + p3(phi)*V )
    p0 free (no imposed floor); p1,p2,p3 each an exponential law of phi. Extra
    keyword arguments (qfloor, mode) are accepted and ignored for back-compat.
    Returns a SimulSurface.
    """
    if not HAVE_SCIPY:
        raise RuntimeError("the global simultaneous fit needs scipy "
                           "(conda env tb_analysis2 has it).")
    setup = setup or config.FIT_SETUP
    flus = data_io.fluences_of(sensors, production, setup)

    Vs, Phis, Qs = [], [], []
    for ph in flus:
        v, q = data_io.pooled(sensors, production, ph, setup)
        v = np.asarray(v, dtype=float); q = np.asarray(q, dtype=float)
        Vs.append(v); Phis.append(np.full_like(v, float(ph))); Qs.append(q)
    V = np.concatenate(Vs); PHI = np.concatenate(Phis); Q = np.concatenate(Qs)
    sigma = np.full_like(Q, config.ERR_Q)

    #        p0   a1  b1  c1    a2  b2  c2    a3   b3  c3
    seed = [0.4, 0.0, -3., -8., -1.4, -1., 0.3, 1e-3, -1., -4.]
    lo = [0.0, 0.0, -80., -80., -80., -80., -80., 0.0, -80., -80.]
    hi = [5.0, 0.1, 80., 80., 80., 80., 80., 0.05, 80., 80.]
    popt, pcov = _curve_fit(_simul_charge, (V, PHI), Q, p0=seed, sigma=sigma,
                            absolute_sigma=True, maxfev=400000, bounds=(lo, hi))

    def _law(a, b, c):                          # a + exp(b*(phi/S)+c) -> standard form
        return dict(kind='exp', a=float(a), b=float(b) / PHI_SCALE, c=float(c))
    coeffs = {
        0: dict(kind='lin', m=0.0, q=float(popt[0])),
        1: _law(*popt[1:4]), 2: _law(*popt[4:7]), 3: _law(*popt[7:10]),
    }
    pred = _simul_charge((V, PHI), *popt)
    chi2 = float(np.sum(((pred - Q) / config.ERR_Q) ** 2))
    ndf = max(len(Q) - len(popt), 1)
    rms = float(np.sqrt(np.mean((pred - Q) ** 2)))
    try:
        p1e, p2e, p3e = _node_errors(popt, pcov, flus)
        p0e = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else None
    except Exception:
        p1e = p2e = p3e = None; p0e = None

    if verbose:
        print(f"\n=== {production} ({setup}) global simultaneous fit ===")
        print(f"  n = {len(Q)} points over {len(flus)} fluences   "
              f"chi2/ndf = {chi2:.1f}/{ndf}   RMS = {rms:.2f} fC")

    return SimulSurface(production, coeffs, flus, p1_err=p1e, p2_err=p2e,
                        p3_err=p3e, chi2=chi2, ndf=ndf, rms=rms, p0_err=p0e)


def fit_all_global(sensors=None, setup=None, verbose=True, **_):
    sensors = sensors if sensors is not None else data_io.load_sensors()
    return {prod: fit_global(sensors, prod, setup, verbose)
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
