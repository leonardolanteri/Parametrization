"""
config.py
=========
Central settings for the Q(V, phi) parametrization.

Edit the values here; every module reads from this file.
"""

import os

# --- paths ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "parametrization_data.xlsx")
PLOTS_DIR = os.path.join(HERE, "Plots")
# optional per-production simulation curves to overlay on the charge/bias figure
# (simulation/<production>.csv with columns phi14, charge, bias)
SIM_DIR = os.path.join(HERE, "simulation")

# --- experimental uncertainties (the raw data has no error bars) ------------
# Charge error [fC]: default 1.0, but meant to be switchable (e.g. 2.0).
ERR_Q = 1.0
ERR_V = 2.0          # bias error [V]
RELERR_PHI = 0.10    # relative fluence error (10%)

# --- parametrization of each p_k vs fluence ---------------------------------
# 'interp': node-exact piecewise-linear (reproduces every measured fluence) <-- default
# 'auto' : linear if < MIN_PTS_FOR_EXP fluence points, else best of exp/lin (AIC)
# 'lin'  : always linear          m*phi + q
# 'exp'  : exponential where possible    a + exp(b*phi + c)
#
# NOTE: for this dataset the per-fluence parameters do NOT follow a smooth law
# (parameter degeneracy + mixed sensor generations), so a global lin/exp law
# mis-predicts the charge. 'interp' reproduces every measured fluence and is the
# recommended default; switch to 'auto'/'exp' only for a smooth extrapolation.
PARAM_MODE = "interp"
MIN_PTS_FOR_EXP = 4

# NOTE on the main model (model.fit_global): it is a single GLOBAL SIMULTANEOUS
# fit of  Q(V,phi) = p0(phi) + p1(phi)*V + exp(p2(phi)+p3(phi)*V)  to all (V,phi,Q)
# points at once, with an exponential law for each parameter. It does not use
# PARAM_MODE / LAW_MODE (those drive the older per-fluence diagnostic path only).
LAW_MODE = "auto"

# Give p0 its own fluence law p0(phi)=a+exp(b*phi+c) (True) or keep it a single
# free global constant (False). The bulk charge should not depend on fluence, so
# False is the physically sober choice; True does not improve the fit here and
# leaves p0's high-fluence asymptote poorly constrained.
P0_VS_PHI = True

# --- analysis knobs ---------------------------------------------------------
Q_TARGET = 8.0       # fC, desired charge (e.g. end of life)
FIT_SETUP = "Beta"   # which setup is fitted; others are shown as cross-check

# SEB bias limit per production [V] (single-event-burnout ceiling). Used by the
# charge/bias-vs-fluence plot: above this bias the device cannot be operated, so
# the obtainable charge is Q(V_SEB, phi). Matched by exact name or by substring.
V_SEB = {"HPK": 550.0, "HPK6": 550.0, "FBK": 605.0, "FBK-LF": 605.0}
V_SEB_DEFAULT = 550.0

# target charge for the charge/bias-vs-fluence figure (the solid-line plateau)
PLATEAU_CHARGE = 15.0

# extra charge/bias-vs-fluence comparison figures: each entry is a list of exact
# production names overlaid on one plot (any family mix). save_all writes, per
# group, a log-x version and a linear-x version (up to COMPARE_LIN_PHI_MAX).
COMPARE_GROUPS = [["HPK6", "FBK-LF_PreSerie"]]
COMPARE_LIN_PHI_MAX = 3.0e15

# no-gain (bulk) charge [fC] the surface must decay to at high fluence -- the
# PER-PRODUCTION high-fluence asymptote. Matched by exact production name.
Q_INF = {"HPK": 0.50, "HPK6": 0.50, "FBK": 0.55, "FBK-LF": 0.55}
Q_INF_DEFAULT = 0.50

# The high-fluence asymptote p0 is constrained to this window [fC] (bounded, not a
# hard value): the collected charge settles to the bulk, between 0 and ~1 fC.
P0_BOUNDS = (0.0, 1.0)

# Pull the surface to its bulk asymptote at high fluence. With few / low fluence
# points the gain-decay rate is NOT in the data (e.g. FBK is measured only up to
# 1.5e15, where Q is still 2-10 fC), so the extrapolated charge would otherwise
# plateau well above the bulk. When True the global fit adds a soft high-fluence
# constraint Q = Q_INF (flat in V) beyond the data (see ANCHOR_FACTORS), so the
# surface decays to the per-production bulk. It is a physical prior on the known
# asymptote, not a hard clamp, and leaves the real-data fit essentially unchanged.
# Set False to fit the data alone (then sparse productions will not converge).
ANCHOR_FLOOR = True
# bulk-constraint fluences RELATIVE to each production's highest measured fluence
# (phi_max), so they always sit safely beyond the data and never distort the last
# measured point: the anchor is applied at phi_max * ANCHOR_FACTORS.
ANCHOR_FACTORS = [2.0, 3.5, 6.0, 10.0]
ANCHOR_ERR = 0.5                 # [fC] weight (smaller = stronger pull to Q_INF)


# Per-production fluence correction (multiplicative). The Excel keeps the NOMINAL
# fluence in the block labels; this rescales it to the measured dose for a given
# production, applied when the data is read (data_io). Exact production name only.
# e.g. FBK-LF_PreSerie received ~17% more fluence than nominal -> factor 1.17.
FLUENCE_SCALE = {"FBK-LF_PreSerie": 1.17}


def fluence_scale_for(production):
    """Multiplicative fluence correction for a production (1.0 if none set)."""
    return float(FLUENCE_SCALE.get(production, 1.0))


def set_charge_error(value):
    """Convenience setter for the charge error, e.g. config.set_charge_error(2.0)."""
    global ERR_Q
    ERR_Q = float(value)
    return ERR_Q


def _match(d, production, default):
    """Per-production lookup: exact name first, then by family (FBK*/HPK*), else default."""
    if production in d:
        return d[production]
    u = str(production).upper()
    for fam in ("FBK", "HPK"):
        if fam in u:
            for k, v in d.items():
                if fam in k.upper():
                    return v
    return default


def q_inf_for(production):
    """High-fluence bulk charge Q_inf [fC] for a production (family fallback)."""
    return _match(Q_INF, production, Q_INF_DEFAULT)


def v_seb_for(production):
    """SEB bias ceiling [V] for a production (family fallback)."""
    return _match(V_SEB, production, V_SEB_DEFAULT)
