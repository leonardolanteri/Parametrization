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

# expected no-gain (bulk) charge [fC]: charge a diode collects once the gain
# layer is gone. It is NOT imposed on the fit -- the global model's free p0 comes
# out ~0.3-0.5 fC on its own (compatible with this within the experimental
# uncertainty). Kept here only as the physical reference value.
Q_FLOOR = 0.5


def set_charge_error(value):
    """Convenience setter for the charge error, e.g. config.set_charge_error(2.0)."""
    global ERR_Q
    ERR_Q = float(value)
    return ERR_Q
