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

# --- experimental uncertainties (the raw data has no error bars) ------------
# Charge error [fC]: default 1.0, but meant to be switchable (e.g. 2.0).
ERR_Q = 1.0
ERR_V = 2.0          # bias error [V]
RELERR_PHI = 0.10    # relative fluence error (10%)

# --- parametrization of each p_k vs fluence ---------------------------------
# 'auto' : linear if < MIN_PTS_FOR_EXP fluence points, else best of exp/lin (AIC)
# 'lin'  : always linear          m*phi + q
# 'exp'  : exponential where possible    a + exp(b*phi + c)
# 'interp': node-exact piecewise-linear (reproduces every measured fluence)
PARAM_MODE = "auto"
MIN_PTS_FOR_EXP = 4

# --- analysis knobs ---------------------------------------------------------
Q_TARGET = 8.0       # fC, desired charge (e.g. end of life)
FIT_SETUP = "Beta"   # which setup is fitted; others are shown as cross-check


def set_charge_error(value):
    """Convenience setter for the charge error, e.g. config.set_charge_error(2.0)."""
    global ERR_Q
    ERR_Q = float(value)
    return ERR_Q
