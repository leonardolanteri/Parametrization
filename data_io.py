"""
data_io.py
==========
Read the Q(V) measurements from the Excel workbook.

Layout understood (one *sheet per production*, e.g. 'HPK', 'FBK'):
  * data is arranged in side-by-side blocks, one block per sensor scan;
  * each block spans three columns -- V, Q, Res -- optionally preceded by a
    blank spacer column;
  * one header row holds the tokens 'V' / 'Q' / 'Res' for every block;
  * the row just above holds the block label (sensor name, with the fluence
    written in it, e.g. 'HPK6 S4-B - 1e15').

From each block we build a record:
  production : the sheet name
  label      : the block label
  fluence    : parsed from the label (neq/cm2); 0 if none is written
  setup      : 'TB' if the label mentions a test beam, else 'Beta'
  voltage/charge/resolution : the numeric columns (NaN rows dropped)

This parser is generic: any number of sheets / blocks / fluences works.
"""

import re
import numpy as np
import pandas as pd

import config

_FLU_RE = re.compile(r'(\d+(?:\.\d+)?)\s*[eE]\s*(\d+)')


def _parse_fluence(label):
    """Extract a fluence like 1.5e15 from the label text; 0.0 if absent."""
    m = _FLU_RE.search(str(label))
    if not m:
        return 0.0
    return float(f"{m.group(1)}e{m.group(2)}")


def _parse_setup(label):
    """'TB' if the label mentions a test beam, otherwise 'Beta'."""
    s = str(label).lower()
    if 'tb' in s or 'test beam' in s or 'testbeam' in s:
        return 'TB'
    return 'Beta'


def _header_row(raw, max_scan=6):
    """Row index whose cells contain the most V/Q/Res tokens."""
    def score(r):
        return sum(str(x).strip().upper() in ('V', 'Q', 'RES', 'RESOLUTION')
                   for x in raw.iloc[r])
    n = min(max_scan, len(raw))
    return max(range(n), key=score)


def load_sheet(path, sheet):
    """Return the list of sensor records contained in one sheet."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr = _header_row(raw)
    lbl_row = max(hdr - 1, 0)
    heads = raw.iloc[hdr]

    records = []
    for c in range(raw.shape[1]):
        if str(heads[c]).strip().upper() != 'V':
            continue
        label = str(raw.iloc[lbl_row, c]).strip()
        if label.lower() in ('nan', ''):
            label = f"{sheet} block@col{c}"
        V = pd.to_numeric(raw.iloc[hdr + 1:, c], errors='coerce')
        Q = pd.to_numeric(raw.iloc[hdr + 1:, c + 1], errors='coerce') \
            if c + 1 < raw.shape[1] else pd.Series(dtype=float)
        R = pd.to_numeric(raw.iloc[hdr + 1:, c + 2], errors='coerce') \
            if c + 2 < raw.shape[1] else pd.Series(dtype=float)
        mask = V.notna() & Q.notna()
        if mask.sum() == 0:
            continue
        records.append(dict(
            production=str(sheet).strip(),
            label=label,
            # nominal fluence from the label, rescaled to the measured dose
            fluence=_parse_fluence(label) * config.fluence_scale_for(str(sheet).strip()),
            setup=_parse_setup(label),
            voltage=V[mask].to_numpy(dtype=float),
            charge=Q[mask].to_numpy(dtype=float),
            resolution=R[mask].reindex(V[mask].index).to_numpy(dtype=float),
        ))
    return records


def load_sensors(path=None, sheets=None):
    """
    Load every sensor record from the workbook.
    `sheets`: optional list of sheet names (productions) to restrict to.
    """
    path = path or config.DATA_FILE
    xls = pd.ExcelFile(path)
    names = sheets if sheets is not None else xls.sheet_names
    out = []
    for sh in names:
        out.extend(load_sheet(path, sh))
    return out


# --- convenience selectors --------------------------------------------------
def productions(sensors):
    """Productions in first-seen order."""
    seen = []
    for s in sensors:
        if s['production'] not in seen:
            seen.append(s['production'])
    return seen


def fluences_of(sensors, production, setup='Beta'):
    return sorted({s['fluence'] for s in sensors
                   if s['production'] == production and s['setup'] == setup})


def pooled(sensors, production, fluence, setup='Beta', tol=1e13):
    """Concatenated (V, Q) of all matching sensors."""
    V, Q = [], []
    for s in sensors:
        if s['production'] != production or s['setup'] != setup:
            continue
        if abs(s['fluence'] - fluence) > tol:
            continue
        V.append(s['voltage'])
        Q.append(s['charge'])
    if not V:
        return np.array([]), np.array([])
    return np.concatenate(V), np.concatenate(Q)


def summary(sensors):
    """Human-readable table of what was loaded."""
    lines = []
    for prod in productions(sensors):
        lines.append(f"[{prod}]")
        for s in [x for x in sensors if x['production'] == prod]:
            lines.append(
                f"   {s['label']:<28s} setup={s['setup']:<4s} "
                f"phi={s['fluence']:.2e}  n={len(s['voltage']):2d}  "
                f"V[{s['voltage'].min():.0f}..{s['voltage'].max():.0f}]  "
                f"Q[{s['charge'].min():.2f}..{s['charge'].max():.2f}]")
    return "\n".join(lines)


if __name__ == "__main__":
    s = load_sensors()
    print(summary(s))
