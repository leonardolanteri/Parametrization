# Q(V, φ) parametrization

Parametrization of the charge **Q** collected by ETL LGADs as a function of bias
**V** and neutron-equivalent fluence **φ**. Data is read directly from an Excel
workbook; the fits need no manual tuning.

## Layout

```
parametrization/
├── parametrization_data.xlsx   # the data (one sheet per production: HPK, FBK, …)
├── config.py                   # all settings (errors, modes, paths, target Q)
├── data_io.py                  # Excel reader (block layout → tidy records)
├── model.py                    # fitting engine (auto-seed, vs-φ laws, surface)
├── plots.py                    # ROOT plots
├── run.py                      # command-line driver
└── parametrization.ipynb       # notebook driver
```

Run with the ROOT conda env (ROOT + numpy/scipy/pandas/openpyxl):

```bash
conda activate tb_analysis2
python run.py --save
```

## Excel format

One **sheet per production** (`HPK`, later `FBK`, …). Inside a sheet, sensors are
side-by-side blocks of three columns **V / Q / Res**, optionally separated by a
blank column. A header row holds the `V`/`Q`/`Res` tokens; the row above holds
the block label (sensor name, with the fluence written in it, e.g.
`HPK6 S4-B - 1e15`). The reader (`data_io.py`) figures out:

- **production** = sheet name
- **fluence**    = parsed from the label (`1e15`, `1.5e15`, `4e14`, …; none → 0)
- **setup**      = `TB` if the label mentions a test beam, else `Beta`

Any number of sheets / blocks / fluences works. Sensors at the same fluence are
pooled for the fit.

## Model

**Per fluence:** `Q(V) = p0 + p1·V + exp(p2 + p3·V)`, seeded analytically so it
converges without per-point tuning. Only uniform physical bounds are applied
(`p1 ≥ 0`, `p3 ≥ 0`).

**Vs fluence** (`config.PARAM_MODE`):
- `auto` (default): linear if `< MIN_PTS_FOR_EXP` (=4) fluences, otherwise the
  better of exponential / linear per parameter (by AIC).
- `lin` / `exp` / `interp` force a specific law (`interp` = node-exact
  piecewise-linear, reproduces every measured fluence exactly).

Combined: `Q(V,φ) = P0(φ) + P1(φ)·V + exp( P2(φ) + P3(φ)·V )`.

**Global simultaneous model (recommended).** The headline model is a single fit
of the whole surface to all `(V, φ, Q)` points at once, with an exponential law
for each fluence-dependent parameter:

```
Q(V,φ) = p0 + p1(φ)·V + exp( p2(φ) + p3(φ)·V )
p_k(φ) = a_k + exp( b_k·φ + c_k )            # k = 1,2,3
```

`p0` is a **free** global constant (no floor imposed): it comes out ~0.3–0.5 fC,
the no-gain bulk charge, so the charge decays to it **on its own** at high fluence
(the gain term →0), never to 0. Fitting all fluences together forces the
parameters onto smooth `exp` laws, which breaks the per-fluence degeneracy (no
high-fluence outlier) — for HPK, χ²/ndf ≈ 26/50, prediction RMS ≈ 0.66 fC. Built
by `model.fit_global` → `model.SimulSurface` (`.p1/.p2/.p3(φ)`, `.Q(V,φ)`,
`.V_for_Q`, `.tf2()`); the 4-parameter per-fluence fits remain as diagnostics.

## Settings (`config.py`)

| setting | meaning | default |
|---|---|---|
| `ERR_Q` | charge error [fC] — **switchable** | `1.0` (use `2.0` when needed) |
| `ERR_V` | bias error [V] | `2.0` |
| `RELERR_PHI` | relative fluence error | `0.10` |
| `PARAM_MODE` | vs-φ law | `auto` |
| `Q_TARGET` | target charge [fC] | `8.0` |
| `PLATEAU_CHARGE` | target for the operating-point figure [fC] | `15.0` |
| `Q_FLOOR` | no-gain (bulk) charge the surface asymptotes to [fC] | `0.55` |
| `LAW_MODE` | analytic law for `p2(φ)`,`p3(φ)`: `auto` (lin/exp by AIC), `lin`, `exp` | `auto` |
| `V_SEB` | per-production SEB bias ceiling [V] | HPK 550, FBK 600 |

Change the charge error from code with `config.set_charge_error(2.0)`, or on the
command line with `python run.py --errq 2`.

## Quick API

```python
import data_io, model, plots, config
sensors = data_io.load_sensors()
results = model.fit_all(sensors)          # dict keyed by production
r = results['HPK']

model.Q_of_V_phi(500, 1e15, r['coeffs'])  # predicted charge
model.V_for_Q(8.0, 1e15, r['coeffs'])     # bias to reach 8 fC
model.build_QVphi_TF2(r['coeffs'], 'TF2') # closed-form ROOT TF2
```

## Notes

- Test-beam (`TB`) points are shown as a cross-check but are not fitted.
- With few fluence points a plain linear law for `p3` (to which Q is
  exponentially sensitive) can be inaccurate at the endpoints of the surface;
  use `param_mode='interp'` to reproduce the data exactly, or add fluence points
  so `auto` switches to the exponential law.
- `FBK` is added later simply by putting an `FBK` sheet in the workbook.
