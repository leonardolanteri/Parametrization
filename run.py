"""
run.py
======
Command-line driver: load the Excel data, fit every production, print the
parametrization and the bias needed to reach the target charge, and (optionally)
save all plots.

Examples:
    python run.py                     # fit + print summary
    python run.py --save              # also write PDFs into Plots/<timestamp>/
    python run.py --errq 2            # use +-2 fC charge error
    python run.py --mode interp       # node-exact vs-fluence law
    python run.py --production HPK     # restrict to one sheet/production
"""

import os
import argparse

import ROOT

import config
import data_io
import model
import plots


def main():
    ap = argparse.ArgumentParser(description="Q(V,phi) parametrization")
    ap.add_argument("--errq", type=float, help="charge error [fC] (default %.1f)"
                    % config.ERR_Q)
    ap.add_argument("--mode", choices=["auto", "lin", "exp", "interp"],
                    help="vs-fluence law (default %s)" % config.PARAM_MODE)
    ap.add_argument("--production", help="restrict to one sheet/production")
    ap.add_argument("--target", type=float, help="target charge [fC] (default %.1f)"
                    % config.Q_TARGET)
    ap.add_argument("--save", action="store_true", help="save plots to Plots/<ts>/")
    args = ap.parse_args()

    if args.errq is not None:
        config.set_charge_error(args.errq)
    if args.mode:
        config.PARAM_MODE = args.mode
    if args.target is not None:
        config.Q_TARGET = args.target

    ROOT.gROOT.SetBatch(True)

    sensors = data_io.load_sensors()
    print(data_io.summary(sensors))
    print(f"\n(charge error = +-{config.ERR_Q:.1f} fC, param_mode = {config.PARAM_MODE})")

    sheets = [args.production] if args.production else data_io.productions(sensors)
    results = {prod: model.fit_production(sensors, prod) for prod in sheets}

    # global smooth surfaces (one trend over the whole range, floor at Q_FLOOR)
    surfaces = {name: model.fit_global(sensors, name) for name in sheets}

    print(f"\nBias to reach Q = {config.Q_TARGET:.0f} fC (global surface):")
    for name, surf in surfaces.items():
        row = {}
        for phi in surf.fluences:
            v = surf.V_for_Q(config.Q_TARGET, phi)
            row['%.1e' % phi] = 'n/a' if v != v else '%.0f V' % v
        print(f"  {name:16s}", row)

    if args.save:
        outdir = config.PLOTS_DIR          # fixed folder: plots are overwritten
        os.makedirs(outdir, exist_ok=True)
        for name, r in results.items():
            s = model._short(name)
            plots.plot_QV(sensors, r, save=os.path.join(outdir, f"QV_{s}.pdf"))
            plots.plot_params(r, save=os.path.join(outdir, f"par_{s}.pdf"))
            plots.plot_prediction(r, save=os.path.join(outdir, f"pred_{s}.pdf"))
        plots.plot_bias_for_target(surfaces,
                                   save=os.path.join(outdir, "bias.pdf"))
        plots.plot_charge_bias_vs_fluence(
            surfaces, save=os.path.join(outdir, "charge_bias_vs_fluence.pdf"))
        print("\nplots saved in", outdir)


if __name__ == "__main__":
    main()
