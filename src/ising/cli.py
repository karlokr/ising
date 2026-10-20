"""Command-line interface: ``ising run | scan | equilibrate | animate``."""

import argparse
import sys
import time

import numpy as np


def parse_temperatures(specs):
    """Expand ``start:stop:step`` ranges (inclusive) and plain numbers."""
    temps = []
    for spec in specs:
        if ":" in spec:
            start, stop, step = (float(x) for x in spec.split(":"))
            temps.extend(np.arange(start, stop + step / 2, step))
        else:
            temps.append(float(spec))
    return np.round(np.array(temps), 10)


def _common(parser, sweeps=10_000, burn_in=1_000):
    parser.add_argument("--dim", type=int, default=2, choices=(1, 2), help="lattice dimension")
    parser.add_argument("-J", type=float, default=1.0, help="coupling (default 1)")
    parser.add_argument("--field", type=float, default=0.0, help="external field h (default 0)")
    parser.add_argument(
        "--algorithm", "-a", default="metropolis", choices=("metropolis", "wolff"),
        help="update algorithm (wolff: h = 0 only, much faster near Tc)",
    )  # fmt: skip
    parser.add_argument("--sweeps", type=int, default=sweeps, help="measured sweeps")
    parser.add_argument("--burn-in", type=int, default=burn_in, help="discarded sweeps")
    parser.add_argument("--seed", type=int, default=None, help="random seed")


def _sim_kwargs(args):
    return {
        "dim": args.dim,
        "J": args.J,
        "h": args.field,
        "algorithm": args.algorithm,
        "n_sweeps": args.sweeps,
        "burn_in": args.burn_in,
    }


def _save(fig, path):
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def cmd_run(args):
    from .plotting import plot_snapshots
    from .simulate import simulate

    start = time.perf_counter()
    result = simulate(args.T, args.L, seed=args.seed, **_sim_kwargs(args))
    elapsed = time.perf_counter() - start
    print(result.observables())
    print(f"  {args.algorithm}, {args.sweeps + args.burn_in} sweeps in {elapsed:.2f} s")
    if args.snapshot:
        _save(plot_snapshots([result.spins], [f"T = {args.T:g}"]), args.snapshot)


def cmd_scan(args):
    from .plotting import plot_binder, plot_observables
    from .scan import temperature_scan, write_csv

    temps = parse_temperatures(args.T)
    scans = []
    for L in args.L:
        start = time.perf_counter()
        scan = temperature_scan(temps, L, seed=args.seed, workers=args.workers, **_sim_kwargs(args))
        scans.append(scan)
        print(f"L = {L}: {len(temps)} temperatures in {time.perf_counter() - start:.1f} s")
        print(f"  {'T':>7} {'e':>10} {'|m|':>9} {'c':>9} {'chi':>10} {'U':>7}")
        for i, T in enumerate(scan.T):
            print(
                f"  {T:7.4f} {scan.energy[i]:10.5f} {scan.abs_magnetization[i]:9.5f} "
                f"{scan.heat_capacity[i]:9.4f} {scan.susceptibility[i]:10.4f} "
                f"{scan.binder[i]:7.4f}"
            )
    if args.out:
        write_csv(scans, args.out)
        print(f"saved {args.out}")
    if args.plot:
        _save(plot_observables(scans, exact_curves=args.dim == 2 and args.field == 0), args.plot)
    if args.binder_plot:
        _save(plot_binder(scans), args.binder_plot)


def cmd_equilibrate(args):
    from .plotting import plot_equilibration
    from .simulate import equilibration

    eq = equilibration(
        args.T, args.L, dim=args.dim, J=args.J, h=args.field, n_sweeps=args.sweeps,
        window=args.window, algorithm=args.algorithm, seed=args.seed,
    )  # fmt: skip
    if eq.sweep is None:
        print(f"hot and cold starts did not meet within {args.sweeps} sweeps")
    else:
        print(f"equilibrated after ~{eq.sweep} sweeps")
    if args.plot:
        _save(plot_equilibration(eq), args.plot)


def cmd_animate(args):
    import matplotlib.pyplot as plt

    from .plotting import animate

    anim = animate(
        args.T, args.L, frames=args.frames, sweeps_per_frame=args.sweeps_per_frame,
        algorithm=args.algorithm, seed=args.seed, init=args.init,
    )  # fmt: skip
    anim.save(args.out, writer="pillow", fps=args.fps)
    plt.close("all")
    print(f"saved {args.out}")


def build_parser():
    parser = argparse.ArgumentParser(prog="ising", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate one temperature and print observables")
    run.add_argument("--T", type=float, required=True, help="temperature")
    run.add_argument("--L", type=int, required=True, help="linear lattice size")
    run.add_argument("--snapshot", metavar="PNG", help="save an image of the final lattice")
    _common(run)
    run.set_defaults(func=cmd_run)

    scan = sub.add_parser("scan", help="sweep temperatures for one or more lattice sizes")
    scan.add_argument(
        "--T", nargs="+", required=True, metavar="SPEC",
        help="temperatures: numbers and/or inclusive start:stop:step ranges",
    )  # fmt: skip
    scan.add_argument("--L", type=int, nargs="+", required=True, help="lattice sizes")
    scan.add_argument("--workers", type=int, default=None, help="processes (default: all CPUs)")
    scan.add_argument("--out", metavar="CSV", help="write results to a CSV file")
    scan.add_argument("--plot", metavar="PNG", help="plot e, |m|, c and chi against T")
    scan.add_argument("--binder-plot", metavar="PNG", help="plot the Binder cumulant against T")
    _common(scan)
    scan.set_defaults(func=cmd_scan)

    eq = sub.add_parser("equilibrate", help="compare hot and cold starts")
    eq.add_argument("--T", type=float, required=True, help="temperature")
    eq.add_argument("--L", type=int, required=True, help="linear lattice size")
    eq.add_argument("--window", type=int, default=10, help="sweeps averaged per comparison")
    eq.add_argument("--plot", metavar="PNG", help="plot both runs")
    _common(eq, sweeps=1_000, burn_in=0)
    eq.set_defaults(func=cmd_equilibrate)

    anim = sub.add_parser("animate", help="save a GIF of the lattice evolving")
    anim.add_argument("--T", type=float, required=True, help="temperature")
    anim.add_argument("--L", type=int, default=128, help="linear lattice size")
    anim.add_argument("--frames", type=int, default=150)
    anim.add_argument("--sweeps-per-frame", type=int, default=1)
    anim.add_argument("--fps", type=int, default=20)
    anim.add_argument("--init", choices=("hot", "cold"), default="hot")
    anim.add_argument("--algorithm", "-a", default="metropolis", choices=("metropolis", "wolff"))
    anim.add_argument("--seed", type=int, default=None)
    anim.add_argument("--out", default="ising.gif", metavar="GIF")
    anim.set_defaults(func=cmd_animate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "burn_in", 0) < 0 or getattr(args, "sweeps", 1) < 1:
        sys.exit("need --sweeps >= 1 and --burn-in >= 0")
    try:
        args.func(args)
    except ValueError as err:
        sys.exit(f"error: {err}")


if __name__ == "__main__":
    main()
