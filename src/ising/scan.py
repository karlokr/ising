"""Temperature scans, run in parallel across processes."""

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields

import numpy as np

from .analysis import Estimate
from .simulate import simulate

COLUMNS = (
    "energy",
    "magnetization",
    "abs_magnetization",
    "heat_capacity",
    "susceptibility",
    "binder",
)


@dataclass(frozen=True)
class Scan:
    """Observables at a series of temperatures for one lattice size.

    Each observable is an array of values with a matching ``<name>_err`` array.
    """

    L: int
    T: np.ndarray
    energy: np.ndarray
    energy_err: np.ndarray
    magnetization: np.ndarray
    magnetization_err: np.ndarray
    abs_magnetization: np.ndarray
    abs_magnetization_err: np.ndarray
    heat_capacity: np.ndarray
    heat_capacity_err: np.ndarray
    susceptibility: np.ndarray
    susceptibility_err: np.ndarray
    binder: np.ndarray
    binder_err: np.ndarray
    tau_energy: np.ndarray
    tau_magnetization: np.ndarray

    def estimate(self, name, i):
        return Estimate(getattr(self, name)[i], getattr(self, f"{name}_err")[i])


def _one_point(args):
    T, L, kwargs, seed = args
    return simulate(T, L, seed=seed, **kwargs).observables()


def temperature_scan(temperatures, L, *, seed=None, workers=None, **kwargs):
    """Run :func:`ising.simulate` at each temperature and collect observables.

    Extra keyword arguments are passed to :func:`ising.simulate`. Each
    temperature gets an independent random stream derived from ``seed``.
    ``workers`` sets the number of processes (default: all CPUs; 1 runs
    serially in this process).
    """
    temperatures = np.asarray(temperatures, dtype=float)
    seeds = np.random.SeedSequence(seed).spawn(len(temperatures))
    jobs = [(T, L, kwargs, s) for T, s in zip(temperatures, seeds, strict=True)]
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        results = [_one_point(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one_point, jobs))
    data = {"L": int(L), "T": temperatures}
    for name in COLUMNS:
        data[name] = np.array([getattr(r, name).value for r in results])
        data[f"{name}_err"] = np.array([getattr(r, name).error for r in results])
    data["tau_energy"] = np.array([r.tau_energy for r in results])
    data["tau_magnetization"] = np.array([r.tau_magnetization for r in results])
    return Scan(**data)


def write_csv(scans, path):
    """Write one or more scans to a CSV file with one row per (L, T)."""
    names = [f.name for f in fields(Scan)]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for scan in scans:
            for i in range(len(scan.T)):
                row = [scan.L] + [getattr(scan, n)[i] for n in names[1:]]
                writer.writerow(row)


def read_csv(path):
    """Read scans written by :func:`write_csv`, returned as ``{L: Scan}``."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    scans = {}
    for L in sorted({int(r["L"]) for r in rows}):
        subset = [r for r in rows if int(r["L"]) == L]
        data = {"L": L}
        for f in fields(Scan)[1:]:
            data[f.name] = np.array([float(r[f.name]) for r in subset])
        scans[L] = Scan(**data)
    return scans
