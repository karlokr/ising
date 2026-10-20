"""Estimating the critical temperature from finite-size data.

Three standard methods, each using the known critical behaviour of the 2D
Ising model rather than a generic peak shape:

* Binder cumulant crossing: U_L(T) for different L cross at Tc, with only
  small corrections to scaling.
* Peak-position extrapolation: the heat-capacity and susceptibility peaks of
  an L x L lattice sit at T_peak(L) = Tc + a L^(-1/nu), with nu = 1.
* Logarithmic divergence: the specific heat exponent is alpha = 0, so
  C ~ -A ln|1 - T/Tc| near Tc and the finite-size peak grows as A ln L.
"""

import numpy as np

from .analysis import Estimate


def _weighted_polyfit(x, y, yerr, degree):
    w = None if yerr is None else 1.0 / np.asarray(yerr)
    return np.polyfit(x, y, degree, w=w)


def peak(T, y, yerr=None, frac=0.9, min_points=5, n_boot=400, seed=0):
    """Location and height of a peak by a quadratic fit around the maximum.

    The fit uses the contiguous run of points around the maximum with
    ``y >= frac * max(y)``, widened to at least ``min_points``, so the window
    scales with the peak width (narrow for large lattices, broad for small
    ones). Errors come from refitting with ``y`` resampled within ``yerr``.
    Returns ``(T_peak, height)`` as :class:`Estimate` objects.
    """
    T, y = np.asarray(T, float), np.asarray(y, float)
    i = int(np.argmax(y))
    lo, hi = i, i + 1
    while lo > 0 and y[lo - 1] >= frac * y[i]:
        lo -= 1
    while hi < len(y) and y[hi] >= frac * y[i]:
        hi += 1
    while hi - lo < min(min_points, len(y)):
        if lo > 0 and (hi == len(y) or y[lo - 1] >= y[hi]):
            lo -= 1
        else:
            hi += 1
    Ts, ys = T[lo:hi], y[lo:hi]
    es = None if yerr is None else np.asarray(yerr, float)[lo:hi]

    def vertex(values):
        a, b, c = _weighted_polyfit(Ts, values, es, 2)
        return -b / (2 * a), c - b * b / (4 * a)

    t0, h0 = vertex(ys)
    if es is None:
        return Estimate(t0, np.nan), Estimate(h0, np.nan)
    rng = np.random.default_rng(seed)
    boot = np.array([vertex(ys + es * rng.standard_normal(len(ys))) for _ in range(n_boot)])
    return Estimate(t0, boot[:, 0].std()), Estimate(h0, boot[:, 1].std())


def _linear_fit(x, y, yerr):
    """Weighted straight-line fit; returns (slope, intercept) Estimates."""
    w = 1.0 / np.asarray(yerr, float)
    coef, cov = np.polyfit(x, y, 1, w=w, cov="unscaled")
    chi2 = np.sum(((np.polyval(coef, x) - y) * w) ** 2)
    dof = max(len(x) - 2, 1)
    scale = max(1.0, chi2 / dof)  # inflate errors if the fit is poor
    err = np.sqrt(np.diag(cov) * scale)
    return Estimate(coef[0], err[0]), Estimate(coef[1], err[1])


def extrapolate_peak(L, T_peak, T_peak_err, nu=1.0):
    """Fit T_peak(L) = Tc + a L^(-1/nu); returns ``(Tc, a)``."""
    x = np.asarray(L, float) ** (-1.0 / nu)
    a, tc = _linear_fit(x, T_peak, T_peak_err)
    return tc, a


def fit_log_growth(L, height, height_err):
    """Fit peak height = A ln L + B; returns ``(A, B)``."""
    return _linear_fit(np.log(np.asarray(L, float)), height, height_err)


def binder_crossing(T, u_small, u_large, err_small=None, err_large=None, n_points=2, n_boot=400):
    """Temperature where the Binder cumulants of two lattice sizes cross.

    Below Tc the larger lattice has the larger cumulant, above Tc the smaller.
    Deep in the ordered phase both sit at 2/3 and noise can flip the sign of
    the difference, so the crossing used is the last change from negative to
    positive. A straight line is fitted to the difference over ``n_points``
    temperatures around it; the difference is S-shaped on a scale ~1/L, so
    keep this small. Errors from resampling within the given errors.
    """
    T = np.asarray(T, float)
    d = np.asarray(u_small, float) - np.asarray(u_large, float)
    change = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
    if change.size == 0:
        raise ValueError("Binder cumulants do not cross in this temperature range")
    i = int(change[-1])
    lo = max(0, min(i - n_points // 2 + 1, len(T) - n_points))
    sl = slice(lo, lo + n_points)
    have_errors = err_small is not None and err_large is not None
    err = np.hypot(np.asarray(err_small)[sl], np.asarray(err_large)[sl]) if have_errors else None

    def root(values):
        slope, intercept = _weighted_polyfit(T[sl], values, err, 1)
        return -intercept / slope

    t0 = root(d[sl])
    if not have_errors:
        return Estimate(t0, np.nan)
    rng = np.random.default_rng(0)
    boot = [root(d[sl] + err * rng.standard_normal(len(err))) for _ in range(n_boot)]
    return Estimate(t0, float(np.std(boot)))


def fit_log_divergence(T, c, c_err, exclude, T_guess=None, n_grid=2001):
    """Fit C(T) = -A ln|1 - T/Tc| + B_(below/above) + D (T - Tc) around the peak.

    Points within ``exclude`` of ``T_guess`` (default: the data maximum) are
    dropped, since on a finite lattice the divergence is rounded off there.
    Tc is found by minimising chi^2 over a grid between the innermost
    retained points; A, the two constants B and the background slope D are
    fitted linearly at each trial Tc. Returns a dict with ``Tc``, ``A`` (Estimates), ``B_below``,
    ``B_above``, ``chi2_dof`` and the boolean mask of points used.
    """
    T, c, c_err = (np.asarray(v, float) for v in (T, c, c_err))
    if T_guess is None:
        T_guess = T[np.argmax(c)]
    used = np.abs(T - T_guess) > exclude
    below, above = used & (T < T_guess), used & (T > T_guess)
    if below.sum() < 2 or above.sum() < 2:
        raise ValueError("need at least two points on each side of the peak")
    Tu, cu, wu = T[used], c[used], 1.0 / c_err[used]
    is_below = (Tu < T_guess).astype(float)

    def solve(tc):
        X = np.column_stack([-np.log(np.abs(1.0 - Tu / tc)), is_below, 1.0 - is_below, Tu - tc])
        coef, *_ = np.linalg.lstsq(X * wu[:, None], cu * wu, rcond=None)
        chi2 = np.sum(((X @ coef - cu) * wu) ** 2)
        return coef, chi2, X

    grid = np.linspace(T[below].max(), T[above].min(), n_grid)[1:-1]
    chi2 = np.array([solve(tc)[1] for tc in grid])
    k = int(np.argmin(chi2))
    tc = grid[k]
    coef, best, X = solve(tc)
    dof = max(len(Tu) - 5, 1)
    scale = max(1.0, best / dof)
    # Tc error from the chi^2 profile (delta chi^2 = scale); A, B from the
    # linear least-squares covariance at the best Tc.
    inside = grid[chi2 <= best + scale]
    tc_err = 0.5 * (inside.max() - inside.min()) if inside.size > 1 else np.nan
    cov = np.linalg.inv((X * wu[:, None]).T @ (X * wu[:, None])) * scale
    err = np.sqrt(np.diag(cov))
    return {
        "Tc": Estimate(tc, tc_err),
        "A": Estimate(coef[0], err[0]),
        "B_below": Estimate(coef[1], err[1]),
        "B_above": Estimate(coef[2], err[2]),
        "chi2_dof": best / dof,
        "used": used,
    }
