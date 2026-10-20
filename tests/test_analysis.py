import numpy as np
import pytest

from ising.analysis import autocorrelation_time, jackknife, thermodynamics


def ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = noise[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def test_jackknife_of_mean_matches_standard_error_for_iid_data():
    x = np.random.default_rng(0).standard_normal(100_000)
    est = jackknife(np.mean, x, n_blocks=50)
    assert est.value == pytest.approx(x.mean())
    assert est.error == pytest.approx(x.std() / np.sqrt(x.size), rel=0.3)


def test_jackknife_needs_enough_samples():
    with pytest.raises(ValueError):
        jackknife(np.mean, np.ones(5), n_blocks=20)


def test_autocorrelation_time_iid_and_ar1():
    assert autocorrelation_time(np.random.default_rng(1).standard_normal(50_000)) == pytest.approx(
        0.5, abs=0.05
    )
    phi = 0.9  # exact tau_int = (1 + phi) / (2 (1 - phi)) = 9.5
    assert autocorrelation_time(ar1(phi, 200_000, 2)) == pytest.approx(9.5, rel=0.15)
    assert autocorrelation_time(np.ones(100)) == 0.5


def test_thermodynamics_on_constant_series():
    obs = thermodynamics(np.full(100, -50.0), np.full(100, 25.0), n_sites=25, T=1.0)
    assert obs.energy.value == -2.0
    assert obs.abs_magnetization.value == 1.0
    assert obs.heat_capacity.value == 0.0
    assert obs.susceptibility.value == 0.0
    assert obs.binder.value == pytest.approx(2 / 3)
