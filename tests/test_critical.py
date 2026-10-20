import numpy as np
import pytest

from ising import critical, exact


def test_peak_of_parabola():
    T = np.linspace(2.0, 2.6, 25)
    y = 3.0 - 10 * (T - 2.31) ** 2
    t_peak, height = critical.peak(T, y, np.full_like(T, 1e-3))
    assert t_peak.value == pytest.approx(2.31)
    assert height.value == pytest.approx(3.0)
    assert t_peak.error < 1e-3


def test_extrapolate_peak_recovers_intercept():
    L = np.array([16, 32, 64, 128])
    t_peak = 2.269 + 0.8 / L
    tc, slope = critical.extrapolate_peak(L, t_peak, np.full(4, 1e-4))
    assert tc.value == pytest.approx(2.269)
    assert slope.value == pytest.approx(0.8)


def test_fit_log_growth():
    L = np.array([8, 16, 32, 64])
    a, b = critical.fit_log_growth(L, 0.4945 * np.log(L) + 0.14, np.full(4, 0.01))
    assert a.value == pytest.approx(0.4945)
    assert b.value == pytest.approx(0.14)


def test_binder_crossing_of_lines():
    T = np.linspace(2.2, 2.35, 16)
    small = 0.61 - 0.5 * (T - 2.27)
    large = 0.61 - 1.5 * (T - 2.27)
    err = np.full_like(T, 1e-4)
    crossing = critical.binder_crossing(T, small, large, err, err)
    assert crossing.value == pytest.approx(2.27)
    with pytest.raises(ValueError):
        critical.binder_crossing(T, small + 1, large)


def test_log_divergence_fit_on_onsager_curve():
    T = np.arange(1.9, 2.7, 0.01)
    c = exact.onsager_heat_capacity(T)
    fit = critical.fit_log_divergence(T, c, np.full_like(T, 0.01), exclude=0.02, T_guess=2.27)
    assert fit["Tc"].value == pytest.approx(exact.TC, abs=2e-3)
    # Leading amplitude (2/pi)(2/Tc)^2 = 0.4945 of the logarithmic divergence.
    assert fit["A"].value == pytest.approx(2 / np.pi * (2 / exact.TC) ** 2, rel=0.03)
