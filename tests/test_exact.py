import numpy as np
import pytest

from ising import exact


def test_elliptic_integrals_known_values():
    K, E = exact._elliptic_ke(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(K[:2], [np.pi / 2, 1.6857503548125960])
    np.testing.assert_allclose(E, [np.pi / 2, 1.4674622093394272, 1.0])
    assert np.isinf(K[2])


def test_critical_temperature():
    assert exact.TC == pytest.approx(2.269185314213022)
    assert exact.critical_temperature(2.0) == pytest.approx(2 * exact.TC)


def test_onsager_energy_limits():
    assert exact.onsager_energy(exact.TC) == pytest.approx(-np.sqrt(2))
    assert exact.onsager_energy(0.2) == pytest.approx(-2.0, abs=1e-6)
    assert abs(exact.onsager_energy(1e4)) < 1e-3


@pytest.mark.parametrize("T", [0.8, 1.5, 2.0, 2.2, 2.35, 3.0, 6.0])
def test_onsager_heat_capacity_is_energy_derivative(T):
    d = 1e-5
    numeric = (exact.onsager_energy(T + d) - exact.onsager_energy(T - d)) / (2 * d)
    assert exact.onsager_heat_capacity(T) == pytest.approx(numeric, rel=1e-5)


def test_onsager_magnetization():
    T = np.array([0.5, 1.5, 2.2, exact.TC, 2.5])
    m = exact.onsager_magnetization(T)
    assert m[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(m[:3]) < 0)
    np.testing.assert_array_equal(m[3:], 0.0)


def _log_eigenvalue(beta, J, h):
    eK = np.exp(beta * J)
    return np.log(eK * np.cosh(beta * h) + np.sqrt(eK**2 * np.sinh(beta * h) ** 2 + eK**-2))


@pytest.mark.parametrize("T, J, h", [(1.0, 1.0, 0.0), (0.7, 1.0, 0.4), (2.0, -1.0, 0.3)])
def test_chain_results_are_free_energy_derivatives(T, J, h):
    beta, d = 1.0 / T, 1e-6
    u = -(_log_eigenvalue(beta + d, J, h) - _log_eigenvalue(beta - d, J, h)) / (2 * d)
    m = (_log_eigenvalue(beta, J, h + d) - _log_eigenvalue(beta, J, h - d)) / (2 * d * beta)
    assert exact.chain_energy(T, J, h) == pytest.approx(u, rel=1e-6)
    assert exact.chain_magnetization(T, J, h) == pytest.approx(m, rel=1e-6, abs=1e-12)


def test_chain_zero_field():
    assert exact.chain_energy(1.3) == pytest.approx(-np.tanh(1 / 1.3))
    assert exact.chain_magnetization(1.3) == 0.0
