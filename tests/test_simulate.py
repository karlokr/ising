import warnings

import numpy as np
import pytest

import ising
from ising import _kernels, exact
from ising.lattice import energy, magnetization, neighbour_table, random_spins

ALGORITHMS = ["metropolis", "wolff"]


def assert_close(estimate, expected, sigmas=5, slack=1e-3):
    """Statistical comparison allowing a small systematic finite-size slack."""
    assert abs(estimate.value - expected) < sigmas * estimate.error + slack, (
        f"{estimate} vs exact {expected:.6g}"
    )


# --- bookkeeping -----------------------------------------------------------


@pytest.mark.parametrize("shape, h", [((50,), 0.0), ((50,), 0.4), ((9, 13), 0.0), ((9, 13), 0.3)])
def test_metropolis_tracks_energy_and_magnetization(shape, h):
    spins = random_spins(shape, rng=0)
    table = neighbour_table(shape)
    accept = _kernels.acceptance_table(2.0, 1.0, h, table.shape[1])
    rng = np.random.default_rng(1)
    e, m = _kernels.metropolis(spins.reshape(-1), table, accept, 1.0, h, 50, rng)
    assert e[-1] == pytest.approx(energy(spins, 1.0, h))
    assert m[-1] == magnetization(spins)


@pytest.mark.parametrize("shape", [(50,), (12, 12), (8, 5)])
def test_wolff_tracks_energy_and_magnetization(shape):
    spins = random_spins(shape, rng=0)
    rng = np.random.default_rng(1)
    p_add = 1 - np.exp(-2 / 2.3)
    e, m, sizes = _kernels.wolff(spins.reshape(-1), neighbour_table(shape), p_add, 1.0, 30, 3, rng)
    assert e[-1] == pytest.approx(energy(spins))
    assert m[-1] == magnetization(spins)
    assert np.all(sizes >= 1)


def test_acceptance_table():
    table = _kernels.acceptance_table(2.0, 1.0, 0.0, 4)
    assert table.shape == (2, 5)
    # Flipping an up spin with all four neighbours up costs dE = 8J.
    assert table[1, 4] == pytest.approx(np.exp(-8 / 2.0))
    assert table[1, 0] == 1.0
    assert np.all((table > 0) & (table <= 1))


# --- simulate() API ----------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_init_array_is_not_modified(algorithm):
    init = random_spins((8, 8), rng=5)
    before = init.copy()
    result = ising.simulate(2.5, init=init, n_sweeps=20, burn_in=5, algorithm=algorithm, seed=0)
    np.testing.assert_array_equal(init, before)
    assert result.spins.shape == (8, 8)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_seed_reproducibility(algorithm):
    a = ising.simulate(2.3, 10, n_sweeps=50, burn_in=10, algorithm=algorithm, seed=7)
    b = ising.simulate(2.3, 10, n_sweeps=50, burn_in=10, algorithm=algorithm, seed=7)
    c = ising.simulate(2.3, 10, n_sweeps=50, burn_in=10, algorithm=algorithm, seed=8)
    np.testing.assert_array_equal(a.energy, b.energy)
    np.testing.assert_array_equal(a.spins, b.spins)
    assert not np.array_equal(a.energy, c.energy)


def test_result_shapes_and_values():
    r = ising.simulate(2.0, 6, dim=1, n_sweeps=40, burn_in=0, init="cold", seed=0)
    assert r.energy.shape == r.magnetization.shape == (40,)
    assert r.n_sites == 6
    assert r.spins.dtype == np.int8


def test_low_temperature_cold_start_stays_ordered_without_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = ising.simulate(0.01, 16, n_sweeps=100, burn_in=0, init="cold", seed=0)
    assert np.all(r.magnetization == 256)
    assert np.all(r.energy == -512)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"T": 0.0},
        {"T": -1.0},
        {"algorithm": "glauber"},
        {"algorithm": "wolff", "h": 0.1},
        {"algorithm": "wolff", "J": -1.0},
        {"n_sweeps": 0},
        {"dim": 3},
        {"init": "warm"},
        {"init": np.ones((4, 4)), "L": 5},
    ],
)
def test_invalid_arguments(kwargs):
    params = {"T": 2.0, "L": 4, **kwargs}
    with pytest.raises(ValueError):
        ising.simulate(**params)


def test_equilibration_detects_convergence():
    eq = ising.equilibration(3.0, 16, n_sweeps=500, seed=0)
    assert eq.sweep is not None and eq.sweep < 200
    assert eq.hot.burn_in == eq.cold.burn_in == 0


# --- physics against exact results ------------------------------------------


@pytest.mark.parametrize("algorithm, n_sweeps", [("metropolis", 40_000), ("wolff", 20_000)])
@pytest.mark.parametrize("T", [1.8, 3.5])
def test_2d_matches_onsager(algorithm, n_sweeps, T):
    r = ising.simulate(T, 32, n_sweeps=n_sweeps, burn_in=2_000, algorithm=algorithm, seed=11)
    obs = r.observables()
    assert_close(obs.energy, exact.onsager_energy(T))
    assert_close(obs.heat_capacity, exact.onsager_heat_capacity(T), slack=0.01)
    if T < exact.TC:
        assert_close(obs.abs_magnetization, exact.onsager_magnetization(T))


def test_algorithms_agree_near_tc():
    T = 2.3
    m = ising.simulate(T, 16, n_sweeps=100_000, burn_in=5_000, seed=3).observables()
    w = ising.simulate(T, 16, n_sweeps=20_000, burn_in=1_000, algorithm="wolff", seed=4)
    w = w.observables()
    for name in ("energy", "abs_magnetization", "heat_capacity", "susceptibility"):
        a, b = getattr(m, name), getattr(w, name)
        assert abs(a.value - b.value) < 5 * np.hypot(a.error, b.error), name
    assert w.tau_magnetization < m.tau_magnetization


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_1d_zero_field(algorithm):
    T = 1.0
    obs = ising.simulate(T, 200, dim=1, n_sweeps=20_000, algorithm=algorithm, seed=5).observables()
    assert_close(obs.energy, exact.chain_energy(T))
    assert_close(obs.magnetization, 0.0, slack=0.01)


def test_1d_in_field():
    T, h = 1.0, 0.3
    obs = ising.simulate(T, 200, dim=1, h=h, n_sweeps=20_000, seed=6).observables()
    assert_close(obs.energy, exact.chain_energy(T, h=h))
    assert_close(obs.magnetization, exact.chain_magnetization(T, h=h))
