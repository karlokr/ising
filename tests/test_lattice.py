import numpy as np
import pytest

from ising.lattice import (
    delta_energy,
    energy,
    magnetization,
    neighbour_sum,
    neighbour_table,
    ordered_spins,
    random_spins,
    validate_spins,
)


def brute_force_energy(spins, J, h):
    """Explicit loop over each site's right/down bonds, as in the original code."""
    spins = np.atleast_2d(spins)
    rows, cols = spins.shape
    total = 0.0
    for i in range(rows):
        for j in range(cols):
            s = spins[i, j]
            total -= J * s * spins[i, (j + 1) % cols]
            if rows > 1:
                total -= J * s * spins[(i + 1) % rows, j]
            total -= h * s
    return total


@pytest.mark.parametrize("shape", [(8,), (2,), (5, 5), (4, 7), (2, 3)])
@pytest.mark.parametrize("J, h", [(1.0, 0.0), (-0.7, 0.3)])
def test_energy_matches_brute_force(shape, J, h):
    spins = random_spins(shape, rng=1)
    assert energy(spins, J, h) == pytest.approx(brute_force_energy(spins, J, h))


def test_ordered_energy_and_magnetization():
    assert energy(ordered_spins((6, 6))) == -2 * 36
    assert energy(ordered_spins((10,))) == -10
    assert magnetization(ordered_spins((6, 6), value=-1)) == -36


@pytest.mark.parametrize("shape", [(7,), (5, 5), (3, 6)])
def test_delta_energy_matches_full_recalculation(shape):
    spins = random_spins(shape, rng=2)
    for site in np.ndindex(shape):  # includes every boundary site
        flipped = spins.copy()
        flipped[site] *= -1
        expected = energy(flipped, 1.3, 0.4) - energy(spins, 1.3, 0.4)
        assert delta_energy(spins, site, 1.3, 0.4) == pytest.approx(expected)


@pytest.mark.parametrize("shape", [(9,), (4, 6)])
def test_neighbour_table_matches_neighbour_sum(shape):
    spins = random_spins(shape, rng=3)
    table = neighbour_table(shape)
    assert table.shape == (spins.size, 2 * len(shape))
    from_table = spins.ravel()[table].sum(axis=1)
    np.testing.assert_array_equal(from_table, neighbour_sum(spins).ravel())


def test_random_spins_are_reproducible_and_valid():
    a, b = random_spins((10, 10), rng=4), random_spins((10, 10), rng=4)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.int8
    assert set(np.unique(a)) == {-1, 1}


@pytest.mark.parametrize(
    "bad", [np.zeros((4, 4)), np.ones((2, 2, 2)), np.ones((1, 5)), np.full((3, 3), 2)]
)
def test_validate_spins_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        validate_spins(bad)
