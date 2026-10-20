import numpy as np
import pytest

from ising.cli import main, parse_temperatures
from ising.scan import read_csv, temperature_scan, write_csv


def test_parse_temperatures():
    np.testing.assert_allclose(parse_temperatures(["2.0:2.2:0.1", "3"]), [2.0, 2.1, 2.2, 3.0])


def test_scan_csv_round_trip(tmp_path):
    kwargs = {"n_sweeps": 200, "burn_in": 50, "algorithm": "wolff"}
    scans = [temperature_scan([2.0, 3.0], L, seed=1, workers=1, **kwargs) for L in (4, 8)]
    again = temperature_scan([2.0, 3.0], 4, seed=1, workers=1, **kwargs)
    np.testing.assert_array_equal(scans[0].energy, again.energy)
    path = tmp_path / "scan.csv"
    write_csv(scans, path)
    loaded = read_csv(path)
    assert sorted(loaded) == [4, 8]
    np.testing.assert_allclose(loaded[8].heat_capacity, scans[1].heat_capacity)
    np.testing.assert_allclose(loaded[4].T, [2.0, 3.0])


def test_cli_run(capsys, tmp_path):
    snap = tmp_path / "snap.png"
    main(
        ["run", "--T", "2.5", "--L", "8", "--sweeps", "200", "--seed", "1", "--snapshot", str(snap)]
    )
    out = capsys.readouterr().out
    assert "heat capacity" in out and "binder cumulant" in out
    assert snap.stat().st_size > 0


def test_cli_scan_and_equilibrate(capsys, tmp_path):
    csv, png = tmp_path / "s.csv", tmp_path / "s.png"
    main(["scan", "--T", "2.0:2.4:0.2", "--L", "4", "--sweeps", "200", "--workers", "1",
          "--out", str(csv), "--plot", str(png)])  # fmt: skip
    assert csv.exists() and png.exists()
    main(["equilibrate", "--T", "3", "--L", "8", "--sweeps", "300", "--seed", "0"])
    assert "equilibrated" in capsys.readouterr().out


def test_cli_reports_bad_arguments():
    with pytest.raises(SystemExit, match="wolff requires"):
        main(["run", "--T", "1", "--L", "4", "-a", "wolff", "--field", "1"])
