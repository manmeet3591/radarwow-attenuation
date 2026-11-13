import numpy as np
import pytest

from radarwow.attenuation import Bringi2001SelfConsistent, LinearPhiAttenuation, Testud2000ZPHI, synthetic_attenuated_sweep


def _end_error(out, ds):
    """Median |Z_corr - Z_true| over the last rain gates (where PIA is largest); C band keeps signal there."""
    err = np.abs(out["DBZH"].values - ds["Z_TRUE"].values)[:, 400:440]
    return float(np.nanmedian(err))


def test_extinct_gates_left_alone():
    ds = synthetic_attenuated_sweep(alpha_true=0.28, beta=0.05, a_at_50dbz=1.0)
    out = Testud2000ZPHI()(ds, band="X")
    dead = ~(ds["SNRH"].values > 3) & np.isfinite(ds["DBZH"].values)
    assert dead.any()
    assert np.allclose(out["DBZH"].values[dead], ds["DBZH"].values[dead])


@pytest.mark.parametrize("cls", [LinearPhiAttenuation, Testud2000ZPHI, Bringi2001SelfConsistent])
def test_corrects_when_alpha_matches(cls):
    ds = synthetic_attenuated_sweep(alpha_true=0.08)
    pia_end = float(np.nanmedian(ds["PIA_TRUE"].values[:, 400:440]))
    assert pia_end > 5  # the test is meaningful
    out = cls()(ds, band="C")
    assert _end_error(out, ds) < 1.0, _end_error(out, ds)
    assert np.nanmedian(np.abs(out["ZDR"].values - ds["ZDR_TRUE"].values)[:, 400:440]) < 0.3


def test_self_consistent_finds_alpha():
    ds = synthetic_attenuated_sweep(alpha_true=0.11, beta=0.0275)
    out = Bringi2001SelfConsistent()(ds, band="C")
    assert abs(float(np.median(out["ALPHA"].values)) - 0.11) < 0.015
    fixed = Testud2000ZPHI()(ds, band="C")  # wrong alpha (0.08)
    assert _end_error(out, ds) < _end_error(fixed, ds)


def test_raw_kept_and_pia_monotonic():
    ds = synthetic_attenuated_sweep()
    out = Testud2000ZPHI()(ds, band="C")
    assert np.allclose(out["DBZH_UNCORR"].values, ds["DBZH"].values, equal_nan=True)
    assert (np.diff(out["PIA"].values, axis=1) >= -1e-6).all()


def test_inconsistent_phase_is_not_used():
    # synthetic storm (its KDP per unit Z is ~2.5x the Marshall-Palmer expectation; real radars: 1.2-1.6):
    # most rays pass and keep their correction
    ds = synthetic_attenuated_sweep(alpha_true=0.08)
    out = Testud2000ZPHI()(ds, band="C")
    assert np.median(out["PHASE_RATIO"].values) < 10 and out["PHASE_CONSISTENT"].values.mean() > 0.95
    # same Z, but PHIDP rising 100 deg (as on the Tablazo mountain-top radar): flagged, no correction
    bad = ds.copy()
    r = bad["range"].values / 1e3
    bad["PHIDP"] = bad["PHIDP"] * 0 + 60 + 100 * r[None, :] / r.max()
    bad["DBZH"] = bad["DBZH"].clip(max=30)
    out = Testud2000ZPHI()(bad, band="C")
    assert out["PHASE_CONSISTENT"].values.mean() < 0.1
    assert float(np.nanmax(out["PIA"].values)) < 0.5
