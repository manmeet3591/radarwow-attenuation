"""Synthetic attenuated sweeps with known truth.

Rain obeys A_H = a Z^b and K_DP = A_H / alpha_true, so ZPHI is exact when alpha matches. Measured
Z = Z_true - PIA, ZDR = ZDR_true - (beta/alpha) PIA, PHIDP = offset + 2 integral(K_DP) + noise.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def synthetic_attenuated_sweep(alpha_true=0.08, beta=0.02, b=0.78, n_az=90, n=480, dr_m=250.0, phi_noise=2.0,
                               peak_dbz=52.0, seed=0, a_at_50dbz=0.35):
    rng = np.random.default_rng(seed)
    r = (np.arange(n) + 0.5) * dr_m / 1e3
    z_true = np.full((n_az, n), np.nan)
    for i in range(n_az):
        c1, c2 = rng.uniform(20, 45), rng.uniform(60, 100)
        prof = 22 + (peak_dbz - 22) * np.exp(-0.5 * ((r - c1) / rng.uniform(3, 6)) ** 2) \
            + (peak_dbz - 27) * np.exp(-0.5 * ((r - c2) / rng.uniform(4, 8)) ** 2)
        prof[(r < 3) | (r > 110)] = np.nan
        z_true[i] = prof
    a_coef = a_at_50dbz / (1e5 ** b)
    ah = np.where(np.isfinite(z_true), a_coef * (10 ** (0.1 * np.nan_to_num(z_true))) ** b, 0.0)
    dr = dr_m / 1e3
    pia = 2 * np.cumsum(ah, axis=1) * dr
    kdp = ah / alpha_true
    phi = 40.0 + 2 * np.cumsum(kdp, axis=1) * dr + rng.normal(0, phi_noise, z_true.shape)
    zdr_true = np.where(np.isfinite(z_true), np.clip(0.2 + 0.05 * (z_true - 20), 0.2, 3.0), np.nan)
    noise_dbz = -45.0 + 20 * np.log10(np.maximum(r, 0.1))  # receiver noise in dBZ (~-45 at 1 km, as measured on CSAPR2), rises 20 log r
    ds = xr.Dataset({
        "DBZH": (("azimuth", "range"), z_true - pia),
        "SNRH": (("azimuth", "range"), z_true - pia - noise_dbz[None, :]),
        "ZDR": (("azimuth", "range"), zdr_true - beta / alpha_true * pia),
        "PHIDP": (("azimuth", "range"), np.where(np.isfinite(z_true), phi, np.nan)),
        "RHOHV": (("azimuth", "range"), np.where(np.isfinite(z_true), 0.99, np.nan)),
        "Z_TRUE": (("azimuth", "range"), z_true),
        "ZDR_TRUE": (("azimuth", "range"), zdr_true),
        "PIA_TRUE": (("azimuth", "range"), pia),
        "KDP_TRUE": (("azimuth", "range"), kdp),
    }, coords={"azimuth": np.arange(n_az) * (360 / n_az) + 0.5, "range": r * 1e3,
               "elevation": ("azimuth", np.full(n_az, 0.5))})
    return ds
