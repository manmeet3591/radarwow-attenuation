"""Shared pieces for attenuation correction: the rain path along each ray and the PHIDP it crossed.

All three methods need the same inputs:
- a rain mask (signal, rhoHV above the band threshold, and below the freezing level when the file
  carries ``height_over_iso0``): attenuation relations hold for rain only;
- the path [r0, r1] from the first to the last rain gate of each ray;
- a smooth, offset-free, monotonic-ish PHIDP (``PHIDP_REC`` from radarwow-kdp, computed with
  Vulpiani2012KDP when absent).
"""

from __future__ import annotations

import numpy as np

from radarwow.core.noise import signal_mask
from radarwow.kdp import Vulpiani2012KDP
from radarwow.kdp.phase import gate_km, rain_mask


def rain_path(ds, cfg):
    """Return (valid, seg, i0, i1): rain gates, gates inside [r0, r1], first and last rain gate per ray."""
    valid = rain_mask(ds, cfg["rhohv_rain_min"]) & np.isfinite(ds["DBZH"].values)
    if "height_over_iso0" in ds:
        h = ds["height_over_iso0"].values
        valid &= ~(np.isfinite(h) & (h > 0))
    n_az, n_r = valid.shape
    any_ = valid.any(axis=1)
    i0 = np.where(any_, valid.argmax(axis=1), 0)
    i1 = np.where(any_, n_r - 1 - valid[:, ::-1].argmax(axis=1), -1)
    j = np.arange(n_r)[None, :]
    seg = (j >= i0[:, None]) & (j <= i1[:, None])
    return valid, seg, i0, i1


def phidp_rec(ds, cfg):
    """Offset-free reconstructed PHIDP (deg); reuse the kdp stage's output if present."""
    if "PHIDP_REC" not in ds:
        ds = Vulpiani2012KDP()(ds, band=cfg)
    return ds["PHIDP_REC"].values.astype(float)


def path_dphi(phi, i0, i1):
    rows = np.arange(phi.shape[0])
    d = phi[rows, np.clip(i1, 0, None)] - phi[rows, i0]
    return np.where(i1 >= i0, np.nan_to_num(d), 0.0)


def phase_consistency(ds, cfg, valid, seg, i0, i1, phi, min_dphi: float = 5.0):
    """Per ray, measured delta PHIDP over the rain path divided by the delta PHIDP the measured Z could produce.

    Expected phase: R from Z (Marshall-Palmer, Z capped at 53 dBZ), then KDP from R by inverting the band's R(KDP)
    relation (radarwow.estimation.coefficients.RKDP), integrated along the rain gates (two-way). A self-consistency
    test in the spirit of Bringi et al. 2001. Typical medians: 1.2-1.6 on NEXRAD KTLX/KDAX and FMI fianj; 18 on the
    Tablazo (Colombia) mountain-top radar, whose PHIDP rose 40-110 deg through 25-35 dBZ rain. Rays with less than
    ``min_dphi`` of measured span get ratio 1 (too little phase to judge)."""
    from radarwow.estimation import coefficients as co

    band = cfg.band.value
    z = np.where(valid & seg, 10 ** (0.1 * np.minimum(ds["DBZH"].values.astype(float), 53.0)), 0.0)
    rz = (z / 200.0) ** (1 / 1.6)
    p = co.RKDP[band]
    kexp = (rz / p["c"]) ** (1 / p["d"])
    dexp = 2 * np.nansum(np.where(valid & seg, kexp, 0.0), axis=1) * gate_km(ds)
    dphi = path_dphi(phi, i0, i1)
    return np.where(dphi >= min_dphi, dphi / np.maximum(dexp, 0.5), 1.0)


def hold_after(x, i1):
    """Keep x constant beyond the last rain gate of each ray (PIA does not decrease)."""
    out = x.copy()
    rows = np.arange(x.shape[0])
    last = x[rows, np.clip(i1, 0, None)]
    j = np.arange(x.shape[1])[None, :]
    return np.where(j > i1[:, None], np.where(i1[:, None] >= 0, last[:, None], 0.0), out)


def apply(ds, pia, cfg, alpha, name, ah=None, extra=None):
    """Write corrected Z/ZDR (raw copies kept as *_UNCORR) and diagnostics.

    PIA is a property of the path and is kept everywhere, but only gates with real signal are
    corrected: adding PIA to receiver noise behind a storm would turn the noise into fake echo.
    """
    out = ds.copy()
    sig = signal_mask(ds)
    dims = ("azimuth", "range")
    gamma = cfg["zdr_beta"] / np.asarray(alpha, dtype=float)
    pida = pia * (gamma[:, None] if np.ndim(gamma) else gamma)
    out["DBZH_UNCORR"] = ds["DBZH"]
    out["DBZH"] = ds["DBZH"] + np.where(sig, pia, 0.0)
    out["DBZH"].attrs = {**ds["DBZH"].attrs, "radarwow_corrected": f"attenuation ({name})"}
    if "ZDR" in ds:
        out["ZDR_UNCORR"] = ds["ZDR"]
        out["ZDR"] = ds["ZDR"] + np.where(sig, pida, 0.0)
        out["ZDR"].attrs = {**ds["ZDR"].attrs, "radarwow_corrected": f"differential attenuation ({name}, beta/alpha)"}
    out["PIA"] = (dims, pia.astype("float32"), {"units": "dB", "long_name": f"two-way path-integrated attenuation ({name})"})
    out["PIDA"] = (dims, pida.astype("float32"), {"units": "dB", "long_name": "two-way path-integrated differential attenuation"})
    if ah is not None:
        out["AH"] = (dims, ah.astype("float32"), {"units": "dB/km", "long_name": "one-way specific attenuation",
                                                  "comment": "NaN off the rain path or where delta PHIDP < min_dphi"})
    for k, v in (extra or {}).items():
        out[k] = v
    return out


__all__ = ["apply", "gate_km", "hold_after", "path_dphi", "phidp_rec", "rain_path"]
