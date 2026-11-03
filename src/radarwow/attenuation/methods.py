"""Three attenuation corrections, from simplest to most constrained.

LinearPhiAttenuation:       PIA(r) = alpha * PHIDP(r)            (A_H = alpha K_DP, e.g. Bringi et al. 1990)
Testud2000ZPHI:             A_H(r) from the Z profile, constrained so that 2 * integral(A_H) = alpha * delta PHIDP
Bringi2001SelfConsistent:   ZPHI for a range of alpha; per ray, keep the alpha whose implied PHIDP
                            (2 integral A_H / alpha) best matches the measured PHIDP.

ZPHI (Testud et al. 2000, Eq. 22; two-way PIA, Z in linear units, b from A = a Z^b):
    A_H(r) = Z(r)^b C / [ I(r0, r1) + C I(r, r1) ],   C = 10^(0.1 b alpha dPHI) - 1,
    I(r, r1) = 0.46 b * integral_r^r1 Z(s)^b ds.
Differential attenuation uses PIDA = (beta / alpha) PIA (band config ``zdr_beta``).
"""

from __future__ import annotations

import numpy as np

from radarwow.core import Algorithm, Reference, register

from .common import apply, gate_km, hold_after, path_dphi, phidp_rec, rain_path

DIMS = ("azimuth", "range")


def zphi(z_dbz, phi, valid, seg, i0, i1, alpha, b, dr_km, min_dphi=2.0):
    """ZPHI for every ray; ``alpha`` scalar or per ray.

    Returns (A_H one-way dB/km, two-way PIA dB, delta PHIDP). PIA uses the closed form of
    2 * integral(A_H), which is exact for the discrete Z profile (summing A_H gate by gate biases it):
        PIA(r) = (10 / b) log10[ (1 + C) I(r0) / (I(r0) + C I(r)) ],   so PIA(r1) = alpha dPHI.
    I(r, r1) uses the midpoint rule (half of gate r counted).
    """
    zb = np.where(valid & seg, 10.0 ** (0.1 * b * np.nan_to_num(z_dbz, nan=-50.0)), 0.0)
    i_r = 0.46 * b * dr_km * (np.flip(np.cumsum(np.flip(zb, 1), 1), 1) - 0.5 * zb)  # I(r, r1)
    rows = np.arange(zb.shape[0])
    i_r0 = i_r[rows, i0] + 0.46 * b * dr_km * 0.5 * zb[rows, i0]  # include the whole first gate
    dphi = np.clip(path_dphi(phi, i0, i1), 0.0, None)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), dphi.shape)
    c = 10.0 ** (0.1 * b * alpha * dphi) - 1.0
    ok = (dphi >= min_dphi)[:, None] & seg
    with np.errstate(invalid="ignore", divide="ignore"):
        den = i_r0[:, None] + c[:, None] * i_r
        a = np.where(ok, zb * c[:, None] / den, 0.0)
        pia = np.where(ok, (10.0 / b) * np.log10((1.0 + c[:, None]) * i_r0[:, None] / den), 0.0)
    a = np.where(np.isfinite(a), a, 0.0)
    pia = hold_after(np.maximum.accumulate(np.where(np.isfinite(pia), pia, 0.0), axis=1), i1)
    return a, pia, dphi


_TESTUD = Reference("J. Testud, E. Le Bouar, E. Obligis, M. Ali-Mehenni", 2000,
                    "The Rain Profiling Algorithm Applied to Polarimetric Weather Radar",
                    "J. Atmos. Oceanic Technol. 17, 332-356", "10.1175/1520-0426(2000)017<0332:TRPAAT>2.0.CO;2")
_BRINGI = Reference("V. N. Bringi, T. D. Keenan, V. Chandrasekar", 2001,
                    "Correcting C-band radar reflectivity and differential reflectivity data for rain attenuation: "
                    "a self-consistent method with constraints", "IEEE Trans. Geosci. Remote Sens. 39, 1906-1915", "10.1109/36.951081")


@register
class LinearPhiAttenuation(Algorithm):
    name = "linear_phi_attenuation"
    stage = "attenuation"
    reference = _BRINGI
    requires = ("DBZH", "PHIDP")
    optional = ("ZDR", "RHOHV", "PHIDP_REC", "height_over_iso0")
    provides = ("DBZH", "ZDR", "PIA", "PIDA")
    notes = "PIA = alpha * PHIDP along the rain path; alpha and beta from the band config. Baseline, no Z information."

    def run_sweep(self, ds, cfg):
        valid, seg, i0, i1 = rain_path(ds, cfg)
        phi = phidp_rec(ds, cfg)
        rows = np.arange(phi.shape[0])
        rel = np.where(seg, np.nan_to_num(phi - phi[rows, i0][:, None]), 0.0)
        pia = hold_after(np.maximum.accumulate(np.clip(cfg["zphi_alpha"] * rel, 0, None), axis=1), i1)
        return apply(ds, pia, cfg, cfg["zphi_alpha"], "linear PHIDP")


@register
class Testud2000ZPHI(Algorithm):
    name = "testud2000_zphi"
    stage = "attenuation"
    reference = _TESTUD
    requires = ("DBZH", "PHIDP")
    optional = ("ZDR", "RHOHV", "PHIDP_REC", "height_over_iso0")
    provides = ("DBZH", "ZDR", "PIA", "PIDA", "AH")
    notes = "Fixed alpha from the band config; A_H follows the Z profile, total PIA constrained by delta PHIDP."
    params = dict(min_dphi=2.0)

    def run_sweep(self, ds, cfg):
        valid, seg, i0, i1 = rain_path(ds, cfg)
        phi = phidp_rec(ds, cfg)
        dr = gate_km(ds)
        a, pia, _ = zphi(ds["DBZH"].values, phi, valid, seg, i0, i1, cfg["zphi_alpha"], cfg["zphi_b"], dr, self.params["min_dphi"])
        return apply(ds, pia, cfg, cfg["zphi_alpha"], "ZPHI", ah=a)


@register
class Bringi2001SelfConsistent(Algorithm):
    name = "bringi2001_self_consistent"
    stage = "attenuation"
    reference = _BRINGI
    requires = ("DBZH", "PHIDP")
    optional = ("ZDR", "RHOHV", "PHIDP_REC", "height_over_iso0")
    provides = ("DBZH", "ZDR", "PIA", "PIDA", "AH", "ALPHA", "ALPHA_FITTED")
    notes = ("Per-ray alpha from the band's zphi_alpha_range minimising |PHIDP_ZPHI - PHIDP| over the rain path. "
             "alpha is only identifiable when the path attenuates noticeably (otherwise PHIDP_ZPHI hardly depends on "
             "alpha), so a ray keeps the band default unless default_alpha * dPHI >= min_pia_fit dB, the best alpha is "
             "inside the search range, and it lowers the misfit by >= min_gain versus the default (ALPHA_FITTED). "
             "The paper's ZDR constraint for beta is not implemented (beta/alpha from the band config).")
    params = dict(n_alpha=25, min_dphi=2.0, min_pia_fit=3.0, min_gain=0.1)

    def run_sweep(self, ds, cfg):
        valid, seg, i0, i1 = rain_path(ds, cfg)
        phi = phidp_rec(ds, cfg)
        dr = gate_km(ds)
        z = ds["DBZH"].values
        rows = np.arange(phi.shape[0])
        rel = np.where(seg, np.nan_to_num(phi - phi[rows, i0][:, None]), 0.0)
        lo, hi = cfg["zphi_alpha_range"]
        alphas = np.linspace(lo, hi, self.params["n_alpha"])
        err = np.empty((alphas.size, phi.shape[0]))
        for k, al in enumerate(alphas):
            _, pia, dphi = zphi(z, phi, valid, seg, i0, i1, al, cfg["zphi_b"], dr, self.params["min_dphi"])
            phi_c = pia / al
            err[k] = np.sum(np.abs(phi_c - rel) * (valid & seg), axis=1)
        kbest = np.argmin(err, axis=0)
        best = alphas[kbest]
        _, pia_d, _ = zphi(z, phi, valid, seg, i0, i1, cfg["zphi_alpha"], cfg["zphi_b"], dr, self.params["min_dphi"])
        err_d = np.sum(np.abs(pia_d / cfg["zphi_alpha"] - rel) * (valid & seg), axis=1)
        rows_ = np.arange(err.shape[1])
        gain = 1.0 - err[kbest, rows_] / np.maximum(err_d, 1e-9)
        fitted = ((cfg["zphi_alpha"] * dphi >= self.params["min_pia_fit"]) & (kbest > 0) & (kbest < alphas.size - 1)
                  & (gain >= self.params["min_gain"]))
        best = np.where(fitted, best, cfg["zphi_alpha"])
        a, pia, _ = zphi(z, phi, valid, seg, i0, i1, best, cfg["zphi_b"], dr, self.params["min_dphi"])
        extra = {"ALPHA": (("azimuth",), best.astype("float32"), {"units": "dB/deg", "long_name": "per-ray A/KDP ratio"}),
                 "ALPHA_FITTED": (("azimuth",), fitted, {"long_name": "alpha fitted (True) or band default (False)"})}
        return apply(ds, pia, cfg, best, "Bringi 2001 self-consistent", ah=a, extra=extra)
