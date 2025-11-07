#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PBH beta/f ceilings with NG (p) and μ-distortion — CLI-friendly version

What it does:
  - Computes f_PBH,max and beta_max(M,p) given:
      (i) inventory caps on f_PBH (p–independent): SMBH census, lensing, custom
     (ii) μ-distortion cap (p–dependent) via σ(β,p) -> μ(σ^2, M)
  - Saves CSVs and a PNG plot to an output directory (default: ./outputs).

Quick use:
  python beta_f_pbh.py
  python beta_f_pbh.py --masses 1e4,1e5,1e7 --p-list 2,1,0.6 --outdir ./outputs

To plug YOUR NG mapping:
  - Set USE_EXTERNAL_SIGMA_MU=True below
  - Implement sigma_true_for_beta(beta,p) and mu_of_sigma2_true(sigma2,M)

Notes:
  - Uses your f–β conversion with prefactor 1.69e8 (RD scalings).
  - Default μ-limit is COBE/FIRAS: 9e-5 (change with --mu-limit).
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional, List
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 0) Master switches
# =========================

USE_EXTERNAL_SIGMA_MU = False   # True to use your own functions (define below)
USE_CENSUS_CAP        = True    # SMBH census cap (constant f_max)
USE_LENSING_CAP       = False   # flat lensing cap across wide mass range
USE_CUSTOM_CAP        = False   # define your own f_max(M) function below
USE_MU_CAP            = True    # apply μ-distortion cap (depends on p)

# =========================
# 1) Cosmology + formation constants
# =========================

gamma   = 0.2        # collapse efficiency
gstar   = 106.75     # relativistic d.o.f. at formation
h       = 0.674      # Planck-ish
OmegaDM = 0.26       # DM density parameter today

# Your f–beta relation (radiation era)
pref = 1.69e8 * (gstar/106.75)**(-1/4) * (gamma/0.2)**(1/2)  # dimensionless

def f_from_beta(beta: float, M_solar: float) -> float:
    """Present-day fraction f_PBH(M) from formation fraction beta for mass M (in Msun)."""
    return pref * beta * (M_solar)**(-0.5)

def beta_from_f(f: float, M_solar: float) -> float:
    """Invert: beta from f_PBH(M)."""
    return (f / pref) * math.sqrt(M_solar)

# =========================
# 2) f_PBH caps (p–independent)
# =========================

def rho_crit_Msun_per_Mpc3(h_val: float) -> float:
    """Critical density in Msun/Mpc^3."""
    return 2.775e11 * h_val**2

# Local SMBH mass density (adjustable)
rho_SMBH = 4.2e5  # Msun/Mpc^3 (Shankar+ '04 order-of-mag; set what you prefer)

def Omega_SMBH(h_val: float = h) -> float:
    return rho_SMBH / rho_crit_Msun_per_Mpc3(h_val)

def fmax_census(h_val: float = h, OmegaDM_val: float = OmegaDM) -> float:
    return Omega_SMBH(h_val) / OmegaDM_val

# Lensing cap (optional): constant fraction across a broad mass range
fmax_lensing_flat = 0.02  # ~2%

# Custom cap f_max(M): change as needed
def fmax_custom(M_solar: float) -> float:
    return 1e-8  # placeholder or inject an interpolator

def active_caps(M_solar: float) -> List[float]:
    caps = []
    if USE_CENSUS_CAP:
        caps.append(fmax_census())
    if USE_LENSING_CAP:
        caps.append(fmax_lensing_flat)
    if USE_CUSTOM_CAP:
        caps.append(fmax_custom(M_solar))
    return caps

def fmax_tight(M_solar: float) -> float:
    caps = active_caps(M_solar)
    return min(caps) if caps else math.inf

def beta_max_from_f(M_solar: float) -> float:
    fcap = fmax_tight(M_solar)
    return beta_from_f(fcap, M_solar)

# =========================
# 3) μ-distortion cap (depends on p via σ(β,p))
# =========================

# --- USER HOOKS ---
def sigma_true_for_beta(beta: float, p: float) -> float:
    """User-supplied σ(β,p). Replace with your mapping if USE_EXTERNAL_SIGMA_MU=True."""
    raise NotImplementedError("Define sigma_true_for_beta or set USE_EXTERNAL_SIGMA_MU=False.")

def mu_of_sigma2_true(sigma2: float, M_solar: float) -> float:
    """User-supplied μ(σ^2, M). Replace with your mapping if USE_EXTERNAL_SIGMA_MU=True."""
    raise NotImplementedError("Define mu_of_sigma2_true or set USE_EXTERNAL_SIGMA_MU=False.")

# --- TOY (fallback) model if you haven't plugged your own yet ---
zeta_crit = 1.0
tail_C    = math.e

def sigma_of_beta_toy(beta: float, p: float) -> float:
    """Toy inversion of p-family tail to get σ from β (captures scaling only)."""
    if beta <= 0:
        return float("nan")
    x = math.log(tail_C / beta)
    if x <= 0:
        return float("nan")
    return zeta_crit / (math.sqrt(2.0) * (x ** (1.0/p)))

def mu_window(M_solar: float) -> float:
    """Crude μ sensitivity window; replace/remove when you plug your model."""
    if 1e5 <= M_solar <= 1e8:
        return 1.0
    if 1e4 <= M_solar < 1e5 or 1e8 < M_solar <= 1e10:
        return 0.2
    return 0.0

def mu_toy(sigma2: float, M_solar: float) -> float:
    return 2.0 * mu_window(M_solar) * sigma2

def sigma_of_beta(beta: float, p: float) -> float:
    if USE_EXTERNAL_SIGMA_MU:
        return sigma_true_for_beta(beta, p)
    return sigma_of_beta_toy(beta, p)

def mu_of_sigma2(sigma2: float, M_solar: float) -> float:
    if USE_EXTERNAL_SIGMA_MU:
        return mu_of_sigma2_true(sigma2, M_solar)
    return mu_toy(sigma2, M_solar)

# μ-limit (default; can be overridden by CLI)
mu_limit = 9e-5   # COBE/FIRAS 95% CL

@dataclass
class MuSolveOptions:
    beta_min: float = 1e-40
    beta_max: float = 1e-5
    max_iter: int   = 60
    tol: float      = 1e-12

def beta_max_from_mu(M_solar: float, p: float, mu_lim: float, opts: Optional[MuSolveOptions] = None) -> float:
    """Largest β such that μ(σ^2(β,p), M) <= mu_lim, via bracketed bisection."""
    if not USE_MU_CAP:
        return math.inf
    if opts is None:
        opts = MuSolveOptions()

    beta_lo = opts.beta_min
    beta_hi = opts.beta_max

    def mu_at(beta):
        s = sigma_of_beta(beta, p)
        return mu_of_sigma2(s*s, M_solar)

    mu_lo = mu_at(beta_lo)
    mu_hi = mu_at(beta_hi)

    if not (np.isfinite(mu_lo) and np.isfinite(mu_hi)):
        return float("nan")

    # If even at high β we're under the μ-limit, μ isn't the active cap
    if mu_hi <= mu_lim:
        return beta_hi

    # If already over at tiny β_lo, μ is extremely restrictive
    if mu_lo > mu_lim:
        return beta_lo

    # Bisection on monotone μ(β)
    for _ in range(opts.max_iter):
        beta_mid = math.sqrt(beta_lo * beta_hi)  # multiplicative bisection
        mu_mid = mu_at(beta_mid)
        if not np.isfinite(mu_mid):
            return float("nan")

        if mu_mid <= mu_lim:
            beta_lo = beta_mid
        else:
            beta_hi = beta_mid

        if abs(beta_hi / beta_lo - 1.0) < 1e-8 or abs(mu_mid - mu_lim) < opts.tol:
            break

    return beta_lo

# =========================
# 4) Combine caps
# =========================

def beta_max_combined(M_solar: float, p: float, mu_lim: float) -> float:
    beta_f = beta_max_from_f(M_solar)
    beta_mu = beta_max_from_mu(M_solar, p, mu_lim) if USE_MU_CAP else math.inf
    return min(beta_f, beta_mu)

def f_max_combined(M_solar: float, p: float, mu_lim: float) -> float:
    return f_from_beta(beta_max_combined(M_solar, p, mu_lim), M_solar)

# =========================
# 5) CLI / main
# =========================

def parse_list_of_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="Compute PBH β_max and f_max with NG and μ caps.")
    parser.add_argument("--masses", default="1e4,1e5,1e7",
                        help="Comma-separated masses (Msun), e.g. '1e4,1e5,1e7'")
    parser.add_argument("--p-list", default="2,1,0.6",
                        help="Comma-separated p values, e.g. '2,1,0.6'")
    parser.add_argument("--mu-limit", type=float, default=mu_limit,
                        help="μ upper limit (default: 9e-5)")
    parser.add_argument("--outdir", default="./outputs",
                        help="Directory to save CSVs and plots (default: ./outputs)")
    args = parser.parse_args()

    mass_list = parse_list_of_floats(args.masses)
    p_list = parse_list_of_floats(args.p_list)
    mu_lim = args.mu_limit

    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inventory-only caps (p–independent)
    rows_fcaps = []
    for M in mass_list:
        rows_fcaps.append({
            "M [Msun]": M,
            "beta_max_from_f": beta_max_from_f(M),
            "f_at_beta_1e-20": f_from_beta(1e-20, M),
            "fmax_tight": fmax_tight(M)
        })
    df_fcaps = pd.DataFrame(rows_fcaps)
    df_fcaps_path = out_dir / "pbh_inventory_caps.csv"
    df_fcaps.to_csv(df_fcaps_path, index=False)

    # Caps including μ (p–dependent)
    rows_combined = []
    for M in mass_list:
        for pval in p_list:
            beta_mu = beta_max_from_mu(M, pval, mu_lim)
            beta_comb = beta_max_combined(M, pval, mu_lim)
            rows_combined.append({
                "M [Msun]": M,
                "p": pval,
                "beta_max_mu": beta_mu,
                "beta_max_combined": beta_comb,
                "f_max_combined": f_from_beta(beta_comb, M)
            })
    df_combined = pd.DataFrame(rows_combined)
    df_combined_path = out_dir / "pbh_caps_with_mu.csv"
    df_combined.to_csv(df_combined_path, index=False)

    # Plot β_max(M,p)
    plt.figure(figsize=(7,4.5))
    for pval in p_list:
        y = [beta_max_combined(M, pval, mu_lim) for M in mass_list]
        plt.loglog(mass_list, y, marker='o', label=f"p={pval}")
    plt.xlabel("M [Msun]")
    plt.ylabel(r"$\beta_{\max}(M,p)$")
    plt.title(r"PBH $\beta_{\max}$ from inventory + $\mu$")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plot_path = out_dir / "beta_max_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.show()
    # Console output
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")
    print("\nInventory-only caps (p-independent):")
    print(df_fcaps.to_string(index=False))
    print("\nCaps including μ (depends on p):")
    print(df_combined.to_string(index=False))

    print("\nSaved files:")
    print(" -", df_fcaps_path)
    print(" -", df_combined_path)
    print(" -", plot_path)

if __name__ == "__main__":
    main()
