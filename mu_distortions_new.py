"""
Toy PDF + upward-step (Cai et al.) models for PBH abundance and μ-distortions.

This script:
  1) Defines mappings between PBH mass scale and μ-distortion response.
  2) Implements a generalized-normal ("p-PDF") toy model for ζ.
  3) Implements Cai et al.'s upward-step non-Gaussian model.
  4) Produces several comparison plots (toggle with RUN_* flags below).

Notes for readers:
  - Most "physics logic" lives in the function blocks.
  - The plot blocks at the bottom are intentionally verbose but guarded by flags.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import gamma


def configure_plot_style(use_tex=True):
    """Centralize plotting style so the rest of the file is easier to read."""
    # If you want to disable LaTeX (e.g., on a machine without TeX),
    # set use_tex=False below or override before calling this function.
    plt.rc("text", usetex=use_tex)
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}"
        r"\usepackage{amssymb}"
        r"\usepackage{xcolor}"
        r"\boldmath"
    )

    # Global figure defaults.
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["font.size"] = 11

    # Axes + ticks styling.
    plt.rcParams["axes.labelsize"] = plt.rcParams["font.size"]
    plt.rcParams["axes.titlesize"] = 1.4 * plt.rcParams["font.size"]
    plt.rcParams["xtick.labelsize"] = 1.4 * plt.rcParams["font.size"]
    plt.rcParams["ytick.labelsize"] = 1.4 * plt.rcParams["font.size"]
    plt.rcParams["axes.linewidth"] = 1

    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1

    # Ticks on left/bottom only (avoids implicit gca() calls at import time).
    plt.rcParams["xtick.top"] = False
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.right"] = False


configure_plot_style(use_tex=True)





# ---- Mappings ----

def k_from_mass(M_sun):
    """Map PBH mass (M_sun) to the characteristic wavenumber k [Mpc^-1]."""
    return 92.0 * np.sqrt(5e8 / M_sun)

def window_factor(k):
    """Silk damping window factor W(k) entering the μ response."""
    k = np.asarray(k, float)
    a = np.exp(-k/5400.0)
    b = k/31.6
    return a - np.where(b > 50.0, 0.0, np.exp(-b*b))  # avoid underflow noise

def mu_of_sigma2_true(s2_true, M_sun):
    """μ-distortion amplitude for a given true variance and mass scale."""
    return 2.2 * np.asarray(s2_true, float) * window_factor(k_from_mass(M_sun))

# Variance map: σ_true^2 = C(p) * σ̃^2
def C_of_p(p):
    """Variance conversion factor for the generalized-normal p-PDF."""
    return (2.0 * gamma(1.0 + 3.0/p)) / (3.0 * gamma(1.0 + 1.0/p))

# ---- Settings you can change ----
MU_LIMIT = 9e-5
mass_list = [1e4, 1e5, 1e7]          # representative masses (M_sun)
p_values  = [0.6]                    # tail index for toy p-PDF examples
s2tilde_grid = np.logspace(-7, -1, 400)  # x-axis = σ̃^2 (NCS parameter)

# ---- Plot all 8 curves ----
# plt.figure()
# for i, M in enumerate(mass_list):
#     color = plt.cm.tab10(i % 10)
#     for p in p_values:
#         s2_true = C_of_p(p) * s2tilde_grid
#         mu_line = mu_of_sigma2_true(s2_true, M)
#         style = '-' if abs(p - 2.0) < 1e-12 else '--'
#         plt.plot(s2tilde_grid, mu_line, style, color=color, lw=2,
#                  label=f"M={M:.0e} Msun, p={p:g}")



# === Additions: β(σ,p) and σ(β,p) using the toy p-PDF  ===

def sigma_tilde_from_sigma_true(sigma_true, p):
    """
   variance map:
    σ_true^2 = [2 Γ(1+3/p)]/[3 Γ(1+1/p)] * σ_tilde^2  ->  σ_tilde(σ_true, p)
    """
    num = 3.0 * gamma(1.0 + 1.0/p)
    den = 2.0 * gamma(1.0 + 3.0/p)
    return sigma_true * np.sqrt(num/den)

def beta_from_sigma_true(sigma_true, p, zeta_c=0.67, n=12000):
    """
    β(σ_true, p) = (1 / [2 Γ(1+1/p)]) ∫_{u_c}^∞ exp(-u^p) du,
    with u = ζ / (√2 σ_tilde), u_c = ζ_c / (√2 σ_tilde).
    Simple trapezoid integral to avoid extra packages.
    """
    sig_t = sigma_tilde_from_sigma_true(sigma_true, p)
    u_c = zeta_c / (np.sqrt(2.0) * sig_t)
    u_max = np.maximum(u_c + 60.0, 60.0)   # safe tail cutoff
    u = np.linspace(u_c, u_max, n)
    f = np.exp(-(u**p))
    integral = np.trapz(f, u)
    return (1.0 / (2.0 * gamma(1.0 + 1.0/p))) * integral

def sigma_true_for_beta(beta_target, p, zeta_c=0.67,
                        s_lo=1e-10, s_hi=1.0, iters=80):
    """
    Invert β(σ,p)=β_target with bisection (β ↑ with σ).
    """
    def f(s): return beta_from_sigma_true(s, p, zeta_c)
    # ensure the bracket
    for _ in range(60):
        if f(s_hi) < beta_target: s_hi *= 2.0
        else: break
    for _ in range(60):
        if f(s_lo) > beta_target: s_lo *= 0.5
        else: break
    # bisection
    for _ in range(iters):
        mid = 0.5*(s_lo + s_hi)
        if f(mid) >= beta_target: s_hi = mid
        else: s_lo = mid
    return 0.5*(s_lo + s_hi)

###############################




# === Settings for ζ p-type vs Cai upward-step comparison ===
# These are used both in the p-type μ–β plots and in the new Cai μ–β plots.

M_cmp         = 1e4                     # comparison mass scale [M_sun]
p_list_small  = [0.8, 1.2, 2.0]         # small set of p values for ζ p-type curves
beta_grid     = np.logspace(-28, -6, 24)  # β grid for μ(β) plots

def mu_from_beta_zeta(beta_val, p):
    """
    μ(β; p) for the ζ p-type toy PDF at the comparison mass M_cmp.
    Used to overlay ζ p-type and Cai upward-step curves on the same μ–β plot.
    """
    sigma_true = sigma_true_for_beta(beta_val, p)   # invert β → σ_true for ζ
    return mu_of_sigma2_true(sigma_true**2, M_cmp)  # μ(σ_true^2, M_cmp)






# --- Compaction-function PBH abundance with toy p-PDF ---
# ---------- Windows & kernels ----------
def W_TH(x):
    # Spherical top-hat window in k-space; safe for vector x>0
    return 3.0*(np.sin(x) - x*np.cos(x))/np.where(x==0, 1.0, x**3)
def dj0_dz(z):
    z = np.asarray(z, dtype=float)
    # derivative of j0(z) = sin z / z
    num = np.cos(z)*z - np.sin(z)
    den = np.where(z == 0.0, 1.0, z**2)  # safe at z=0
    return num / den


# ---------- Power -> variance ----------
# Σ_XX = ∫ d ln k (k r)^2 [j0'(k r)]^2 P_R(k) W_TH^2(k r)
def sigma2_X_from_PR(P_R, r, kmin=1e-3, kmax=1e4, n=2000):
    ks  = np.logspace(np.log10(kmin), np.log10(kmax), n)
    kr  = ks*r
    integrand = (kr**2) * (dj0_dz(kr)**2) * P_R(ks) * (W_TH(kr)**2)
    return np.trapz(integrand, np.log(ks))

# Var(C_ℓ) with C_ℓ = -(4/3) X  (for J=1)
def sigma2_Cl_from_PR(P_R, r, **kw):
    return (16.0/9.0) * sigma2_X_from_PR(P_R, r, **kw)

# Nonlinear threshold -> linear threshold (type-I)
def Cl_threshold_from_Cth(C_th):
    return (4.0/3.0) * (1.0 - np.sqrt(1.0 - 1.5*C_th))

# ---------- p-PDF on C_ℓ (generalized normal) ----------
# Var relation: sigma2_Cl = [2 Γ(1+3/p) / (3 Γ(1+1/p))] * sigma0^2
def _sigma0_from_sigma2(sigma2_Cl, p):
    C = 2.0*gamma(1.0 + 3.0/p) / (3.0*gamma(1.0 + 1.0/p))
    return np.abs(np.sqrt(sigma2_Cl / C))

def ppdf_Cl(u, p, sigma2_Cl):
    sigma0 = _sigma0_from_sigma2(sigma2_Cl, p)
    a = np.sqrt(2.0)*sigma0
    norm = 1.0/(2.0*a*gamma(1.0 + 1.0/p))
    return norm * np.exp(- (np.abs(u)/a)**p)

# ---------- Cheap & accurate quadrature (no scipy) ----------
_LEG_CACHE = {}
def _leg_nodes(n):
    if n not in _LEG_CACHE:
        _LEG_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _LEG_CACHE[n]

def quad_legendre(f, a, b, n=128):
    x, w = _leg_nodes(n)           # cached
    t = 0.5*(b - a); y = t*x + 0.5*(b + a)
    return float(t * np.sum(w * f(y)))

# ---------- Main: β from σ^2_{C_ℓ} with optional mass weighting ----------
def beta_from_sigma2_Cl(sigma2_Cl, *,
                        p=2.0,
                        C_th=0.587,         # nonlinear compaction threshold
                        weighted=False,     # include M/M_H weighting?
                        K=4.0, gamma_c=0.38,# critical-collapse params
                        quad_n=256):
    Cl_th  = Cl_threshold_from_Cth(C_th)
    Cl_max = 4.0/3.0

    def pdf(u):
        return ppdf_Cl(u, p, sigma2_Cl)

    if not weighted:
        return quad_legendre(pdf, Cl_th, Cl_max, n=quad_n)

    # weighted integral: ∫ P(C_ℓ) [M(C_ℓ)/M_H] dC_ℓ, with C = C_ℓ - 3 C_ℓ^2/8
    def Cnl(u):  return u - 3.0*u*u/8.0
    def Mfrac(u):
        c = Cnl(u)
        out = np.zeros_like(u, dtype=float)
        mask = (c > C_th)
        out[mask] = K * (c[mask] - C_th)**gamma_c
        return out

    return quad_legendre(lambda u: pdf(u)*Mfrac(u), Cl_th, Cl_max, n=quad_n)

# ---------- Convenience: narrow log-normal "δ-peak" spectrum ----------

def sigma2_Cl_dirac_like(kstar, amp=0.01, r_m=None, sigma_ln=0.02):
    """
    Narrow log-normal P_R centered at kstar with width sigma_ln.
    Use dynamic integration bounds that always include the peak.
    """
    if r_m is None:
        r_m = 2.74 / kstar   # compaction-peak radius for type-I
    # cover ±10σ in ln k; this is plenty for sigma_ln~0.02
    kmin = kstar * np.exp(-10.0 * sigma_ln)
    kmax = kstar * np.exp(+10.0 * sigma_ln)

    def P_R(k):
        ln = np.log(k / kstar)
        return amp * np.exp(-(ln*ln) / (2.0 * sigma_ln * sigma_ln)) / (np.sqrt(2.0*np.pi) * sigma_ln)

    return sigma2_Cl_from_PR(P_R, r_m, kmin=kmin, kmax=kmax, n=1200)


# ##############################

# # === Compaction Block: μ vs β (single mass, p-zoo) ============================
# # Coherent mapping: β  --invert-->  σ^2_{Cℓ}  --(shape-fixed)-->  amp  -->  σ^2_true(ζ) = amp
# # and finally μ = 2.2 σ^2_true × window_factor(k_from_mass(M))

# # ---- Inverter: σ^2_{Cℓ} for a target β (toy p-PDF on C_ℓ) -------------------
# def sigma2_Cl_for_beta(beta_target, p, *, C_th=0.587, weighted=False, K=4.0, gamma_c=0.38,
#                        s2_lo=1e-14, s2_hi=1.0, iters=80, quad_n=256):
#     """Solve β(σ^2_{Cℓ}) = beta_target by bisection (β increases with σ^2_{Cℓ})."""
#     f = lambda s2: beta_from_sigma2_Cl(s2, p=p, C_th=C_th,
#                                        weighted=weighted, K=K, gamma_c=gamma_c, quad_n=quad_n)
#     # grow upper bound until it brackets
#     for _ in range(80):
#         if f(s2_hi) < beta_target: s2_hi *= 2.0
#         else: break
#     # shrink lower bound if needed
#     for _ in range(80):
#         if f(s2_lo) > beta_target: s2_lo *= 0.5
#         else: break
#     # bisection
#     for _ in range(iters):
#         mid = 0.5*(s2_lo + s2_hi)
#         if f(mid) >= beta_target: s2_hi = mid
#         else: s2_lo = mid
#     return 0.5*(s2_lo + s2_hi)

# # ---- Plot: μ vs β at one mass, multiple p (compaction) -----------------------
# # ----------------- Simplified μ–β comparison (ζ vs compaction) -----------------
# M_cmp         = 1e4
# p_list_small  = [0.8, 1.2, 2.0]
# beta_grid     = np.logspace(-28, -6, 24)    # modest range & density
# sigma_ln_shape= 0.02
# USE_WEIGHTED  = False
# Cth_nonlinear = 0.587
# quad_n_comp   = 128

# # Precompute compaction "gain": σ^2_{Cℓ} per unit amp
# kstar         = k_from_mass(M_cmp)
# s2Cl_per_amp1 = sigma2_Cl_dirac_like(kstar, amp=1.0, sigma_ln=sigma_ln_shape)

# def sigma2_Cl_for_beta(beta_target, p, *, C_th=Cth_nonlinear,
#                        weighted=USE_WEIGHTED, K=4.0, gamma_c=0.38,
#                        s2_lo=1e-14, s2_hi=1.0, iters=70, quad_n=quad_n_comp):
#     f = lambda s2: beta_from_sigma2_Cl(s2, p=p, C_th=C_th,
#                                        weighted=weighted, K=K, gamma_c=gamma_c, quad_n=quad_n)
#     # bracket
#     for _ in range(60):
#         if f(s2_hi) < beta_target: s2_hi *= 2.0
#         else: break
#     for _ in range(60):
#         if f(s2_lo) > beta_target: s2_lo *= 0.5
#         else: break
#     # bisection
#     for _ in range(iters):
#         mid = 0.5*(s2_lo + s2_hi)
#         if f(mid) >= beta_target: s2_hi = mid
#         else: s2_lo = mid
#     return 0.5*(s2_lo + s2_hi)

# def mu_from_beta_comp(beta_val, p):
#     s2Cl = sigma2_Cl_for_beta(beta_val, p)
#     amp  = s2Cl / s2Cl_per_amp1          # fixed spectral shape ⇒ amplitude
#     s2_true = amp                        # for our normalized log-δ, ζ-variance = amp
#     return mu_of_sigma2_true(s2_true, M_cmp)

# def mu_from_beta_zeta(beta_val, p):
#     s_true = sigma_true_for_beta(beta_val, p)  # your ζ inverter
#     return mu_of_sigma2_true(s_true**2, M_cmp)

# # Compute both sets
# curves_cmp = {}
# curves_zet = {}
# for p in p_list_small:
#     mu_cmp = [mu_from_beta_comp(b, p)  for b in beta_grid]
#     mu_zet = [mu_from_beta_zeta(b, p)  for b in beta_grid]
#     curves_cmp[p] = np.array(mu_cmp)
#     curves_zet[p] = np.array(mu_zet)

# # Plot
# fig, ax = plt.subplots()
# colors = plt.cm.tab10(np.linspace(0, 1, len(p_list_small)))

# y_all = []
# for c, p in zip(colors, p_list_small):
#     ax.plot(beta_grid, curves_cmp[p], '-',  lw=2.7, color=c, label=rf"Compaction, $p={p:g}$")
#     ax.plot(beta_grid, curves_zet[p], '--', lw=2.4, color=c, label=rf"$\zeta$, $p={p:g}$")
#     y_all += curves_cmp[p].tolist() + curves_zet[p].tolist()

# ax.axhline(MU_LIMIT, ls="--", c="red", lw=2.0, label=r"$\mu_{\rm max}$")
# ax.set_xscale('log'); ax.set_yscale('log')
# # auto y-limits to ensure visibility (ignore zeros/negatives)
# y_pos = np.array([y for y in y_all if y > 0])
# if y_pos.size:
#     ax.set_ylim(0.6*y_pos.min(), 1.8*y_pos.max())
# ax.set_xlim(beta_grid.min(), beta_grid.max())

# ax.set_xlabel(r"PBH mass fraction at formation $\beta$", fontsize=20)
# ax.set_ylabel(r"Spectral distortion $\mu$", fontsize=20)
# ax.grid(True, ls='--', alpha=0.5)
# ax.legend(ncol=2, fontsize=12)
# ax.set_title(rf"$M=10^{{{int(np.log10(M_cmp))}}}\,M_\odot$: $\mu(\beta)$, $\zeta$ vs compaction", fontsize=15)

# fig.tight_layout()
# fig.savefig(f"mu_vs_beta_compare_M{int(M_cmp):d}.pdf", bbox_inches="tight")
# fig.savefig(f"mu_vs_beta_compare_M{int(M_cmp):d}.png", dpi=300, bbox_inches="tight")
# plt.show()
# -------------------------------------------------------------------------------

# ==============================================================================

# (Optional) To visually compare with your ζ p-zoo on the same axes, run your
# existing ζ block and then call:
#   axC.plot(beta_grid, mu_vals_zeta, '--', color=cmap(norm(p)), lw=2)
# using the same p list and beta_grid.



##############################

# def _gauss_cdf(y, sigma):
#     # local, so we don't change your imports
#     from math import erf
#     return 0.5*(1.0 + erf(y/(np.sqrt(2.0)*sigma)))

# def beta_from_sigma_fNL(sigma_g, fNL, zeta_c=0.67):
#     """
#     Press–Schechter tail for quadratic local NG:
#     ζ(y) = y + a (y^2 - σ_g^2), where y ≡ ζ_g ~ N(0,σ_g^2), a = (3/5) fNL.
#     β = Prob[ζ ≥ ζ_c] = ∫ 1_{ζ(y)≥ζ_c} N(y;0,σ_g^2) dy
#     Closed-form via roots of a y^2 + y - (a σ_g^2 + ζ_c) = 0.
#     """
#     a = 0.6 * fNL  # (3/5) fNL
#     if abs(a) < 1e-14:
#         # Gaussian limit
#         from math import erfc
#         return 0.5*erfc(zeta_c/(np.sqrt(2.0)*sigma_g))

#     # Quadratic roots where ζ(y) = ζ_c
#     D = 1.0 + 4.0*a*(a*sigma_g**2 + zeta_c)
#     if D < 0:
#         # No real roots: decide by vertex value
#         yv = -1.0/(2.0*a)
#         zeta_v = yv + a*(yv**2 - sigma_g**2)
#         if a > 0:
#             return 1.0 if zeta_v >= zeta_c else 0.0
#         else:
#             # for a<0 and D<0, vertex < ζ_c ⇒ always below threshold
#             return 0.0

#     r1 = (-1.0 - np.sqrt(D)) / (2.0*a)
#     r2 = (-1.0 + np.sqrt(D)) / (2.0*a)
#     r1, r2 = (min(r1, r2), max(r1, r2))

#     if a > 0:
#         # ζ≥ζc for y ≤ r1 or y ≥ r2  ⇒ β = 1 - [Φ(r2) - Φ(r1)]
#         return 1.0 - (_gauss_cdf(r2, sigma_g) - _gauss_cdf(r1, sigma_g))
#     else:
#         # a<0: ζ≥ζc for r1 ≤ y ≤ r2  ⇒ β = Φ(r2) - Φ(r1)
#         return _gauss_cdf(r2, sigma_g) - _gauss_cdf(r1, sigma_g)

# def sigma_g_for_beta_fNL(beta_target, fNL, zeta_c=0.67, s_lo=1e-8, s_hi=1.0, iters=80):
#     """Invert β(σ_g,fNL)=β_target via bisection (β increases with σ_g)."""
#     f = lambda s: beta_from_sigma_fNL(s, fNL, zeta_c)
#     # expand bracket if needed
#     for _ in range(80):
#         if f(s_hi) < beta_target: s_hi *= 2.0
#         else: break
#     for _ in range(80):
#         if f(s_lo) > beta_target: s_lo *= 0.5
#         else: break
#     # bisection
#     for _ in range(iters):
#         mid = 0.5*(s_lo + s_hi)
#         if f(mid) >= beta_target: s_hi = mid
#         else: s_lo = mid
#     return 0.5*(s_lo + s_hi)

# def mu_from_fNL_at_beta(M_sun, fNL, beta_target, zeta_c=0.67):
#     """
#     Given (M, fNL, β): solve σ_g, compute true variance
#       σ^2 = σ_g^2 + (18/25) fNL^2 σ_g^4  (to O(fNL^2)),
#     then μ = 2.2 σ^2 W(k*).
#     """
#     sigma_g = sigma_g_for_beta_fNL(beta_target, fNL, zeta_c=zeta_c)
#     sigma2_true = sigma_g**2 + (18.0/25.0)*(fNL**2)*(sigma_g**4)
#     return mu_of_sigma2_true(sigma2_true, M_sun)





#PLOTS


############################################################################################################

# # === Plot A: μ vs σ̃^2 at fixed masses for p in p_values (LaTeX-safe) ===
# s2tilde_grid = np.logspace(-7, -1, 400)   # x-axis = NCS variance parameter

# fig, ax = plt.subplots()
# for i, M in enumerate(mass_list):
#     color = plt.cm.tab10(i % 10)
#     for p in p_values:
#         # σ_true^2 = C(p) * σ̃^2  ⇒  μ(σ̃^2; M, p)
#         s2_true = C_of_p(p) * s2tilde_grid
#         mu_line = mu_of_sigma2_true(s2_true, M)

#         style = '-' if abs(p - 2.0) < 1e-12 else '--'  # solid for p=2, dashed otherwise
#         ax.plot(s2tilde_grid, mu_line, style, color=color, lw=2,
#                 label=rf"$M={M:.0e}\,M_\odot,\ p={p:g}$")

# # guides
# ax.axhline(MU_LIMIT, ls=":", c="k", label=r"$|\mu|$ limit")

# # axes/labels
# ax.set_xscale("log")
# ax.set_yscale("log")  # if you prefer linear y, comment this line out
# # ax.set_ylim(6e-5, 1.5e-4)  # optional fixed y-range
# ax.set_xlim(1e-7, 1e-1)
# ax.set_xlabel(r"Variance $\sigma^2$", fontsize=20)
# ax.set_ylabel(r"Spectral distortion $\mu$", fontsize=20)
# ax.legend(ncol=2, fontsize=9)
# ax.grid(True, ls="--", alpha=0.5)
# fig.tight_layout()

# # Saving the plot
# fig.savefig("mu_vs_sigmatilde2.pdf", bbox_inches="tight")
# fig.savefig("mu_vs_sigmatilde2.png", dpi=300, bbox_inches="tight")

# plt.show()





#####################################################################################################################


# # === Plot B: β vs true variance σ^2 for your p_values ===


# s2_true_grid = np.logspace(-10, -1, 400)



# # === Active version using fig/ax and LaTeX-bold strings ===
# figA, axA = plt.subplots(figsize=(10,7))

# # for p in p_values:
# #     beta_vals = [beta_from_sigma_true(np.sqrt(s2), p) for s2 in s2_true_grid]
# #     axA.plot(s2_true_grid, beta_vals, lw=2.5, label=rf"$p={p:g}$")


# # axA.text(7e-7, 1.44e-259,
# #          r"$\beta = \int_{0.67}^{\infty} \frac{1}{2\sqrt{2}\,\sigma_g\, \Gamma \left( 1 + \frac{1}{p} \right)} \exp \left[- \left(\frac{\zeta}{\sqrt{2}\,\sigma_g}\right)^p\right]\,{\rm d}\zeta$",
# #          fontsize=18, fontweight='bold', color='brown',
# #          va='center', ha='center', rotation='horizontal',
# #          bbox={'facecolor': 'gray', 'alpha': 0.12, 'pad': 4.5},
# #          zorder=100, clip_on=False)



# for p in p_values:
#     beta_vals = [beta_from_sigma_true(np.sqrt(s2), p) for s2 in s2_true_grid]
#     axA.plot(s2_true_grid, beta_vals, lw=2.5, label=rf"$p={p:g}$")


# # plt.axvline(x=1.364e-4, color='gray', linestyle='--', lw=2)
# # plt.axvline(x=3.41e-5, color='gray', linestyle='--', lw=2)
# xv = 1.84e-3
# y_target = 2.4e-55

# # get bottom of current axis (data coordinates)
# ymin, ymax = axA.get_ylim()
    
# # ensure the target is within the axis range (optional)
# # if needed, set limits: ax.set_ylim(min(ymin, y_target*0.9), ymax)

# # draw vertical line from bottom to target y
# axA.plot([1.84e-3, 1.83e-3], [ymin, 2.4e-55], color='dimgrey', linestyle=':', lw=2)
# axA.plot([4.616e-4, 4.616e-4], [ymin, 7.42e-22], color='dimgrey', linestyle=':', lw=2)
# # axA.plot([1.364e-4, 1.364e-4], [ymin, 7.42e-22], color='dimgrey', linestyle=':', lw=2)
# axA.plot([3.41e-5, 3.41e-5], [ymin, 1e-72], color='dimgrey', linestyle=':', lw=2)
# axA.plot([8.678e-7, 8.678e-7], [ymin, 1e-60], color='dimgrey', linestyle=':', lw=2)
# # plt.axvline(x=1.1536e-5, color='gray', linestyle='--', lw=2)
# axA.plot([1.1536e-5, 1.1536e-5], [ymin, 3.4e-122], color='dimgrey', linestyle=':', lw=2)
# axA.plot([2.566e-6, 2.566e-6], [ymin, 3.4e-43], color='dimgrey', linestyle=':', lw=2)


# axA.plot(1.84e-3, 2.4e-55, marker='o', markersize=10,
#          markerfacecolor='magenta', markeredgecolor='dimgrey', markeredgewidth=1.2, zorder=10)
# axA.plot(4.616e-4, 7.42e-22, marker='o', markersize=10,
#          markerfacecolor='magenta', markeredgecolor='dimgrey', markeredgewidth=1.2, zorder=10)
# axA.plot(3.4e-5, 1e-72, marker='o', markersize=10,
#          markerfacecolor='purple', markeredgecolor='dimgrey', markeredgewidth=1.2, zorder=10)
# axA.plot(2.566e-6, 3.4e-43, marker='o', markersize=10,
#          markerfacecolor='purple', markeredgecolor='dimgrey', markeredgewidth=1.2, zorder=10)
# axA.plot(1.1536e-5, 3.4e-122, marker='o', markersize=10,
#          markerfacecolor='teal', markeredgecolor='dimgrey', markeredgewidth=1.2, zorder=10)
# axA.plot(2.566e-6, 3.4e-43, marker='o', markersize=10,
#          markerfacecolor='purple', markeredgecolor='dimgrey', markeredgewidth=1.2, zorder=10)
# axA.plot(8.67e-7, 7.9e-60, marker='o', markersize=10,
#          markerfacecolor='teal', markeredgecolor='dimgrey', markeredgewidth=1.2, zorder=10)


# axA.set_ylim(1e-320, 1e8)
# axA.set_yscale("log")
# axA.set_xscale("log"); axA.set_yscale("log")
# axA.set_xlabel(r"Variance  $\sigma^2$ ", fontsize=20)
# axA.set_ylabel(r"PBH mass fraction at formation  $\beta$", fontsize=20)
# # axA.set_title(r"$\beta$ vs $\sigma^2$  (NCS $p$-PDF, $\zeta_c=0.67$)")
# axA.set_xlim(8e-10, 1e-1)








# # axA.legend(loc='lower left', fontsize=15, frameon=False, handlelength=1.5,
# #         handletextpad=0.5, borderpad=0.5, labelspacing=0.5, facecolor='brown', edgecolor='brown',alpha=0.1)
# axA.grid(True, ls="--", alpha=0.7)


# #Annotations for Plot B

# # plt.text(7e-7, 1.44e-259, r"$\beta = \int_{0.67}^{\infty} \frac{1}{2\sqrt{2}\,\sigma_g\, \Gamma \left( 1 + \frac{1}{p} \right)} \exp \left[- \left(\frac{\zeta}{\sqrt{2}\,\sigma_g}\right)^p\right]\,{\rm d}\zeta$", fontsize=18, fontweight='bold', color='brown',
# #         verticalalignment='center', horizontalalignment='center', rotation='horizontal',
# #         bbox={'facecolor': 'gray', 'alpha': 0.12, 'pad': 4.5})



# plt.text(
#     1.2e-6, 1.44e-289,
#     r"$\boldsymbol{\beta = \int_{0.67}^{\infty} \frac{1}{2\sqrt{2}\,\sigma_g\, \Gamma \left( 1 + \frac{1}{p} \right)} \exp \left[- \left(\frac{\zeta}{\sqrt{2}\,\sigma_g}\right)^p\right]\,{\rm d}\zeta}$",
#     fontsize=20, fontweight='bold', color='brown',
#     verticalalignment='center', horizontalalignment='center', rotation='horizontal',
#     bbox={'facecolor': 'lightgray', 'alpha': 0.8, 'pad': 7.5}
# )


# plt.text(3.5e-8, 6.2e-43, r"\textbf{``o''}$\rightarrow \sigma^2_{\rm{max}}$ \textbf{for a given mass}", fontsize=16, fontweight='bold', color='brown',
#         verticalalignment='center', horizontalalignment='center', rotation='horizontal',
#         bbox={'facecolor': 'gray', 'alpha': 0.12, 'pad': 4.5})

# plt.text(8e-9, 1.44e-83, r"\textbf{$\mu < 9 \times 10^{-5}$}", fontsize=16, fontweight='bold', color='brown',
#         verticalalignment='center', horizontalalignment='center', rotation='horizontal',
#         bbox={'facecolor': 'gray', 'alpha': 0.12, 'pad': 4.5})

# plt.text(4.63e-6, 2e-3, r"\textbf{PBH overproduction}", fontsize=19, fontweight='bold', color='olive',
#         verticalalignment='center', horizontalalignment='center', rotation='horizontal',
#         bbox={'facecolor': 'salmon', 'alpha': 0.1, 'pad': 4.5})

# plt.text(0.00329, 3.26e-136, r"\textbf{$M_{\rm{BH}} = 10^4\,M_{\odot} $}", fontsize=16, fontweight='bold', color='magenta',
#         verticalalignment='center', horizontalalignment='center', rotation='vertical',
#         bbox={'facecolor': 'gray', 'alpha': 0.1, 'pad': 4.5})

# plt.text(6.22e-5, 3.26e-136, r"\textbf{$M_{\rm{BH}} = 10^5\,M_{\odot} $}", fontsize=16, fontweight='bold', color='purple',
#         verticalalignment='center', horizontalalignment='center', rotation='vertical',
#         bbox={'facecolor': 'gray', 'alpha': 0.1, 'pad': 4.5})

# plt.text(4.53e-7, 3.26e-136, r"\textbf{$M_{\rm{BH}} = 10^7\,M_{\odot} $}", fontsize=16, fontweight='bold', color='teal',
#         verticalalignment='center', horizontalalignment='center', rotation='vertical',
#         bbox={'facecolor': 'gray', 'alpha': 0.1, 'pad': 4.5})



# plt.legend(loc='lower right', fontsize=14, frameon=True, borderpad=0.9, labelspacing=0.8,edgecolor='brown')


# # # # shaded legend box
# # frame = plt.legend().get_frame()
# # frame.set_facecolor('whitesmoke')  # fill
# # frame.set_edgecolor('gray')        # border
# # frame.set_alpha(0.9)
# # frame.set_linewidth(0.8)
# plt.axhline(y=1e-20, color='r', linestyle='--', lw=2)

# # figA.tight_layout()
# # Saving the plot
# plt.savefig('beta_vs_variance.pdf')
# plt.savefig('beta_vs_variance.png')
# plt.show()





############################################################################################################



# # === Plot C: μ vs β at fixed masses for a single p ===


# p_fixed   = 1.0                  # ← change to 2.0 for the other figure
# beta_grid = np.logspace(-40, -1, 28)
# beta_mark = 1e-20

# fig, ax = plt.subplots()
# for i, M in enumerate(mass_list):
#     color   = plt.cm.tab10(i % 10)
#     sigmas  = [sigma_true_for_beta(b, p_fixed) for b in beta_grid]   # σ(β,p_fixed)
#     mu_vals = [mu_of_sigma2_true(s**2, M) for s in sigmas]           # μ(σ^2, M)

#     ax.plot(beta_grid, mu_vals, '-', color=color, lw=3,
#             label=rf"$M=10^{{{int(np.log10(M))}}}\,M_\odot $")  # legend shows only M; p goes in legend title

#     # dot at β = 1e-20
#     # s_mark  = sigma_true_for_beta(beta_mark, p_fixed)
#     # mu_mark = mu_of_sigma2_true(s_mark**2, M)
#     # ax.scatter([beta_mark], [mu_mark], s=20, color=color, zorder=10)

# # # guides
# ax.axhline(MU_LIMIT, ls="--", c="purple")
# # ax.axvline(beta_mark, ls="--", c="r")

# # axes/labels
# ax.set_xscale("log")
# ax.set_yscale("log")
# # ax.set_ylim(1e-6, 1e-3)
# ax.set_xlim(1e-35, 1e-7) # for p=0.6
# # ax.axvspan(beta_mark, ax.get_xlim()[1], color='dimgrey', alpha=0.18, zorder=0)  # PBH overproduction region

# ax.set_xlabel(r"PBH mass fraction at formation $\beta$", fontsize=20)
# ax.set_ylabel(r"Spectral distortion $\mu$", fontsize=20)



# plt.text(
#     3.291e-28,3e-2,
#     r"${P[\zeta] = \frac{1}{2\sqrt{2}\,\sigma_g\, \Gamma \left( 1 + \frac{1}{p} \right)} \exp \left[- \left(\frac{\zeta}{\sqrt{2}\,\sigma_g}\right)^p\right]}$",
#     fontsize=18, fontweight='bold', color='brown',
#     verticalalignment='center', horizontalalignment='center', rotation='horizontal',
#     bbox={'facecolor': 'lightgray', 'alpha': 0.4, 'pad': 7.5}
# )

# plt.text(6.117e-33, 5e-3, r"\textbf{{where $p=1.0 $}", fontsize=16, fontweight='bold', color='brown',
#         verticalalignment='center', horizontalalignment='center', rotation='horizontal',
#         bbox={'facecolor': 'gray', 'alpha': 0.1, 'pad': 4.5})



# # plt.text(2.31e-14, 3.e-2, r"\textbf{PBH overproduction}", fontsize=20, fontweight='bold', color='brown',
# #         verticalalignment='center', horizontalalignment='center', rotation='horizontal',
# #         bbox={'facecolor': 'salmon', 'alpha': 0.1, 'pad': 4.5})

# plt.text(1.88e-25, 4.63e-5, r"\textbf{$\mu_{\rm max}$ (COBE/FIRAS)}", fontsize=18, fontweight='bold', color='purple',
#         verticalalignment='center', horizontalalignment='center', rotation='horizontal',
#         bbox={'facecolor': 'salmon', 'alpha': 0.1, 'pad': 4.5})


# # legend with LaTeX title (math mode)
# leg = ax.legend( ncol=2, fontsize=12,loc='upper left')
# leg.get_title().set_fontsize(12)

# # shaded legend box

# frame = leg.get_frame()
# frame.set_facecolor('whitesmoke')  # fill
# frame.set_edgecolor('gray')        # border
# frame.set_alpha(0.9)
# frame.set_linewidth(0.8)

# ax.grid(True, ls="--", alpha=0.5)
# fig.tight_layout()

# # save one file per p
# p_tag = str(p_fixed).replace('.', 'p')
# fig.savefig(f"mu_vs_beta_p{p_tag}.pdf", bbox_inches="tight")
# fig.savefig(f"mu_vs_beta_p{p_tag}.png", dpi=300, bbox_inches="tight")
# plt.show()







############################################################################################################






# === Plot D (single-mass "p-zoo"): μ vs β at one mass for many close p values ===


# M_single = 1e4                            # pick a seed mass (e.g., 1e4 Msun)
# p_values_zoo = np.arange(0.4, 2.5, 0.2)   # dense around p=2; tweak as you like
# beta_grid = np.logspace(-32, np.log10(3e-4), 36)      # β range; widen/narrow as needed


# fig, ax = plt.subplots()

# # color by p with a colorbar (keeps legend uncluttered)
# cmap = plt.cm.viridis
# norm = plt.Normalize(vmin=p_values_zoo.min(), vmax=p_values_zoo.max())

# for p in p_values_zoo:
#     mu_vals = [mu_of_sigma2_true(sigma_true_for_beta(b, p)**2, M_single) for b in beta_grid]
#     ax.plot(beta_grid, mu_vals, color=cmap(norm(p)), lw=2.5)

 

# # guides and styling
# ax.axhline(MU_LIMIT, ls="--", c="red", lw=2.5)
# ax.set_xscale("log"); ax.set_yscale("log")
# ax.set_ylim(1e-6, 7e-4); ax.set_xlim(1e-32, 2.3e-4)
# ax.set_xlabel(r"PBH mass fraction at formation $\beta$", fontsize=20)
# ax.set_ylabel(r"Spectral distortion $\mu$", fontsize=20)
# ax.grid(True, ls="--", alpha=0.5)

# # colorbar
# sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
# cbar = fig.colorbar(sm, ax=ax, pad=0.015)
# # cbar.set_label(r"$p$", rotation=0, labelpad=8)
# cbar.set_ticks(np.round(np.linspace(p_values_zoo.min(), p_values_zoo.max(), 6), 1))

# # concise title with mass & p-range
# ax.set_title(
#     rf"$M=10^{{{int(np.log10(M_single))}}}\,M_\odot$; "
#     rf"$p\in[{p_values_zoo.min():.1f},{p_values_zoo.max():.1f}]$",
#     fontsize=16
# )
# # optional tiny note on step size
# # ax.text(0.02, 0.03, rf"$\Delta p={p_values_zoo[1]-p_values_zoo[0]:.1f}$",
# #         transform=ax.transAxes, fontsize=10)

# plt.text(9.8e-8, 2.48e-6, r"\textbf{$\mu_{\rm max}$}", fontsize=18, fontweight='bold', color='red',
#         verticalalignment='center', horizontalalignment='center', rotation='horizontal',
#         bbox={'facecolor': 'salmon', 'alpha': 0.1, 'pad': 4.5})

# plt.text(2.3e-8, 1.637e-6, r"\textbf{(COBE/FIRAS)}", fontsize=16, fontweight='bold', color='red',
#         verticalalignment='center', horizontalalignment='center', rotation='horizontal',
#         bbox={'facecolor': 'salmon', 'alpha': 0.1, 'pad': 4.5})




# fig.tight_layout()
# fig.savefig(f"mu_vs_beta_pzoo_M{int(M_single):d}.pdf", bbox_inches="tight")
# fig.savefig(f"mu_vs_beta_pzoo_M{int(M_single):d}.png", dpi=300, bbox_inches="tight")
# plt.show()









# ############################################################################################################

# # === Plot E: μ vs f_NL at fixed β for one mass (quadratic local NG only) ===


# # ζ = ζ_g + (3/5) f_NL (ζ_g^2 - ⟨ζ_g^2⟩), keep up to O(f_NL^2) in variance.


# M_fnl    = 1e5                       # choose the seed mass
# betas    = [1e-17, 1e-14, 1e-12]     # pick 1+ β levels to show
# fNL_grid = np.linspace(-0.8, 15.5, 41)
# # ----------------------------------

# fig, ax = plt.subplots()
# for j, beta_target in enumerate(betas):
#     mu_vals = [mu_from_fNL_at_beta(M_fnl, f, beta_target) for f in fNL_grid]
#     ax.plot(fNL_grid, mu_vals, lw=2.5, label=rf"$\beta={beta_target:.0e}$")

# # guides and labels in your style
# ax.axhline(MU_LIMIT, ls="--", c="red", label=r"$\mu_{\rm max}$ (COBE/FIRAS)",lw=2.5)
# ax.set_xlabel(r"$f_{\mathrm{NL}}$", fontsize=20)
# ax.set_ylabel(r"Spectral distortion $\mu$", fontsize=20)
# ax.set_title(rf"$M=10^{{{int(np.log10(M_fnl))}}}\,M_\odot$", fontsize=16)
# ax.set_yscale("log")  # optional; comment out if you prefer linear y
# ax.grid(True, ls="--", alpha=0.5)
# ax.legend(ncol=2, fontsize=14)
# fig.tight_layout()
# fig.savefig(f"mu_vs_fNL_M{int(M_fnl):d}.pdf", bbox_inches="tight")
# fig.savefig(f"mu_vs_fNL_M{int(M_fnl):d}.png", dpi=300, bbox_inches="tight")
# plt.show()


############################################################################################################

# # === Plot E: beta_at_mu_limit vs p  AND  vs f_NL  (single mass) ===
# M_star = 1e4                 # choose the seed mass you want to diagnose
# p_scan = np.linspace(1.0, 3.0, 17)
# fNL_scan = np.linspace(-0.5, 2.0, 21)

# # variance cap from FIRAS at this mass:
# sigma2_cap = MU_LIMIT / (2.2 * window_factor(k_from_mass(M_star)))

# # p-PDF path: β_μlim(p) = β(σ = sqrt(sigma2_cap), p)
# beta_mu_lim_p = [beta_from_sigma_true(np.sqrt(sigma2_cap), p) for p in p_scan]

# # local-NG path: solve σ_g at μ-limit then β(σ_g, fNL)
# def sigma_g2_at_mu_cap(fNL):
#     A = (18.0/25.0) * (fNL**2)  # coefficient in σ^2 = s + A s^2 (s ≡ σ_g^2)
#     if A == 0.0:
#         return sigma2_cap  # Gaussian case
#     # positive root of A s^2 + s - sigma2_cap = 0
#     return ( -1.0 + np.sqrt(1.0 + 4.0*A*sigma2_cap) ) / (2.0*A)

# beta_mu_lim_f = []
# for f in fNL_scan:
#     s_g2 = sigma_g2_at_mu_cap(f)
#     s_g  = np.sqrt(max(s_g2, 0.0))
#     beta_mu_lim_f.append(beta_from_sigma_fNL(s_g, f))

# # --- Plot (two panels to avoid clutter) ---
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

# ax1.plot(p_scan, beta_mu_lim_p, lw=2)
# ax1.set_xlabel(r"$p$")
# ax1.set_ylabel(r"$\beta$ at $\mu=\mu_{\rm lim}$")
# # ax1.set_title(rf"$M={M_star:.0f}\,M_\odot$: $\beta_{\mu\text{-lim}}(p)$")
# ax1.set_yscale("log")
# ax1.grid(True, ls="--", alpha=0.5)

# ax2.plot(fNL_scan, beta_mu_lim_f, lw=2)
# ax2.set_xlabel(r"$f_{\rm NL}$")
# # ax2.set_title(rf"$M={M_star:.0f}\,M_\odot$: $\beta_{\mu\text{-lim}}(f_{{\rm NL}})$")
# ax2.grid(True, ls="--", alpha=0.5)

# fig.tight_layout()
# fig.savefig(f"beta_at_mu_limit_M{int(M_star):d}.pdf", bbox_inches="tight")
# fig.savefig(f"beta_at_mu_limit_M{int(M_star):d}.png", dpi=300, bbox_inches="tight")
# plt.show()






# =============================================================================
# NEW BLOCKS: μ(p) at fixed β* and mass M* (ζ-based toy p-PDF)

RUN_MU_VS_P_MULTI_BETA = True

if RUN_MU_VS_P_MULTI_BETA:
    M_star     = 1e4                             # representative SMBH-seed mass [M_sun]
    beta_list  = [1e-20, 1e-15, 1e-12]           # a few target PBH abundances
    p_grid     = np.linspace(0.4, 2.4, 11)       # scan tail index p

    fig, ax = plt.subplots()

    colors = plt.cm.plasma(np.linspace(0, 1, len(beta_list)))

    for beta_star, col in zip(beta_list, colors):
        mu_vals = []
        for p in p_grid:
            # invert β(σ,p) = β_*  →  σ_true(p)
            sigma_true = sigma_true_for_beta(beta_star, p)
            # μ(σ_true^2, M_star)
            mu_val = mu_of_sigma2_true(sigma_true**2, M_star)
            mu_vals.append(mu_val)

        mu_vals = np.array(mu_vals)
        ax.plot(p_grid, mu_vals, "o-", lw=2.5, color=col,
                label=rf"$\beta={beta_star:.0e}$")

    # COBE/FIRAS limit
    ax.axhline(MU_LIMIT, ls="--", c="red", lw=2.0, label=r"$\mu_{\rm max}$")

    # Optional: mark Gaussian case p=2
    ax.axvline(2.0, ls=":", c="gray", lw=1.5)

    ax.set_xlabel(r"Tail index $p$ in $P(\zeta)\propto e^{-|\zeta|^p}$", fontsize=18)
    ax.set_ylabel(r"Spectral distortion $\mu$", fontsize=18)
    ax.set_title(
        rf"$\mu(p)$ for several $\beta$ at "
        rf"$M=10^{{{int(np.log10(M_star))}}}\,M_\odot$",
        fontsize=16
    )

    ax.set_yscale("log")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(f"mu_vs_p_multi_beta_M{int(M_star):d}.pdf", bbox_inches="tight")
    fig.savefig(f"mu_vs_p_multi_beta_M{int(M_star):d}.png", dpi=300, bbox_inches="tight")
    plt.show()








# # =============================================================================
# # NEW PLOT 2 (multi-M): β_max(p) at μ = μ_limit for several masses
# # =============================================================================

RUN_BETA_MU_LIMIT_VS_P_MULTI_M = True

if RUN_BETA_MU_LIMIT_VS_P_MULTI_M:
    # --- User choices ---
    M_list  = [1e4, 1e5, 1e7]                # masses in M_sun
    p_grid  = np.linspace(0.4, 1.4, 11)      # tail index p-range

    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(M_list)))

    for M_star, col in zip(M_list, colors):
        # variance cap from μ-limit at this mass:
        sigma2_cap = MU_LIMIT / (2.2 * window_factor(k_from_mass(M_star)))
        sigma_cap  = np.sqrt(sigma2_cap)

        beta_mu_lim = [beta_from_sigma_true(sigma_cap, p) for p in p_grid]
        beta_mu_lim = np.array(beta_mu_lim)

        ax.plot(p_grid, beta_mu_lim, "o-", lw=2.5, color=col,
                label=rf"$M=10^{{{int(np.log10(M_star))}}}\,M_\odot$")

    ax.set_xlabel(r"Tail index $p$ in $P(\zeta)\propto e^{-|\zeta|^p}$", fontsize=18)
    ax.set_ylabel(r"Max. PBH fraction $\beta_{\mu\text{-lim}}(p)$", fontsize=18)
    ax.set_title(r"$\beta_{\mu\text{-lim}}(p)$ at $\mu=\mu_{\rm max}$", fontsize=16)

    ax.set_yscale("log")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig("beta_at_mu_limit_vs_p_multiM.pdf", bbox_inches="tight")
    fig.savefig("beta_at_mu_limit_vs_p_multiM.png", dpi=300, bbox_inches="tight")
    plt.show()








##############################

####################################################################################################
#  Upward step (Cai et al.) non-Gaussian model
#
#  This section implements:
#    - the non-perturbative mapping R = f(R_G; h) on the non-trapped branch,
#    - the induced PDF P(R | sigma_G, h),
#    - PBH fraction β(sigma_G, h),
#    - true variance σ_true^2(sigma_G, h),
#    - μ(sigma_G, h; M),
#    - μ(β; h, M) by inverting β(sigma_G, h).
#
#  It assumes the following are already defined above in this file:
#    - mu_of_sigma2_true(s2_true, M_sun)
#    - k_from_mass(M_sun), window_factor(k)
#    - MU_LIMIT
#    - M_cmp, beta_grid, p_list_small, mu_from_beta_zeta (from the p-type block)
####################################################################################################

import math   # for erf, sqrt, etc.

# ------------------------- Core mapping: R_G  <->  R  -------------------------
# In Cai et al.'s step-dominated regime (g^2 |h| << 1), the non-perturbative δN result
# gives the relation between the "linear" curvature R_G (Gaussian) and the full curvature R:
#
#   R  = -2/|h| [ sqrt(1 - |h| R_G) - 1 ]
#
# This can be inverted exactly to:
#
#   R_G(R) = R - (|h|/4) R^2.
#
# On the non-trapped branch, R ranges from -∞ up to a finite maximum R_max = 2/|h|.
# The trapped branch (R_G > 1/|h|) gives additional PBHs but has negligible impact on μ.
# For now we condition on *not* being trapped and normalise on this branch only.

def cai_RG_from_R(R, h):
    """
    Inverse map: given R and step parameter h, return R_G on the non-trapped branch.

    R_G(R) = R - (|h|/4) R^2

    This is independent of sigma_G; sigma_G only sets the Gaussian weight.
    """
    alpha = abs(h)
    return R - 0.25 * alpha * R**2

def cai_Rmax(h):
    """
    Maximum curvature reachable on the non-trapped branch.

    This corresponds to R_G = 1/|h| in Cai et al.
    """
    alpha = abs(h)
    return 2.0 / alpha

# -------------------- Unnormalised branch PDF on R --------------------

def _cai_branch_pdf_R_unnorm(R, sigma_G, h):
    """
    Unnormalised PDF for R on the non-trapped branch.

    - Underlying Gaussian variable R_G ~ N(0, sigma_G^2).
    - Mapping: R_G = R - (|h|/4) R^2.
    - Jacobian: dR_G/dR = 1 - (|h|/2) R.
    - Domain: R ≤ R_max = 2/|h| (equivalently R_G ≤ 1/|h|).

    We do *not* divide by the branch normalisation here; that is done in cai_pdf_R.
    """
    alpha = abs(h)
    s2G   = sigma_G**2
    R     = np.asarray(R, float)

    # Map R -> R_G
    RG  = cai_RG_from_R(R, h)
    jac = 1.0 - 0.5 * alpha * R   # dR_G/dR

    # Gaussian PDF in R_G
    pref   = 1.0 / np.sqrt(2.0 * np.pi * s2G)
    pdf_RG = pref * np.exp(-0.5 * RG*RG / s2G)

    # Enforce branch domain R <= R_max
    Rmax = cai_Rmax(h)
    out  = np.zeros_like(R)
    mask = (R <= Rmax)
    out[mask] = pdf_RG[mask] * jac[mask]
    return out

def _cai_branch_norm(sigma_G, h):
    """
    Branch normalisation N_branch = Prob(R_G ≤ 1/|h|) under the Gaussian.

    This can be computed analytically as a Gaussian CDF, so we do not need a numerical
    integral here.  This is the probability weight of non-trapped trajectories.
    """
    alpha = abs(h)
    if sigma_G <= 0.0:
        return 0.0

    sqrt2 = math.sqrt(2.0)
    t_top = 1.0 / (alpha * sigma_G * sqrt2)        # 1/(|h| σ_G sqrt(2))

    # Φ(t_top) = 0.5 [1 + erf(t_top)]
    return 0.5 * (1.0 + math.erf(t_top))

def cai_pdf_R(R, sigma_G, h):
    """
    Normalised PDF P(R | sigma_G, h) on the non-trapped branch.

    We divide the unnormalised PDF by the branch probability N_branch, so this is
    conditional on *not* being trapped in the step minimum.
    """
    N = _cai_branch_norm(sigma_G, h)
    R = np.asarray(R, float)
    if N == 0.0:
        return np.zeros_like(R)
    return _cai_branch_pdf_R_unnorm(R, sigma_G, h) / N

# ------------------------------ β(σ_G, h) ------------------------------

def cai_beta_from_sigmaG(sigma_G, h, zeta_c=0.67):
    """
    PBH mass fraction β(σ_G, h) on the non-trapped branch, for a curvature
    threshold zeta_c (e.g. ζ_c = 0.67).

    Because R_G(R) is monotonic on the branch, we can compute β analytically
    from the Gaussian tail in R_G:

      - R ∈ [ζ_c, R_max]  maps to  R_G ∈ [R_G(ζ_c), 1/|h|].
      - β_uncond = Prob(R_G ∈ [R_G(ζ_c), 1/|h|]).
      - N_branch = Prob(R_G ≤ 1/|h|).
      - β_branch = β_uncond / N_branch.

    We return β_branch, i.e. β conditional on non-trapped trajectories.
    """
    alpha = abs(h)
    if sigma_G <= 0.0:
        return 0.0

    Rmax = cai_Rmax(h)
    if zeta_c >= Rmax:
        # Threshold above maximum curvature on this branch: no PBHs from this channel.
        return 0.0

    # Map ζ_c to R_G(ζ_c)
    RG_c = cai_RG_from_R(zeta_c, h)

    sqrt2 = math.sqrt(2.0)
    t_top = 1.0 / (alpha * sigma_G * sqrt2)      # R_G = 1/|h|
    t_c   = RG_c / (sigma_G * sqrt2)             # R_G = R_G(ζ_c)

    Phi_top = 0.5 * (1.0 + math.erf(t_top))
    Phi_c   = 0.5 * (1.0 + math.erf(t_c))

    N_branch    = Phi_top                              # Prob(R_G ≤ 1/|h|)
    beta_uncond = Phi_top - Phi_c                      # Prob(R_G ∈ [RG_c, 1/|h|])

    if N_branch == 0.0:
        return 0.0

    beta_branch = beta_uncond / N_branch
    return max(beta_branch, 0.0)

# ---------------------- σ_true^2(σ_G, h) and μ(σ_G, h) ----------------------

def cai_sigma2_true_from_sigmaG(sigma_G, h, n=4000):
    """
    True variance of R on the non-trapped branch:

        σ_true^2(sigma_G, h) = ⟨R^2⟩_branch
                             = ∫ R^2 P(R | sigma_G, h) dR.

    We compute this with a simple trapezoidal integral over R.
    """
    if sigma_G <= 0.0:
        return 0.0

    Rmax = cai_Rmax(h)
    Rmin = -10.0 * sigma_G        # generous lower bound; Gaussian weight kills far tail

    R = np.linspace(Rmin, Rmax, n)
    pdf_unnorm = _cai_branch_pdf_R_unnorm(R, sigma_G, h)
    N_branch   = _cai_branch_norm(sigma_G, h)
    if N_branch == 0.0:
        return 0.0

    pdf = pdf_unnorm / N_branch
    return float(np.trapz(pdf * R * R, R))

def cai_mu_from_sigmaG(sigma_G, h, M_sun, n=4000):
    """
    μ-distortion for the Cai upward-step model with a quasi-Dirac spike
    at mass M_sun.

    Implementation:
      1. Compute σ_true^2(sigma_G, h).
      2. Feed it into the existing μ(σ^2, M) mapping.
    """
    s2_true = cai_sigma2_true_from_sigmaG(sigma_G, h, n=n)
    return mu_of_sigma2_true(s2_true, M_sun)

# --------------------------- Invert β: σ_G(β, h) ---------------------------

def cai_sigmaG_for_beta(beta_target, h, zeta_c=0.67,
                        sG_lo=1e-4, sG_hi=0.5, iters=80):
    """
    Solve β(σ_G, h) = β_target for σ_G by bisection.

    For fixed (h, ζ_c), β(σ_G, h) is monotonic increasing in the regime of interest,
    so a simple bisection on σ_G is sufficient.
    """
    if beta_target <= 0.0:
        return 0.0

    def f(sG): return cai_beta_from_sigmaG(sG, h, zeta_c=zeta_c)

    # Expand upper bracket if β is still too small there.
    for _ in range(60):
        if f(sG_hi) < beta_target:
            sG_hi *= 2.0
        else:
            break

    # Shrink lower bracket if β is already too large there.
    for _ in range(60):
        if f(sG_lo) > beta_target and sG_lo > 1e-12:
            sG_lo *= 0.5
        else:
            break

    # Bisection loop.
    for _ in range(iters):
        mid = 0.5 * (sG_lo + sG_hi)
        if f(mid) >= beta_target:
            sG_hi = mid
        else:
            sG_lo = mid

    return 0.5 * (sG_lo + sG_hi)

def cai_mu_from_beta(beta_val, h, M_sun, zeta_c=0.67, n=4000):
    """
    Direct μ(β; h, M) map for the Cai upward-step model:

        β  --invert-->  σ_G(β, h)
                      --→ σ_true^2(σ_G, h)
                      --→ μ(σ_true^2, M_sun).

    This is the quantity we want to compare directly to the p-type μ(β) curves.
    """
    sigma_G = cai_sigmaG_for_beta(beta_val, h, zeta_c=zeta_c)
    return cai_mu_from_sigmaG(sigma_G, h, M_sun, n=n)

####################################################################################################
#  Optional sanity check: normalisation and shape of P(R) for one (σ_G, h).
####################################################################################################

RUN_CAI_SANITY = False  # set True once if you want to inspect the PDF

if RUN_CAI_SANITY:
    test_sigmaG = 0.1
    test_h      = -1.0

    R_test = np.linspace(-10.0 * test_sigmaG, cai_Rmax(test_h), 5000)
    pdf_test = cai_pdf_R(R_test, test_sigmaG, test_h)
    norm_val = np.trapz(pdf_test, R_test)
    print(f"[Cai sanity] ∫ P(R) dR ≈ {norm_val:.6f} (should be ≈ 1)")

    plt.figure()
    plt.plot(R_test, pdf_test, lw=2)
    plt.xlabel(r"$R$")
    plt.ylabel(r"$P(R)$")
    plt.title(rf"Cai PDF, $\sigma_G={test_sigmaG:.2f}$, $h={test_h}$")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

####################################################################################################
# Shared plot parameters for the Cai model (used in multiple figures).
# (Feel free to tweak these once and let all Cai plots update together.)
CAI_H_VALUES = [-0.5, -1.0, -2.5]
CAI_ZETA_C = 0.67

####################################################################################################
#  Plot A: μ(β) at fixed mass, comparing p-type ζ vs Cai upward step
#
#  - Uses the existing beta_grid, M_cmp and p_list_small from your p-type ζ/compaction block.
#  - Overlays:
#      * p-type ζ curves: μ(β; p) from mu_from_beta_zeta,
#      * Cai curves: μ(β; h) from cai_mu_from_beta.
#
#  (Toggle with RUN_CAI_MU_VS_BETA to skip this figure.)
####################################################################################################

RUN_CAI_MU_VS_BETA = True

if RUN_CAI_MU_VS_BETA:
    fig, ax = plt.subplots()

    # p-type ζ curves (already defined via sigma_true_for_beta + mu_of_sigma2_true)
    for p in p_list_small:
        mu_zet = [mu_from_beta_zeta(b, p) for b in beta_grid]
        ax.loglog(beta_grid, mu_zet, "--", lw=2.3,
                  label=rf"$p$-type $\zeta$, $p={p:g}$")

    # Cai upward-step curves
    for h in CAI_H_VALUES:
        mu_cai = [cai_mu_from_beta(b, h, M_cmp, zeta_c=CAI_ZETA_C) for b in beta_grid]
        ax.loglog(beta_grid, mu_cai, "-", lw=2.4,
                  label=rf"Cai step, $h={h}$")

    # μ upper limit (FIRAS)
    ax.axhline(MU_LIMIT, ls="--", c="red", lw=2.0, label=r"$\mu_{\rm max}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(beta_grid.min(), beta_grid.max())

    ax.set_xlabel(r"PBH mass fraction at formation $\beta$", fontsize=20)
    ax.set_ylabel(r"Spectral distortion $\mu$", fontsize=20)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(ncol=1, fontsize=11)

    ax.set_title(
        rf"$M = 10^{{{int(np.log10(M_cmp))}}}\,M_\odot$: "
        r"$\mu(\beta)$, $p$-type $\zeta$ vs Cai upward step",
        fontsize=15
    )

    fig.tight_layout()
    fig.savefig(f"mu_vs_beta_Cai_vs_p_M{int(M_cmp):d}.pdf",  bbox_inches="tight")
    fig.savefig(f"mu_vs_beta_Cai_vs_p_M{int(M_cmp):d}.png",  dpi=300, bbox_inches="tight")
    plt.show()

####################################################################################################
#  Plot B: β vs σ_true^2 at fixed mass, for different h (Cai) + Gaussian baseline
#
#  - x-axis: true variance σ_true^2 of R on the PBH scale.
#  - y-axis: β from the Cai model (non-trapped branch).
#  - Curves: different h values, plus a Gaussian (erfc) reference.
#  - We also mark the μ-limit as a vertical line in σ_true^2.
#
#  (Toggle with RUN_CAI_BETA_VS_SIGMA2 to skip this figure.)
####################################################################################################

RUN_CAI_BETA_VS_SIGMA2 = True

if RUN_CAI_BETA_VS_SIGMA2:
    sG_grid_for_plot = np.logspace(-3, 0, 200)   # underlying Gaussian width range
    zeta_c_plot      = CAI_ZETA_C

    figB, axB = plt.subplots()
    colors = plt.cm.plasma(np.linspace(0, 1, len(CAI_H_VALUES)))

    for color, h in zip(colors, CAI_H_VALUES):
        sigma2_true_vals = []
        beta_vals        = []

        for sG in sG_grid_for_plot:
            s2_true = cai_sigma2_true_from_sigmaG(sG, h)
            beta    = cai_beta_from_sigmaG(sG, h, zeta_c=zeta_c_plot)
            sigma2_true_vals.append(s2_true)
            beta_vals.append(beta)

        sigma2_true_vals = np.array(sigma2_true_vals)
        beta_vals        = np.array(beta_vals)

        idx = np.argsort(sigma2_true_vals)
        sigma2_true_vals = sigma2_true_vals[idx]
        beta_vals        = beta_vals[idx]

        axB.loglog(sigma2_true_vals, beta_vals, lw=2.3, color=color,
                   label=rf"Cai, $h={h}$")

    # Gaussian reference: β_G(σ_true) = 0.5 erfc(ζ_c / (sqrt(2) σ_true)).
    from math import erfc
    s2_gauss = np.logspace(-8, -0.5, 200)
    sigma_gauss = np.sqrt(s2_gauss)
    beta_gauss  = 0.5 * erfc(zeta_c_plot / (np.sqrt(2.0) * sigma_gauss))
    axB.loglog(s2_gauss, beta_gauss, "k--", lw=2.0, label=r"Gaussian tail")

    # Vertical line at σ_true^2 corresponding to μ = μ_max at this mass.
    k_star_cmp   = k_from_mass(M_cmp)
    W_mu_cmp     = window_factor(k_star_cmp)
    sigma2_muLim = MU_LIMIT / (2.2 * W_mu_cmp)

    axB.axvline(sigma2_muLim, color="red", ls=":", lw=2.0,
                label=r"$\sigma_{\rm true}^2$ at $\mu_{\rm max}$")

    axB.set_xscale("log")
    axB.set_yscale("log")

    axB.set_xlabel(r"True variance $\sigma_{\rm true}^2$ on PBH scale", fontsize=20)
    axB.set_ylabel(r"PBH mass fraction at formation $\beta$", fontsize=20)
    axB.grid(True, ls="--", alpha=0.5)
    axB.legend(fontsize=10, loc="lower right")

    axB.set_title(
        rf"$M = 10^{{{int(np.log10(M_cmp))}}}\,M_\odot$: "
        r"$\beta(\sigma_{\rm true}^2)$, Cai upward step vs Gaussian",
        fontsize=15
    )

    figB.tight_layout()
    figB.savefig(f"beta_vs_sigma2_Cai_M{int(M_cmp):d}.pdf", bbox_inches="tight")
    figB.savefig(f"beta_vs_sigma2_Cai_M{int(M_cmp):d}.png", dpi=300, bbox_inches="tight")
    plt.show()
