import numpy as np
import matplotlib.pyplot as plt
from math import gamma
from matplotlib import rcParams
from matplotlib import rc
from matplotlib.patches import FancyArrowPatch
# activate latex text rendering
# plt.rc('text', usetex=True)

#LaTex setting
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams['text.latex.preamble'] = r'\boldmath'


#Plot setting:
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{color}\usepackage{amssymb}\boldmath'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{xcolor}\usepackage{amssymb}\boldmath'

# If you want to enable usetex, set USE_TEX=True and ensure LaTeX is installed
# (kept from your previous draft)
# USE_TEX = False
# plt.rc('text', usetex=USE_TEX)

# === Active, unified LaTeX setup matching your demo style ===
plt.rc('text', usetex=True)  # requires LaTeX installed
plt.rcParams['text.latex.preamble'] = (
    r'\usepackage{amsmath}'
    r'\usepackage{amssymb}'
    r'\usepackage{xcolor}'
    r'\boldmath'
)

#LaTex setting
# original preamble lines preserved below (some users toggle these when usetex=True)
# plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
# plt.rcParams['text.latex.preamble'] = r'\boldmath'


#Plot setting:
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 11
# preserve original preamble assignments as comments (do not delete)
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{color}\usepackage{amssymb}\boldmath'
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{xcolor}\usepackage{amssymb}\boldmath'

#plt.rcParams['font.family'] = 'Times New Roman'

plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.4*plt.rcParams['font.size']
#plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = 1.4*plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = 1.4*plt.rcParams['font.size']
# dots per inch: dpi
#plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']

plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1

#legends
#plt.rcParams['legend.frameon'] = False
#plt.rcParams['legend.loc'] = 'center left'
#plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']

plt.rcParams['axes.linewidth'] = 1

#border setting
#plt.gca().spines['right'].set_color('none')
#plt.gca().spines['top'].set_color('none')

#ticks position setting
# original active calls that create a figure at import time are preserved as comments:
# plt.gca().xaxis.set_ticks_position('bottom')
# plt.gca().yaxis.set_ticks_position('left')

# safer alternative (no figure creation at import time):
plt.rcParams['xtick.top'] = False
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = False

#f = plt.figure()
#ax = f.add_subplot(111)
#ax.tick_params(labeltop=False, labelright=True)
#If we don't want to use x-axis and y-axis values.
#plt.gca().axes.xaxis.set_ticks([])
#plt.gca().axes.yaxis.set_ticks([]) 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





# ---- Mappings ----

def k_from_mass(M_sun):
    return 92.0 * np.sqrt(5e8 / M_sun)

def window_factor(k):
    k = np.asarray(k, float)
    a = np.exp(-k/5400.0)
    b = k/31.6
    return a - np.where(b > 50.0, 0.0, np.exp(-b*b))  # avoid underflow noise

def mu_of_sigma2_true(s2_true, M_sun):
    return 2.2 * np.asarray(s2_true, float) * window_factor(k_from_mass(M_sun))

# Variance map: σ_true^2 = C(p) * σ̃^2
def C_of_p(p):
    return (2.0 * gamma(1.0 + 3.0/p)) / (3.0 * gamma(1.0 + 1.0/p))

# ---- Settings you can change ----
MU_LIMIT = 9e-5
mass_list = [1e4, 1e5, 1e7]     # four distinct masses (Msun)
p_values  = [0.6]     # plot both on same figure
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




def _gauss_cdf(y, sigma):
    # local, so we don't change your imports
    from math import erf
    return 0.5*(1.0 + erf(y/(np.sqrt(2.0)*sigma)))

def beta_from_sigma_fNL(sigma_g, fNL, zeta_c=0.67):
    """
    Press–Schechter tail for quadratic local NG:
    ζ(y) = y + a (y^2 - σ_g^2), where y ≡ ζ_g ~ N(0,σ_g^2), a = (3/5) fNL.
    β = Prob[ζ ≥ ζ_c] = ∫ 1_{ζ(y)≥ζ_c} N(y;0,σ_g^2) dy
    Closed-form via roots of a y^2 + y - (a σ_g^2 + ζ_c) = 0.
    """
    a = 0.6 * fNL  # (3/5) fNL
    if abs(a) < 1e-14:
        # Gaussian limit
        from math import erfc
        return 0.5*erfc(zeta_c/(np.sqrt(2.0)*sigma_g))

    # Quadratic roots where ζ(y) = ζ_c
    D = 1.0 + 4.0*a*(a*sigma_g**2 + zeta_c)
    if D < 0:
        # No real roots: decide by vertex value
        yv = -1.0/(2.0*a)
        zeta_v = yv + a*(yv**2 - sigma_g**2)
        if a > 0:
            return 1.0 if zeta_v >= zeta_c else 0.0
        else:
            # for a<0 and D<0, vertex < ζ_c ⇒ always below threshold
            return 0.0

    r1 = (-1.0 - np.sqrt(D)) / (2.0*a)
    r2 = (-1.0 + np.sqrt(D)) / (2.0*a)
    r1, r2 = (min(r1, r2), max(r1, r2))

    if a > 0:
        # ζ≥ζc for y ≤ r1 or y ≥ r2  ⇒ β = 1 - [Φ(r2) - Φ(r1)]
        return 1.0 - (_gauss_cdf(r2, sigma_g) - _gauss_cdf(r1, sigma_g))
    else:
        # a<0: ζ≥ζc for r1 ≤ y ≤ r2  ⇒ β = Φ(r2) - Φ(r1)
        return _gauss_cdf(r2, sigma_g) - _gauss_cdf(r1, sigma_g)

def sigma_g_for_beta_fNL(beta_target, fNL, zeta_c=0.67, s_lo=1e-8, s_hi=1.0, iters=80):
    """Invert β(σ_g,fNL)=β_target via bisection (β increases with σ_g)."""
    f = lambda s: beta_from_sigma_fNL(s, fNL, zeta_c)
    # expand bracket if needed
    for _ in range(80):
        if f(s_hi) < beta_target: s_hi *= 2.0
        else: break
    for _ in range(80):
        if f(s_lo) > beta_target: s_lo *= 0.5
        else: break
    # bisection
    for _ in range(iters):
        mid = 0.5*(s_lo + s_hi)
        if f(mid) >= beta_target: s_hi = mid
        else: s_lo = mid
    return 0.5*(s_lo + s_hi)

def mu_from_fNL_at_beta(M_sun, fNL, beta_target, zeta_c=0.67):
    """
    Given (M, fNL, β): solve σ_g, compute true variance
      σ^2 = σ_g^2 + (18/25) fNL^2 σ_g^4  (to O(fNL^2)),
    then μ = 2.2 σ^2 W(k*).
    """
    sigma_g = sigma_g_for_beta_fNL(beta_target, fNL, zeta_c=zeta_c)
    sigma2_true = sigma_g**2 + (18.0/25.0)*(fNL**2)*(sigma_g**4)
    return mu_of_sigma2_true(sigma2_true, M_sun)





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









############################################################################################################

# === Plot E: μ vs f_NL at fixed β for one mass (quadratic local NG only) ===


# ζ = ζ_g + (3/5) f_NL (ζ_g^2 - ⟨ζ_g^2⟩), keep up to O(f_NL^2) in variance.


M_fnl    = 1e5                       # choose the seed mass
betas    = [1e-17, 1e-14, 1e-12]     # pick 1+ β levels to show
fNL_grid = np.linspace(-0.8, 15.5, 41)
# ----------------------------------

fig, ax = plt.subplots()
for j, beta_target in enumerate(betas):
    mu_vals = [mu_from_fNL_at_beta(M_fnl, f, beta_target) for f in fNL_grid]
    ax.plot(fNL_grid, mu_vals, lw=2.5, label=rf"$\beta={beta_target:.0e}$")

# guides and labels in your style
ax.axhline(MU_LIMIT, ls="--", c="red", label=r"$\mu_{\rm max}$ (COBE/FIRAS)",lw=2.5)
ax.set_xlabel(r"$f_{\mathrm{NL}}$", fontsize=20)
ax.set_ylabel(r"Spectral distortion $\mu$", fontsize=20)
ax.set_title(rf"$M=10^{{{int(np.log10(M_fnl))}}}\,M_\odot$", fontsize=16)
ax.set_yscale("log")  # optional; comment out if you prefer linear y
ax.grid(True, ls="--", alpha=0.5)
ax.legend(ncol=2, fontsize=14)
fig.tight_layout()
fig.savefig(f"mu_vs_fNL_M{int(M_fnl):d}.pdf", bbox_inches="tight")
fig.savefig(f"mu_vs_fNL_M{int(M_fnl):d}.png", dpi=300, bbox_inches="tight")
plt.show()


############################################################################################################

