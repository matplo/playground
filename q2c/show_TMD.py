import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
import matplotlib.patheffects as pe

# --- Model TMD distributions (phenomenologically motivated) ---
# Using Gaussian ansatz with parameters loosely inspired by MAP/Pavia extractions

def unpolarized_tmd(kx, ky, x=0.1, flavor='u'):
    """f1(x, kT) - unpolarized TMD, Gaussian model"""
    k2 = kx**2 + ky**2
    if flavor == 'u':
        width2 = 0.25  # GeV^2
        norm = 2.0
    elif flavor == 'd':
        width2 = 0.20
        norm = 0.8
    else:  # gluon
        width2 = 0.35
        norm = 3.0
    return norm * np.exp(-k2 / width2)

def sivers_tmd(kx, ky, x=0.1, flavor='u', S_dir='up'):
    """f1T^perp contribution - Sivers asymmetry
    Correlation: k_T x (P x S_T) -> for S_T along y, asymmetry along x
    """
    k2 = kx**2 + ky**2
    kT = np.sqrt(k2)
    
    if flavor == 'u':
        # u-quark Sivers: negative (SIDIS convention)
        sivers_strength = -0.15
        width2 = 0.25
        sivers_width2 = 0.18
    elif flavor == 'd':
        # d-quark Sivers: positive, smaller magnitude
        sivers_strength = 0.06
        width2 = 0.20
        sivers_width2 = 0.15
    else:
        sivers_strength = 0.0
        width2 = 0.35
        sivers_width2 = 0.30
    
    # Sign convention: for proton spin along +y, Sivers shifts along +/-x
    sign = 1.0 if S_dir == 'up' else -1.0
    
    # Sivers modulation: proportional to kx (for S along y-axis)
    f_unp = np.exp(-k2 / width2)
    f_sivers = sivers_strength * sign * kx * np.exp(-k2 / sivers_width2) / 0.5  # normalized
    
    return f_unp + f_sivers

def boer_mulders(kx, ky, x=0.1):
    """h1^perp - Boer-Mulders function (unpolarized hadron, polarized quark)
    Creates cos(2phi) modulation"""
    k2 = kx**2 + ky**2
    phi = np.arctan2(ky, kx)
    strength = 0.08
    width2 = 0.22
    return np.exp(-k2 / width2) * (1 + strength * np.cos(2 * phi))

# --- Set up grid ---
kmax = 1.2  # GeV
npts = 300
kx = np.linspace(-kmax, kmax, npts)
ky = np.linspace(-kmax, kmax, npts)
KX, KY = np.meshgrid(kx, ky)

# --- Create figure ---
fig = plt.figure(figsize=(18, 14), facecolor='#0a0a1a')
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30, 
                       left=0.06, right=0.94, top=0.92, bottom=0.06)

# Custom colormaps
cmap_hot = plt.cm.inferno
cmap_div = plt.cm.RdBu_r
cmap_cool = plt.cm.magma
cmap_viridis = plt.cm.viridis

panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
axes = []

def style_axis(ax, title, label_idx):
    ax.set_xlabel(r'$k_x$ [GeV]', fontsize=13, color='white')
    ax.set_ylabel(r'$k_y$ [GeV]', fontsize=13, color='white')
    ax.set_title(title, fontsize=14, color='white', pad=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.tick_params(colors='white', labelsize=11)
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(0.5)
    # Panel label
    ax.text(0.05, 0.95, panel_labels[label_idx], transform=ax.transAxes,
            fontsize=15, color='white', fontweight='bold', va='top',
            path_effects=[pe.withStroke(linewidth=3, foreground='black')])
    return ax

# ============================================================
# Panel (a): Unpolarized u-quark TMD
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
Z = unpolarized_tmd(KX, KY, flavor='u')
im1 = ax1.pcolormesh(KX, KY, Z, cmap=cmap_hot, shading='gouraud')
cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cb1.ax.tick_params(colors='white', labelsize=9)
cb1.set_label(r'$f_1^u(x, \mathbf{k}_T)$', color='white', fontsize=12)
style_axis(ax1, r'Unpolarized $u$-quark TMD', 0)
ax1.text(0.5, -0.18, r'$x = 0.1$, $Q^2 = 4$ GeV$^2$', transform=ax1.transAxes,
         fontsize=10, color='#aaaaaa', ha='center')

# ============================================================
# Panel (b): Unpolarized gluon TMD
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])
Z_g = unpolarized_tmd(KX, KY, flavor='g')
im2 = ax2.pcolormesh(KX, KY, Z_g, cmap=cmap_cool, shading='gouraud')
cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cb2.ax.tick_params(colors='white', labelsize=9)
cb2.set_label(r'$f_1^g(x, \mathbf{k}_T)$', color='white', fontsize=12)
style_axis(ax2, r'Unpolarized Gluon TMD', 1)
ax2.text(0.5, -0.18, r'Broader width $\langle k_T^2 \rangle_g > \langle k_T^2 \rangle_q$',
         transform=ax2.transAxes, fontsize=10, color='#aaaaaa', ha='center')

# ============================================================
# Panel (c): u-quark Sivers (SIDIS) - proton spin up
# ============================================================
ax3 = fig.add_subplot(gs[0, 2])
Z_siv_u = sivers_tmd(KX, KY, flavor='u', S_dir='up')
# Show the asymmetry: difference from unpolarized
Z_siv_asym_u = Z_siv_u - unpolarized_tmd(KX, KY, flavor='u')
vmax = np.max(np.abs(Z_siv_asym_u)) * 0.9
im3 = ax3.pcolormesh(KX, KY, Z_siv_asym_u, cmap=cmap_div, shading='gouraud',
                      vmin=-vmax, vmax=vmax)
cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cb3.ax.tick_params(colors='white', labelsize=9)
cb3.set_label(r'$\Delta f_{\mathrm{Sivers}}^u$', color='white', fontsize=12)
style_axis(ax3, r'$u$-quark Sivers Asymmetry (SIDIS)', 2)
# Add spin arrow
ax3.annotate('', xy=(0, 0.9), xytext=(0, 0.5),
            arrowprops=dict(arrowstyle='->', color='lime', lw=2.5))
ax3.text(0.12, 0.75, r'$\vec{S}_T$', fontsize=16, color='lime', fontweight='bold')

# ============================================================
# Panel (d): Full u-quark TMD with Sivers (SIDIS)
# ============================================================
ax4 = fig.add_subplot(gs[1, 0])
Z_full_u = sivers_tmd(KX, KY, flavor='u', S_dir='up')
im4 = ax4.pcolormesh(KX, KY, Z_full_u, cmap=cmap_hot, shading='gouraud')
cb4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cb4.ax.tick_params(colors='white', labelsize=9)
cb4.set_label(r'$f_1^u + f_{1T}^{\perp\, u}$ (SIDIS)', color='white', fontsize=11)
style_axis(ax4, r'$u$-quark: Unpol. + Sivers (SIDIS)', 3)
# Add contours to show distortion
contour_levels = np.linspace(0.3, 1.8, 6)
ax4.contour(KX, KY, Z_full_u, levels=contour_levels, colors='white', 
            linewidths=0.6, alpha=0.5)
ax4.annotate('', xy=(0, 0.9), xytext=(0, 0.5),
            arrowprops=dict(arrowstyle='->', color='lime', lw=2.5))
ax4.text(0.12, 0.75, r'$\vec{S}_T$', fontsize=16, color='lime', fontweight='bold')
ax4.text(0.5, -0.18, r'Distortion: quarks shifted $\leftarrow$ (SIDIS)',
         transform=ax4.transAxes, fontsize=10, color='#aaaaaa', ha='center')

# ============================================================
# Panel (e): Sign change - u-quark Sivers in Drell-Yan (flipped!)
# ============================================================
ax5 = fig.add_subplot(gs[1, 1])
# DY has opposite sign!
Z_siv_DY = sivers_tmd(KX, KY, flavor='u', S_dir='up')
Z_siv_asym_DY = -(Z_siv_DY - unpolarized_tmd(KX, KY, flavor='u'))  # SIGN FLIP
im5 = ax5.pcolormesh(KX, KY, Z_siv_asym_DY, cmap=cmap_div, shading='gouraud',
                      vmin=-vmax, vmax=vmax)
cb5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
cb5.ax.tick_params(colors='white', labelsize=9)
cb5.set_label(r'$\Delta f_{\mathrm{Sivers}}^u$ (DY)', color='white', fontsize=12)
style_axis(ax5, r'$u$-quark Sivers Asymmetry (Drell-Yan)', 4)
ax5.annotate('', xy=(0, 0.9), xytext=(0, 0.5),
            arrowprops=dict(arrowstyle='->', color='lime', lw=2.5))
ax5.text(0.12, 0.75, r'$\vec{S}_T$', fontsize=16, color='lime', fontweight='bold')
ax5.text(0.5, -0.18, r'$f_{1T}^{\perp\,\mathrm{DY}} = -f_{1T}^{\perp\,\mathrm{SIDIS}}$ (Collins 2002)',
         transform=ax5.transAxes, fontsize=10, color='#ffcc00', ha='center', fontweight='bold')

# ============================================================
# Panel (f): Boer-Mulders cos(2phi) modulation
# ============================================================
ax6 = fig.add_subplot(gs[1, 2])
Z_bm = boer_mulders(KX, KY)
Z_bm_asym = Z_bm - np.exp(-(KX**2 + KY**2) / 0.22)
vmax_bm = np.max(np.abs(Z_bm_asym)) * 0.9
im6 = ax6.pcolormesh(KX, KY, Z_bm_asym, cmap=cmap_div, shading='gouraud',
                      vmin=-vmax_bm, vmax=vmax_bm)
cb6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
cb6.ax.tick_params(colors='white', labelsize=9)
cb6.set_label(r'$\Delta h_1^{\perp}$', color='white', fontsize=12)
style_axis(ax6, r'Boer-Mulders $\cos(2\phi)$ Modulation', 5)
ax6.text(0.5, -0.18, r'Unpolarized hadron, polarized quark',
         transform=ax6.transAxes, fontsize=10, color='#aaaaaa', ha='center')

# ============================================================
# Supertitle
# ============================================================
fig.suptitle('Momentum-Space Imaging of the Proton: Transverse Momentum Dependent Distributions',
             fontsize=17, color='white', fontweight='bold', y=0.97)
fig.text(0.5, 0.005, 
         'Phenomenological Gaussian models with parameters inspired by MAP/Pavia global TMD extractions  •  Illustrative only',
         ha='center', fontsize=10, color='#888888', style='italic')

plt.savefig('./proton_tmd_visualization.png', dpi=180, facecolor='#0a0a1a',
            bbox_inches='tight')
print("Done!")