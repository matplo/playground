#!/usr/bin/env python3
"""
QCD Phase Diagram: Temperature vs Baryon Chemical Potential
With experimental freeze-out points from RHIC BES, RHIC, LHC, SPS, AGS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

# Set up figure
fig, ax = plt.subplots(figsize=(10, 8))

# ============================================
# FREEZE-OUT DATA (Andronic et al., Nature 561, 321 (2018))
# and STAR BES-II preliminary
# ============================================
# Format: (√s_NN [GeV], T [MeV], μ_B [MeV], T_err, μ_B_err, label)
freeze_out_data = [
    # LHC
    (2760, 156.5, 0.7, 1.5, 3.8, 'LHC 2.76 TeV'),
    (5020, 156.5, 0.4, 1.5, 2.0, 'LHC 5.02 TeV'),
    # RHIC top energy
    (200, 160, 24, 4, 5, 'RHIC 200 GeV'),
    # RHIC BES-I/II
    (62.4, 164, 70, 5, 10, '62.4 GeV'),
    (54, 163, 83, 5, 10, '54 GeV'),
    (39, 162, 103, 5, 10, '39 GeV'),
    (27, 160, 144, 5, 12, '27 GeV'),
    (19.6, 158, 188, 5, 15, '19.6 GeV'),
    (14.5, 156, 239, 6, 18, '14.5 GeV'),
    (11.5, 152, 287, 6, 20, '11.5 GeV'),
    (7.7, 147, 398, 7, 25, '7.7 GeV'),
    # SPS
    (17.3, 158, 210, 6, 20, 'SPS 17.3'),
    (8.8, 150, 380, 8, 30, 'SPS 8.8'),
    # AGS
    (4.9, 138, 540, 10, 40, 'AGS 4.9'),
]

# ============================================
# PHASE BOUNDARY - Lattice QCD parameterization
# T_c(μ_B) = T_c(0) * [1 - κ_2*(μ_B/T_c)^2 - κ_4*(μ_B/T_c)^4]
# HotQCD: T_c(0) = 156.5 MeV, κ_2 ≈ 0.0153 (Bazavov et al. 2019)
# ============================================
T_c0 = 156.5  # MeV
kappa_2 = 0.0153
kappa_4 = 0.00032  # higher order correction

# ============================================
# CRITICAL POINT (define first, needed for crossover endpoint)
# ============================================
mu_B_CEP = 470  # MeV - placed between 7.7 GeV and AGS
T_CEP = 135     # MeV
CEP_mu_err = 80  # uncertainty
CEP_T_err = 12

# Crossover line - ends at CEP
mu_B_crossover = np.linspace(0, mu_B_CEP, 100)
T_crossover_lattice = T_c0 * (1 - kappa_2 * (mu_B_crossover / T_c0)**2 
                       - kappa_4 * (mu_B_crossover / T_c0)**4)
# Adjust to smoothly hit CEP
correction = (T_crossover_lattice[-1] - T_CEP) * (mu_B_crossover / mu_B_CEP)
T_crossover = T_crossover_lattice - correction

# ============================================
# FIRST-ORDER TRANSITION LINE (model-dependent)
# Using typical estimates from various models
# ============================================
# First-order line (schematic, extends from CEP)
mu_B_first = np.linspace(mu_B_CEP, 700, 50)
# Simple parameterization that matches smoothly
T_first = T_CEP - 0.15 * (mu_B_first - mu_B_CEP)

# ============================================
# FREEZE-OUT CURVE PARAMETERIZATION
# Cleymans et al., Phys. Rev. C 73, 034905 (2006)
# T(μ_B) = a - b*μ_B^2 - c*μ_B^4
# ============================================
a = 166.5  # MeV
b = 0.139e-3  # MeV^-1
c = 0.053e-9  # MeV^-3
mu_B_fo = np.linspace(0, 600, 200)
T_fo = a - b * mu_B_fo**2 - c * mu_B_fo**4

# ============================================
# PLOTTING
# ============================================

# Background shading for phases
ax.fill_between([0, 800], [0, 0], [80, 80], color='lightblue', alpha=0.3, label='Hadron Gas')
ax.fill_between(mu_B_crossover, T_crossover, 300, color='lightyellow', alpha=0.5)
ax.axhspan(200, 300, xmin=0, xmax=1, color='#FFE4B5', alpha=0.3, label='QGP')

# Phase boundaries
# Crossover - dashed where lattice reliable, dotted where extrapolated
mu_B_reliable = 350  # MeV - lattice reliable up to here
idx_split = np.searchsorted(mu_B_crossover, mu_B_reliable)
ax.plot(mu_B_crossover[:idx_split], T_crossover[:idx_split], 'b--', lw=3, label='Crossover (Lattice)')
ax.plot(mu_B_crossover[idx_split-1:], T_crossover[idx_split-1:], 'b:', lw=3, label='Crossover (extrap.)')
# First order - solid, starts from CEP
ax.plot(mu_B_first, T_first, 'b-', lw=3, label='1st Order')

# Crossover band (uncertainty) - only where lattice is reliable
mu_B_band = np.linspace(0, 350, 80)  # Lattice reliable only up to ~350 MeV
T_cross_upper = T_c0 * (1 - (kappa_2 - 0.003) * (mu_B_band / T_c0)**2)
T_cross_lower = T_c0 * (1 - (kappa_2 + 0.003) * (mu_B_band / T_c0)**2)
ax.fill_between(mu_B_band, T_cross_lower, T_cross_upper, color='blue', alpha=0.2, 
                label='Lattice uncertainty')

# Critical point with uncertainty ellipse
cep = Ellipse((mu_B_CEP, T_CEP), width=2*CEP_mu_err, height=2*CEP_T_err,
              facecolor='magenta', edgecolor='darkmagenta', alpha=0.4, lw=2)
ax.add_patch(cep)
ax.plot(mu_B_CEP, T_CEP, 'mo', ms=12, mec='darkmagenta', mew=2, label=f'CEP (estimated)', zorder=10)

# Freeze-out curve
ax.plot(mu_B_fo, T_fo, 'k--', lw=2, alpha=0.7, label='Freeze-out curve (param.)')

# Plot freeze-out points with color coding
colors_energy = plt.cm.plasma(np.linspace(0.1, 0.9, len(freeze_out_data)))

for i, (sqrt_s, T, mu_B, T_err, mu_err, label) in enumerate(freeze_out_data):
    if 'LHC' in label:
        color, marker = 'red', 's'
    elif 'RHIC 200' in label:
        color, marker = 'orange', 'o'
    elif 'SPS' in label:
        color, marker = 'green', '^'
    elif 'AGS' in label:
        color, marker = 'purple', 'D'
    else:  # BES
        color, marker = 'blue', 'o'
    
    ax.errorbar(mu_B, T, xerr=mu_err, yerr=T_err, fmt=marker, 
                color=color, ms=8, capsize=3, capthick=1.5, 
                elinewidth=1.5, mec='black', mew=0.5, zorder=5)

# Add energy labels for select points
for sqrt_s, T, mu_B, T_err, mu_err, label in freeze_out_data:
    if 'LHC 2.76' in label:
        ax.annotate('LHC', (mu_B, T), textcoords='offset points', 
                   xytext=(-25, 8), fontsize=10, alpha=0.9, fontweight='bold')
    elif 'RHIC 200' in label:
        ax.annotate('RHIC 200', (mu_B, T), textcoords='offset points', 
                   xytext=(-20, -18), fontsize=9, alpha=0.9)
    elif '7.7 GeV' in label:
        ax.annotate('7.7 GeV', (mu_B, T), textcoords='offset points', 
                   xytext=(10, 8), fontsize=9, alpha=0.8)
    elif 'AGS' in label:
        ax.annotate('AGS 4.9 GeV', (mu_B, T), textcoords='offset points', 
                   xytext=(10, 5), fontsize=9, alpha=0.8)

# Annotations
ax.text(50, 250, 'Quark-Gluon\nPlasma', fontsize=14, fontweight='bold', color='darkorange')
ax.text(400, 60, 'Hadron Gas', fontsize=14, fontweight='bold', color='steelblue')
ax.text(150, 130, 'Crossover', fontsize=11, rotation=-5, color='blue')
ax.text(530, 85, '1st Order', fontsize=11, rotation=-12, color='blue')

# Legend for experiment markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', ms=10, mec='k', label='LHC'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', ms=10, mec='k', label='RHIC (200 GeV)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', ms=10, mec='k', label='RHIC BES'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='green', ms=10, mec='k', label='SPS'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', ms=10, mec='k', label='AGS'),
    Line2D([0], [0], color='blue', lw=3, ls='--', label='Crossover (Lattice)'),
    Line2D([0], [0], color='blue', lw=3, ls=':', label='Crossover (extrap.)'),
    Line2D([0], [0], color='blue', lw=3, ls='-', label='1st Order'),
    Line2D([0], [0], color='black', ls='--', lw=2, label='Freeze-out curve'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', ms=12, 
           mec='darkmagenta', mew=2, label='Critical point (est.)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

# Axis settings
ax.set_xlim(0, 650)
ax.set_ylim(50, 280)
ax.set_xlabel(r'Baryon Chemical Potential $\mu_B$ [MeV]', fontsize=14)
ax.set_ylabel(r'Temperature $T$ [MeV]', fontsize=14)
ax.set_title('QCD Phase Diagram with Experimental Freeze-out Points', fontsize=16, fontweight='bold')
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, alpha=0.3)

# Add reference note
ref_text = ("Data: Andronic et al., Nature 561, 321 (2018)\n"
            "Phase boundary: HotQCD, PLB 795, 15 (2019)\n"
            "CEP location highly uncertain")
ax.text(0.02, 0.02, ref_text, transform=ax.transAxes, fontsize=8, 
        verticalalignment='bottom', style='italic', alpha=0.7)

plt.tight_layout()
plt.savefig('qcd_phase_diagram.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('qcd_phase_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: qcd_phase_diagram.png and .pdf")
