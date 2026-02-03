#!/usr/bin/env python3
"""
QCD Phase Diagram with Heavy-Ion Collision Evolution Trajectories
Illustrating system evolution from initial state through QGP to freeze-out
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# Set up figure
fig, ax = plt.subplots(figsize=(11, 9))

# ============================================
# PHASE STRUCTURE (same as before)
# ============================================
T_c0 = 156.5  # MeV
kappa_2 = 0.0153
kappa_4 = 0.00032

mu_B_CEP = 470  # MeV
T_CEP = 135     # MeV
CEP_mu_err = 80
CEP_T_err = 12

# Crossover line
mu_B_crossover = np.linspace(0, mu_B_CEP, 100)
T_crossover_lattice = T_c0 * (1 - kappa_2 * (mu_B_crossover / T_c0)**2 
                       - kappa_4 * (mu_B_crossover / T_c0)**4)
correction = (T_crossover_lattice[-1] - T_CEP) * (mu_B_crossover / mu_B_CEP)
T_crossover = T_crossover_lattice - correction

# First-order line
mu_B_first = np.linspace(mu_B_CEP, 900, 50)
T_first = T_CEP - 0.12 * (mu_B_first - mu_B_CEP)

# Freeze-out curve
a, b, c = 166.5, 0.139e-3, 0.053e-9
mu_B_fo = np.linspace(0, 700, 200)
T_fo = a - b * mu_B_fo**2 - c * mu_B_fo**4

# ============================================
# COLLISION EVOLUTION TRAJECTORIES
# Approximate isentropic (s/n_B = const) paths
# Higher √s_NN → higher s/n_B → more vertical trajectory
# ============================================

def trajectory(mu_B_initial, mu_B_freezeout, T_initial, T_freezeout):
    """
    Generate a trajectory from thermalized state to freeze-out.
    Isentropic expansion: steeper at high T, flattens near phase boundary.
    """
    t = np.linspace(0, 1, 50)
    
    # Temperature drops from T_initial to T_freezeout
    T = T_initial - (T_initial - T_freezeout) * t
    
    # μ_B evolution: steep initially, flattens near freeze-out
    # Use power > 1 to get correct curvature (steep start, gradual end)
    mu_B = mu_B_initial + (mu_B_freezeout - mu_B_initial) * (t**1.8)
    
    return mu_B, T

def post_freezeout(mu_B_freezeout, T_freezeout, mu_B_ground=931, T_ground=5):
    """
    Evolution from freeze-out toward nuclear ground state.
    
    This is hadronic expansion/cooling - should stay to the RIGHT
    of the heating curve (higher μ_B at given T).
    """
    t = np.linspace(0, 1, 30)
    # Linear-ish drop in T
    T = T_freezeout - (T_freezeout - T_ground) * (t**0.8)
    # μ_B increases gradually back to ground state
    mu_B = mu_B_freezeout + (mu_B_ground - mu_B_freezeout) * (t**1.2)
    return mu_B, T

def pre_equilibrium(mu_B_initial, T_initial, mu_B_nuclei=931, T_nuclei=0):
    """
    Pre-equilibrium phase: cold nuclear matter → thermalized QGP.
    
    Physics: 
    - Initial collision: violent, T rises very fast at ~constant high μ_B
    - Then: system equilibrates, μ_B adjusts to final value
    
    This keeps the path at HIGHER T for any given μ_B compared to cooling.
    """
    t = np.linspace(0, 1, 30)
    
    # T rises rapidly (nearly linear)
    T = T_nuclei + (T_initial - T_nuclei) * t
    
    # μ_B stays near nuclear value initially, then drops sharply
    # Delayed, sharp sigmoid - μ_B doesn't start dropping until T is already high
    delay = 0.6  # don't start dropping μ_B until 60% of heating done
    sharpness = 20
    transition = 1 / (1 + np.exp(-sharpness * (t - delay)))
    mu_B = mu_B_nuclei * (1 - transition) + mu_B_initial * transition
    
    return mu_B, T

# Define trajectories for different collision energies
# (mu_B_initial, mu_B_freezeout, T_initial, T_freezeout, color, label)
# Initial μ_B increases with decreasing √s (more baryon stopping)
trajectories = [
    (1, 5, 450, 156, 'red', 'LHC 5.02 TeV'),
    (8, 24, 420, 160, 'orangered', 'RHIC 200 GeV'),
    (60, 188, 350, 158, 'orange', 'RHIC 19.6 GeV'),
    (150, 398, 280, 147, 'gold', 'RHIC 7.7 GeV'),
    (280, 540, 220, 138, 'green', 'AGS 4.9 GeV'),
]

# ============================================
# PLOTTING
# ============================================

# Background shading
ax.fill_between([0, 1000], [0, 0], [100, 100], color='lightblue', alpha=0.3)
ax.axhspan(200, 500, color='#FFE4B5', alpha=0.4)

# Phase boundaries
mu_B_reliable = 350
idx_split = np.searchsorted(mu_B_crossover, mu_B_reliable)
ax.plot(mu_B_crossover[:idx_split], T_crossover[:idx_split], 'b--', lw=2.5, alpha=0.7)
ax.plot(mu_B_crossover[idx_split-1:], T_crossover[idx_split-1:], 'b:', lw=2.5, alpha=0.7)
ax.plot(mu_B_first, T_first, 'b-', lw=2.5, alpha=0.7)

# Lattice uncertainty band
mu_B_band = np.linspace(0, 350, 80)
T_cross_upper = T_c0 * (1 - (kappa_2 - 0.003) * (mu_B_band / T_c0)**2)
T_cross_lower = T_c0 * (1 - (kappa_2 + 0.003) * (mu_B_band / T_c0)**2)
ax.fill_between(mu_B_band, T_cross_lower, T_cross_upper, color='blue', alpha=0.15)

# CEP
cep = Ellipse((mu_B_CEP, T_CEP), width=2*CEP_mu_err, height=2*CEP_T_err,
              facecolor='magenta', edgecolor='darkmagenta', alpha=0.3, lw=2)
ax.add_patch(cep)
ax.plot(mu_B_CEP, T_CEP, 'mo', ms=10, mec='darkmagenta', mew=2, zorder=10)

# Freeze-out curve (faint)
ax.plot(mu_B_fo, T_fo, 'k--', lw=1.5, alpha=0.4)

# Plot trajectories with arrows
for mu_B_init, mu_B_final, T_init, T_final, color, label in trajectories:
    
    # Pre-equilibrium phase (cold nuclei → thermalization)
    mu_B_pre, T_pre = pre_equilibrium(mu_B_init, T_init)
    ax.plot(mu_B_pre, T_pre, color=color, lw=2, alpha=0.5, ls='--',
            path_effects=[pe.Stroke(linewidth=3, foreground='white', alpha=0.3), pe.Normal()])
    
    # Main trajectory (QGP expansion: thermalization → freeze-out)
    mu_B, T = trajectory(mu_B_init, mu_B_final, T_init, T_final)
    ax.plot(mu_B, T, color=color, lw=3, alpha=0.85,
            path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])
    
    # Post freeze-out evolution (freeze-out → ground state)
    mu_B_post, T_post = post_freezeout(mu_B_final, T_final)
    ax.plot(mu_B_post, T_post, color=color, lw=2, alpha=0.4, ls='-')
    
    # Add arrows along main trajectory to show direction
    for i in [15, 35]:  # Arrow positions
        ax.annotate('', xy=(mu_B[i+2], T[i+2]), xytext=(mu_B[i], T[i]),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Arrow on post-freezeout
    ax.annotate('', xy=(mu_B_post[20], T_post[20]), xytext=(mu_B_post[15], T_post[15]),
               arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.5))
    
    # Initial thermalized state marker (filled circle) - T_max
    ax.plot(mu_B[0], T[0], 'o', color=color, ms=10, mec='black', mew=1, zorder=5)
    
    # Freeze-out marker (star)
    ax.plot(mu_B[-1], T[-1], '*', color=color, ms=14, mec='black', mew=0.5, zorder=5)
    
    # Label at initial state
    if 'LHC' in label:
        ax.text(mu_B[0] + 15, T[0] + 15, label, fontsize=10, color=color, 
                fontweight='bold', ha='left')
    elif '200' in label:
        ax.text(mu_B[0] + 15, T[0] - 5, label, fontsize=10, color=color,
                fontweight='bold', ha='left')
    elif '19.6' in label:
        ax.text(mu_B[0] + 20, T[0] + 10, label, fontsize=10, color=color,
                fontweight='bold', ha='left')
    elif '7.7' in label:
        ax.text(mu_B[0] + 25, T[0] + 5, label, fontsize=10, color=color,
                fontweight='bold', ha='left')
    else:  # AGS
        ax.text(mu_B[0] + 25, T[0] + 5, label, fontsize=10, color=color,
                fontweight='bold', ha='left')

# Cold nuclear matter starting point
ax.plot(931, 0, 'ko', ms=14, mec='black', mew=2, zorder=10)
ax.text(931, 20, 'Cold nuclear\nmatter', fontsize=9, ha='center', 
        style='italic', color='black')

# Nuclear ground state is same point - add label
ax.text(820, 45, r'$\mu_B = m_N - B$', fontsize=9, ha='center', color='gray')

# Annotations for phases and stages
ax.text(30, 380, 'Quark-Gluon\nPlasma', fontsize=14, fontweight='bold', 
        color='darkorange', alpha=0.9)
ax.text(600, 70, 'Hadron Gas', fontsize=14, fontweight='bold', 
        color='steelblue', alpha=0.9)

# Stage labels
ax.text(280, 480, r'Thermalized QGP ($T_{max}$)', fontsize=10, 
        style='italic', ha='center', color='gray')
ax.text(250, 115, 'Chemical\nFreeze-out', fontsize=10, 
        style='italic', ha='center', color='gray')
ax.text(700, 200, 'Pre-equilibrium\n(heating)', fontsize=10, 
        style='italic', ha='center', color='gray', alpha=0.8)

# Removed general expansion arrow - now shown explicitly per trajectory

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
           ms=12, mec='k', label='Cold nuclear matter'),
    Line2D([0], [0], color='gray', lw=2, ls='--', alpha=0.6, label='Pre-equilibrium'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           ms=10, mec='k', label=r'Thermalization ($T_{max}$)'),
    Line2D([0], [0], color='gray', lw=3, label='QGP expansion'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
           ms=14, mec='k', label='Chemical freeze-out'),
    Line2D([0], [0], color='gray', lw=2, alpha=0.4, label='Hadronic phase'),
    Line2D([0], [0], color='blue', lw=2.5, ls='--', label='Crossover'),
    Line2D([0], [0], color='blue', lw=2.5, ls=':', label='Crossover (extrap.)'),
    Line2D([0], [0], color='blue', lw=2.5, ls='-', label='1st Order'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', 
           ms=10, mec='darkmagenta', mew=2, label='Critical point'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

# Axis settings
ax.set_xlim(0, 1000)
ax.set_ylim(0, 500)
ax.set_xlabel(r'Baryon Chemical Potential $\mu_B$ [MeV]', fontsize=14)
ax.set_ylabel(r'Temperature $T$ [MeV]', fontsize=14)
ax.set_title('Heavy-Ion Collision Evolution in the QCD Phase Diagram', 
             fontsize=16, fontweight='bold')
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, alpha=0.3)

# Note
note = ("Trajectories are schematic\n"
        "Pre-equilibrium: T, μ_B not well-defined (far from equilibrium)")
ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9, 
        verticalalignment='bottom', style='italic', alpha=0.7)

plt.tight_layout()
plt.savefig('qcd_trajectories.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('qcd_trajectories.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: qcd_trajectories.png and .pdf")
