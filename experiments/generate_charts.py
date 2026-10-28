"""
Generate defense-presentation charts from saved metrics.
Run: python -m experiments.generate_charts
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────
# Chart 1: Horizon Effect (from Table 5.1)
# ─────────────────────────────────────────────────
horizons = ['Daily', 'Weekly', 'Monthly']
r2      = [-0.0003, 0.003, 0.209]
diracc  = [48.5, 55.2, 67.6]
colors  = ['#cccccc', '#f4b183', '#38761d']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(horizons, r2, color=colors, edgecolor='black')
axes[0].set_title('R² by Prediction Horizon', fontsize=14, weight='bold')
axes[0].set_ylabel('R²', fontsize=12)
axes[0].axhline(0, color='black', linewidth=0.5)
for i, v in enumerate(r2):
    axes[0].text(i, v + 0.005 if v > 0 else v - 0.015, f'{v:.3f}',
                 ha='center', fontsize=11, weight='bold')

axes[1].bar(horizons, diracc, color=colors, edgecolor='black')
axes[1].set_title('Directional Accuracy by Horizon', fontsize=14, weight='bold')
axes[1].set_ylabel('DirAcc (%)', fontsize=12)
axes[1].axhline(50, color='red', linestyle='--', linewidth=1,
                label='Random baseline (50%)')
axes[1].legend()
axes[1].set_ylim(40, 75)
for i, v in enumerate(diracc):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/chart_horizon_effect.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ chart_horizon_effect.png")

# ─────────────────────────────────────────────────
# Chart 2: MTL Negative Result
# ─────────────────────────────────────────────────
labels = ['Regression-only\n(3 heads)', '+Classification\n(λ=0.05)', '+Classification\n(λ=0.30)']
mtl_r2 = [0.209, 0.066, 0.047]
mtl_colors = ['#38761d', '#f4b183', '#cc0000']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, mtl_r2, color=mtl_colors, edgecolor='black', width=0.55)
for bar, v in zip(bars, mtl_r2):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}',
            ha='center', fontsize=13, weight='bold')

ax.annotate('', xy=(1, 0.066), xytext=(0, 0.209),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.annotate('↓ 68%', xy=(0.5, 0.15), fontsize=13, color='red',
            weight='bold', ha='center')
ax.annotate('', xy=(2, 0.047), xytext=(0, 0.209),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='dashed'))
ax.annotate('↓ 78%', xy=(1.5, 0.08), fontsize=13, color='red',
            weight='bold', ha='center')

ax.set_ylabel('Monthly R²', fontsize=13, weight='bold')
ax.set_title('Multi-Task Learning Degrades Regression Performance',
             fontsize=14, weight='bold')
ax.set_ylim(0, 0.25)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/chart_mtl_negative.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ chart_mtl_negative.png")

# ─────────────────────────────────────────────────
# Chart 3: Architecture Ablation
# ─────────────────────────────────────────────────
variants = ['All CNN-LSTM', 'Hybrid TCL', 'Hybrid Full', 'Hybrid CLT', 'All Transformer']
ablation_r2 = [0.164, 0.164, 0.077, 0.069, 0.056]
ablation_colors = ['#38761d', '#38761d', '#f4b183', '#f4b183', '#f4b183']

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(variants, ablation_r2, color=ablation_colors, edgecolor='black')
ax.invert_yaxis()
for bar, v in zip(bars, ablation_r2):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
            va='center', fontsize=12, weight='bold')
ax.set_xlabel('Monthly R²', fontsize=13, weight='bold')
ax.set_title('Architecture Ablation: Monthly Branch Impact',
             fontsize=13, weight='bold')
ax.set_xlim(0, 0.20)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/chart_ablation.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ chart_ablation.png")

# ─────────────────────────────────────────────────
# Chart 4: Per-Year Robustness
# ─────────────────────────────────────────────────
years = [2022, 2023, 2024, 2025]
year_diracc = [65.4, 70.5, 66.8, 65.5]
year_r2 = [0.158, 0.170, 0.217, 0.220]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(years, year_diracc, 'o-', color='#38761d', lw=2.5,
         markersize=10, label='Directional Accuracy (%)')
ax2.plot(years, year_r2, 's--', color='#0b5394', lw=2.5,
         markersize=10, label='R²')
ax1.axhline(50, color='red', linestyle=':', lw=1)
ax1.set_xlabel('Test Year', fontsize=13, weight='bold')
ax1.set_ylabel('Directional Accuracy (%)', color='#38761d',
               fontsize=13, weight='bold')
ax2.set_ylabel('R²', color='#0b5394', fontsize=13, weight='bold')
ax1.set_ylim(55, 80)
ax2.set_ylim(0.10, 0.30)
ax1.set_xticks(years)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
plt.title('Monthly Model Performance by Test Year (Robustness)',
          fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/chart_per_year.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ chart_per_year.png")

print(f"\n✓ All 4 charts saved to {OUT_DIR}/")