import matplotlib.pyplot as plt
import numpy as np

horizons = ['Daily', 'Weekly', 'Monthly']
r2 = [-0.0003, 0.003, 0.209]
diracc = [48.5, 55.2, 67.6]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# R² chart
colors = ['#cccccc', '#f4b183', '#38761d']
axes[0].bar(horizons, r2, color=colors, edgecolor='black')
axes[0].set_title('R² by Prediction Horizon', fontsize=14, weight='bold')
axes[0].set_ylabel('R²', fontsize=12)
axes[0].axhline(0, color='black', linewidth=0.5)
for i, v in enumerate(r2):
    axes[0].text(i, v + 0.005 if v > 0 else v - 0.01, f'{v:.3f}',
                 ha='center', fontsize=11, weight='bold')

# DirAcc chart
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
plt.savefig('chart_horizon_effect.png', dpi=200, bbox_inches='tight')
plt.show()