import matplotlib.pyplot as plt
import numpy as np

labels = ['Regression-only\n(3 heads)', '+Classification\n(λ=0.05)', '+Classification\n(λ=0.30)']
r2 = [0.209, 0.066, 0.047]
colors = ['#38761d', '#f4b183', '#cc0000']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, r2, color=colors, edgecolor='black', width=0.55)

# Annotate values
for bar, v in zip(bars, r2):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}',
            ha='center', fontsize=13, weight='bold')

# Drop arrows
ax.annotate('', xy=(1, 0.066), xytext=(0, 0.209),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.annotate('↓ 68%', xy=(0.5, 0.15), fontsize=13, color='red',
            weight='bold', ha='center')

ax.annotate('', xy=(2, 0.047), xytext=(0, 0.209),
            arrowprops=dict(arrowstyle='->', color='red', lw=2,
                            linestyle='dashed'))
ax.annotate('↓ 78%', xy=(1.5, 0.08), fontsize=13, color='red',
            weight='bold', ha='center')

ax.set_ylabel('Monthly R²', fontsize=13, weight='bold')
ax.set_title('Multi-Task Learning Degrades Regression Performance',
             fontsize=14, weight='bold')
ax.set_ylim(0, 0.25)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('chart_mtl_negative.png', dpi=200, bbox_inches='tight')
plt.show()