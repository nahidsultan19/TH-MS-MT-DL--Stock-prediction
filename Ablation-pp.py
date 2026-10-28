import matplotlib.pyplot as plt

variants = ['All CNN-LSTM', 'Hybrid TCL', 'Hybrid Full', 'Hybrid CLT', 'All Transformer']
r2 = [0.164, 0.164, 0.077, 0.069, 0.056]
colors = ['#38761d', '#38761d', '#f4b183', '#f4b183', '#f4b183']

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(variants, r2, color=colors, edgecolor='black')
ax.invert_yaxis()

for bar, v in zip(bars, r2):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
            va='center', fontsize=12, weight='bold')

ax.set_xlabel('Monthly R²', fontsize=13, weight='bold')
ax.set_title('Architecture Ablation: CNN-LSTM vs. Transformer Monthly Branch',
             fontsize=13, weight='bold')
ax.set_xlim(0, 0.20)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('chart_ablation.png', dpi=200, bbox_inches='tight')
plt.show()