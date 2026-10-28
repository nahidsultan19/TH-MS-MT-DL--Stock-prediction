import matplotlib.pyplot as plt

years = [2022, 2023, 2024, 2025]
diracc = [65.4, 70.5, 66.8, 65.5]
r2 = [0.158, 0.170, 0.217, 0.220]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(years, diracc, 'o-', color='#38761d', lw=2.5,
         markersize=10, label='Directional Accuracy (%)')
ax2.plot(years, r2, 's--', color='#0b5394', lw=2.5,
         markersize=10, label='R²')

ax1.axhline(50, color='red', linestyle=':', lw=1)
ax1.set_xlabel('Test Year', fontsize=13, weight='bold')
ax1.set_ylabel('Directional Accuracy (%)', color='#38761d', fontsize=13, weight='bold')
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
plt.savefig('chart_per_year.png', dpi=200, bbox_inches='tight')
plt.show()