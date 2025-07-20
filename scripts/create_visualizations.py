#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

print("Creating visualizations...")

# Create directory
os.makedirs('architecture_diagrams', exist_ok=True)

# Simple best model visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Title
ax.text(5, 5.5, 'ChromeCRISPR: CNN-GRU+GC Architecture', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Simple boxes for layers
layers = [
    (1, 4, 'Input\n21-mer sgRNA'),
    (3, 4, 'CNN\n128 filters'),
    (5, 4, 'GRU\n384 units'),
    (7, 4, 'FC\n128→64→32'),
    (9, 4, 'Output\nEfficiency')
]

for x, y, label in layers:
    rect = plt.Rectangle((x-0.5, y-0.3), 1, 0.6, 
                        facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows
for i in range(len(layers)-1):
    x1, y1 = layers[i][0] + 0.5, layers[i][1]
    x2, y2 = layers[i+1][0] - 0.5, layers[i+1][1]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Performance
ax.text(5, 1, 'Performance: Spearman = 0.876, MSE = 0.0093', 
        ha='center', va='center', fontsize=12, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('architecture_diagrams/CNN_GRU_GC_Best_Model.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Best model visualization created")

# Performance comparison
models = ['CNN-GRU+GC', 'CNN-BiLSTM+GC', 'CNN-LSTM+GC', 'deepCNN+GC', 'deepBiLSTM+GC']
scores = [0.876, 0.870, 0.867, 0.873, 0.867]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
bars = ax.barh(models, scores, color='lightcoral', alpha=0.7)
ax.set_xlabel('Spearman Correlation')
ax.set_title('ChromeCRISPR Model Performance Comparison')

# Add value labels
for bar, score in zip(bars, scores):
    ax.text(score + 0.005, bar.get_y() + bar.get_height()/2, f'{score:.3f}',
           ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('architecture_diagrams/Model_Performance_Comparison.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Performance comparison created")
print("All visualizations saved to 'architecture_diagrams/' directory")
