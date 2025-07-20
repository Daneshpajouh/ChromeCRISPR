#!/usr/bin/env python3
"""
Professional ChromeCRISPR Architecture Visualizations
High-quality matplotlib-based diagrams for publication
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import os

# Professional academic color scheme
COLORS = {
    'input': '#1f4e79',      # Deep blue
    'embedding': '#6baed6',  # Light blue
    'cnn': '#31a354',        # Green
    'pool': '#fd8d3c',       # Orange
    'gru': '#756bb1',        # Purple
    'dense': '#1f4e79',      # Deep blue
    'output': '#636363',     # Gray
    'gc': '#fd8d3c',         # Orange
    'border': '#2d3748',     # Dark gray
    'text': '#2d3748',       # Dark text
    'arrow': '#1f4e79'       # Arrow color
}

def create_layer_box(ax, x, y, width, height, label, color, layer_type=''):
    """Create a professional layer box"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=COLORS['border'],
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(box)
    
    # Layer label
    ax.text(x + width/2, y + height/2, label, 
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white')
    
    # Layer type
    if layer_type:
        ax.text(x + width/2, y + height + 0.05, layer_type,
                ha='center', va='bottom', fontsize=8, style='italic',
                color=COLORS['text'])

def create_arrow(ax, x1, y1, x2, y2, color=None):
    """Create a professional arrow"""
    if color is None:
        color = COLORS['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', lw=3, color=color))

def create_cnn_gru_architecture():
    """Create the best model: CNN-GRU+GC architecture"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'ChromeCRISPR: CNN-GRU+GC Architecture', 
            ha='center', va='center', fontsize=20, fontweight='bold',
            color=COLORS['text'])
    
    # Input Layer
    create_layer_box(ax, 1, 7.5, 2.5, 1, 'Input\n21-mer sgRNA', COLORS['input'], 'Sequence')
    
    # Embedding Layer
    create_layer_box(ax, 4, 7.5, 2.5, 1, 'Embedding\n64 dim', COLORS['embedding'], 'Feature Extraction')
    
    # CNN Branch
    create_layer_box(ax, 7, 8.5, 2.5, 1, 'Conv1D\n128 filters', COLORS['cnn'], 'Convolutional')
    create_layer_box(ax, 10, 8.5, 2.5, 1, 'MaxPool\nk=2', COLORS['pool'], 'Pooling')
    create_layer_box(ax, 7, 7.5, 2.5, 1, 'Conv1D\n128 filters', COLORS['cnn'], 'Convolutional')
    create_layer_box(ax, 10, 7.5, 2.5, 1, 'MaxPool\nk=2', COLORS['pool'], 'Pooling')
    
    # GRU Branch
    create_layer_box(ax, 7, 6.5, 2.5, 1, 'GRU\n384 units', COLORS['gru'], 'Recurrent')
    create_layer_box(ax, 10, 6.5, 2.5, 1, 'GRU\n384 units', COLORS['gru'], 'Recurrent')
    
    # GC Content Feature
    create_layer_box(ax, 1, 5.5, 2.5, 1, 'GC Content\nFeature', COLORS['gc'], 'Biological')
    
    # Feature Fusion
    create_layer_box(ax, 13, 7.5, 2.5, 1, 'Feature\nFusion', COLORS['dense'], 'Fusion')
    
    # Fully Connected Layers
    create_layer_box(ax, 13, 6.5, 2.5, 1, 'Dense\n128', COLORS['dense'], 'Dense')
    create_layer_box(ax, 13, 5.5, 2.5, 1, 'Dense\n64', COLORS['dense'], 'Dense')
    create_layer_box(ax, 13, 4.5, 2.5, 1, 'Dense\n32', COLORS['dense'], 'Dense')
    
    # Output Layer
    create_layer_box(ax, 13, 3.5, 2.5, 1, 'Output\nEfficiency', COLORS['output'], 'Prediction')
    
    # Arrows - Main Flow
    arrows = [
        (3.5, 8, 4, 8),      # Input to Embedding
        (6.5, 8, 7, 8.5),    # Embedding to CNN1
        (6.5, 8, 7, 7),      # Embedding to GRU1
        (9.5, 9, 10, 9),     # CNN1 to Pool1
        (12.5, 9, 13, 8),    # Pool1 to Fusion
        (12.5, 7, 13, 8),    # GRU2 to Fusion
        (13, 7, 13, 7),      # Fusion to FC1
        (13, 6, 13, 6),      # FC1 to FC2
        (13, 5, 13, 5),      # FC2 to FC3
        (13, 4, 13, 4),      # FC3 to Output
        (3.5, 6, 13, 7),     # GC to Fusion
    ]
    
    for x1, y1, x2, y2 in arrows:
        create_arrow(ax, x1, y1, x2, y2)
    
    # Performance
    ax.text(7, 2, 'Performance: Spearman = 0.876, MSE = 0.0093', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='#27AE60')
    
    plt.tight_layout()
    plt.savefig('architecture_diagrams/CNN_GRU_GC_Professional.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ CNN-GRU+GC architecture created")

def create_performance_comparison():
    """Create model performance comparison"""
    models = [
        ('CNN-GRU+GC', 0.876, COLORS['cnn']),
        ('CNN-BiLSTM+GC', 0.870, COLORS['cnn']),
        ('CNN-LSTM+GC', 0.867, COLORS['cnn']),
        ('deepCNN+GC', 0.873, COLORS['cnn']),
        ('deepBiLSTM+GC', 0.867, COLORS['gru']),
        ('deepGRU+GC', 0.867, COLORS['gru']),
        ('deepLSTM+GC', 0.860, COLORS['gru']),
        ('deepCNN', 0.869, COLORS['cnn']),
        ('deepGRU', 0.868, COLORS['gru']),
        ('deepLSTM', 0.862, COLORS['gru']),
        ('deepBiLSTM', 0.862, COLORS['gru']),
        ('LSTM+GC', 0.856, COLORS['gru']),
        ('BiLSTM+GC', 0.855, COLORS['gru']),
        ('GRU+GC', 0.840, COLORS['gru']),
        ('CNN+GC', 0.781, COLORS['cnn']),
        ('BiLSTM', 0.843, COLORS['gru']),
        ('LSTM', 0.837, COLORS['gru']),
        ('GRU', 0.837, COLORS['gru']),
        ('CNN', 0.793, COLORS['cnn']),
        ('RF', 0.755, COLORS['output'])
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Sort by performance
    models.sort(key=lambda x: x[1], reverse=True)
    
    names = [m[0] for m in models]
    scores = [m[1] for m in models]
    colors = [m[2] for m in models]
    
    bars = ax.barh(range(len(models)), scores, color=colors, alpha=0.8, edgecolor=COLORS['border'])
    
    # Customize
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Spearman Correlation', fontsize=14, fontweight='bold')
    ax.set_title('ChromeCRISPR Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.005, bar.get_y() + bar.get_height()/2, f'{score:.3f}',
               ha='left', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Highlight best model
    best_bar = bars[0]
    best_bar.set_alpha(1.0)
    best_bar.set_edgecolor('#E74C3C')
    best_bar.set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('architecture_diagrams/Model_Performance_Comparison.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Performance comparison created")

def main():
    """Create all professional visualizations"""
    print("🎨 Creating Professional ChromeCRISPR Visualizations")
    print("==================================================")
    
    # Create output directory
    os.makedirs('architecture_diagrams', exist_ok=True)
    
    # Create visualizations
    create_cnn_gru_architecture()
    create_performance_comparison()
    
    print("\n🎉 Professional visualizations completed!")
    print("📁 Output directory: architecture_diagrams/")

if __name__ == "__main__":
    main()
