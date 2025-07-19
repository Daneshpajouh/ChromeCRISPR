#!/usr/bin/env python3
"""
ChromeCRISPR Model Architecture Visualization Script

This script generates accurate visualizations of all model architectures
using multiple visualization tools to ensure accuracy and completeness.

Author: Amir Daneshpajouh, Megan F.A., Kay Wiese
Email: {amir_dp, mfa69, wiese}@sfu.ca
"""

import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    print("Warning: torchviz not available. Install with: pip install torchviz")

try:
    import hiddenlayer as hl
    HIDDENLAYER_AVAILABLE = True
except ImportError:
    HIDDENLAYER_AVAILABLE = False
    print("Warning: hiddenlayer not available. Install with: pip install hiddenlayer")

def create_simple_model_architectures():
    """Create simple model architectures for visualization."""
    models = {}
    
    # Simple CNN
    models['CNN'] = nn.Sequential(
        nn.Embedding(4, 64),  # 4 nucleotides
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    )
    
    # Simple GRU
    models['GRU'] = nn.Sequential(
        nn.Embedding(4, 64),
        nn.GRU(64, 128, num_layers=2, batch_first=True, dropout=0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    )
    
    # Simple LSTM
    models['LSTM'] = nn.Sequential(
        nn.Embedding(4, 64),
        nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    )
    
    # Simple BiLSTM
    models['BiLSTM'] = nn.Sequential(
        nn.Embedding(4, 64),
        nn.LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True),
        nn.Linear(256, 128),  # 256 because bidirectional
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    )
    
    # Hybrid CNN-GRU
    models['CNN-GRU'] = nn.Sequential(
        nn.Embedding(4, 64),
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.GRU(128, 128, num_layers=2, batch_first=True, dropout=0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    )
    
    return models

def visualize_with_torchviz(model, model_name, output_dir):
    """Generate model architecture diagram using torchviz."""
    if not TORCHVIZ_AVAILABLE:
        return
    
    try:
        # Create dummy input
        dummy_input = torch.randint(0, 4, (1, 21))  # 21 nucleotides
        
        # Generate computation graph
        output = model(dummy_input)
        dot = make_dot(output, params=dict(model.named_parameters()))
        
        # Save the graph
        output_path = output_dir / f"{model_name}_torchviz.png"
        dot.render(str(output_path).replace('.png', ''), format='png', cleanup=True)
        print(f"✅ TorchViz diagram saved: {output_path}")
        
    except Exception as e:
        print(f"❌ TorchViz visualization failed for {model_name}: {e}")

def visualize_with_hiddenlayer(model, model_name, output_dir):
    """Generate model architecture diagram using hiddenlayer."""
    if not HIDDENLAYER_AVAILABLE:
        return
    
    try:
        # Create dummy input
        dummy_input = torch.randint(0, 4, (1, 21))
        
        # Generate graph
        hl_graph = hl.build_graph(model, dummy_input)
        hl_graph.theme = hl.graph.THEMES["blue"].copy()
        
        # Save the graph
        output_path = output_dir / f"{model_name}_hiddenlayer.png"
        hl_graph.save(str(output_path), format='png')
        print(f"✅ HiddenLayer diagram saved: {output_path}")
        
    except Exception as e:
        print(f"❌ HiddenLayer visualization failed for {model_name}: {e}")

def create_manual_diagram(model, model_name, output_dir):
    """Create a manual architecture diagram using matplotlib."""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get model layers
        layers = list(model.children())
        
        # Define layer positions
        y_positions = np.linspace(0.1, 0.9, len(layers) + 2)
        
        # Color scheme
        colors = {
            'Embedding': '#B3D9FF',
            'Conv1d': '#FFE6CC',
            'MaxPool1d': '#FFCCCC',
            'GRU': '#CCFFCC',
            'LSTM': '#CCFFCC',
            'Linear': '#E6CCFF',
            'ReLU': '#FFFFCC',
            'Dropout': '#F0F0F0',
            'Flatten': '#E8F4FD',
            'AdaptiveAvgPool1d': '#FFCCCC'
        }
        
        # Draw layers
        for i, (layer, y_pos) in enumerate(zip(layers, y_positions[1:-1])):
            layer_name = layer.__class__.__name__
            color = colors.get(layer_name, '#FFFFFF')
            
            # Draw layer box
            rect = plt.Rectangle((0.1, y_pos - 0.03), 0.8, 0.06, 
                               facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add layer name
            ax.text(0.5, y_pos, layer_name, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Add input and output
        ax.text(0.5, y_positions[0], 'Input\n(21 nucleotides)', ha='center', va='center',
               fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor='#E8F4FD', edgecolor='black'))
        
        ax.text(0.5, y_positions[-1], 'Output\n(Efficiency Score)', ha='center', va='center',
               fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor='#FFE6E6', edgecolor='black'))
        
        # Add arrows
        for i in range(len(y_positions) - 1):
            ax.annotate('', xy=(0.5, y_positions[i+1] - 0.03), 
                       xytext=(0.5, y_positions[i] + 0.03),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Set plot properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_name} Architecture', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Save the diagram
        output_path = output_dir / f"{model_name}_manual.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Manual diagram saved: {output_path}")
        
    except Exception as e:
        print(f"❌ Manual visualization failed for {model_name}: {e}")

def main():
    """Main function to run the visualization script."""
    print("🎨 ChromeCRISPR Model Architecture Visualizer")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("docs/figures/model_architectures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create models
    models = create_simple_model_architectures()
    
    # Generate visualizations
    for model_name, model in models.items():
        print(f"\n📊 Visualizing {model_name} architecture...")
        model.eval()
        
        visualize_with_torchviz(model, model_name, output_dir)
        visualize_with_hiddenlayer(model, model_name, output_dir)
        create_manual_diagram(model, model_name, output_dir)
    
    print(f"\n🎉 Visualization complete! Check {output_dir}")

if __name__ == "__main__":
    main()
