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
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.cnn_model import CNNModel
from models.rnn_models import GRUModel, LSTMModel, BiLSTMModel
from models.hybrid_models import CNNGRUModel, CNNLSTMModel, CNNBiLSTMModel

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

class ModelArchitectureVisualizer:
    """Comprehensive model architecture visualization tool."""

    def __init__(self, output_dir: str = "docs/figures/model_architectures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model configurations
        self.model_configs = {
            'input_size': 21,
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'num_classes': 1
        }

        # Color scheme for different layer types
        self.colors = {
            'input': '#E8F4FD',
            'embedding': '#B3D9FF',
            'conv': '#FFE6CC',
            'pool': '#FFCCCC',
            'rnn': '#CCFFCC',
            'linear': '#E6CCFF',
            'output': '#FFE6E6',
            'batch_norm': '#FFFFCC',
            'dropout': '#F0F0F0'
        }

    def create_all_models(self) -> Dict[str, nn.Module]:
        """Create all model architectures."""
        models = {}

        # Base models
        models['CNN'] = CNNModel(
            input_size=self.model_configs['input_size'],
            embedding_dim=self.model_configs['embedding_dim'],
            hidden_dim=self.model_configs['hidden_dim'],
            dropout=self.model_configs['dropout']
        )

        models['GRU'] = GRUModel(
            input_size=self.model_configs['input_size'],
            embedding_dim=self.model_configs['embedding_dim'],
            hidden_dim=self.model_configs['hidden_dim'],
            num_layers=self.model_configs['num_layers'],
            dropout=self.model_configs['dropout']
        )

        models['LSTM'] = LSTMModel(
            input_size=self.model_configs['input_size'],
            embedding_dim=self.model_configs['embedding_dim'],
            hidden_dim=self.model_configs['hidden_dim'],
            num_layers=self.model_configs['num_layers'],
            dropout=self.model_configs['dropout']
        )

        models['BiLSTM'] = BiLSTMModel(
            input_size=self.model_configs['input_size'],
            embedding_dim=self.model_configs['embedding_dim'],
            hidden_dim=self.model_configs['hidden_dim'],
            num_layers=self.model_configs['num_layers'],
            dropout=self.model_configs['dropout']
        )

        # Hybrid models
        models['CNN-GRU'] = CNNGRUModel(
            input_size=self.model_configs['input_size'],
            embedding_dim=self.model_configs['embedding_dim'],
            hidden_dim=self.model_configs['hidden_dim'],
            num_layers=self.model_configs['num_layers'],
            dropout=self.model_configs['dropout']
        )

        models['CNN-LSTM'] = CNNLSTMModel(
            input_size=self.model_configs['input_size'],
            embedding_dim=self.model_configs['embedding_dim'],
            hidden_dim=self.model_configs['hidden_dim'],
            num_layers=self.model_configs['num_layers'],
            dropout=self.model_configs['dropout']
        )

        models['CNN-BiLSTM'] = CNNBiLSTMModel(
            input_size=self.model_configs['input_size'],
            embedding_dim=self.model_configs['embedding_dim'],
            hidden_dim=self.model_configs['hidden_dim'],
            num_layers=self.model_configs['num_layers'],
            dropout=self.model_configs['dropout']
        )

        return models

    def visualize_with_torchviz(self, model: nn.Module, model_name: str) -> None:
        """Generate model architecture diagram using torchviz."""
        if not TORCHVIZ_AVAILABLE:
            return

        try:
            # Create dummy input
            dummy_input = torch.randint(0, 4, (1, self.model_configs['input_size']))

            # Generate computation graph
            output = model(dummy_input)
            dot = make_dot(output, params=dict(model.named_parameters()))

            # Save the graph
            output_path = self.output_dir / f"{model_name}_torchviz.png"
            dot.render(str(output_path).replace('.png', ''), format='png', cleanup=True)
            print(f"✅ TorchViz diagram saved: {output_path}")

        except Exception as e:
            print(f"❌ TorchViz visualization failed for {model_name}: {e}")

    def visualize_with_hiddenlayer(self, model: nn.Module, model_name: str) -> None:
        """Generate model architecture diagram using hiddenlayer."""
        if not HIDDENLAYER_AVAILABLE:
            return

        try:
            # Create dummy input
            dummy_input = torch.randint(0, 4, (1, self.model_configs['input_size']))

            # Generate graph
            hl_graph = hl.build_graph(model, dummy_input)
            hl_graph.theme = hl.graph.THEMES["blue"].copy()

            # Save the graph
            output_path = self.output_dir / f"{model_name}_hiddenlayer.png"
            hl_graph.save(str(output_path), format='png')
            print(f"✅ HiddenLayer diagram saved: {output_path}")

        except Exception as e:
            print(f"❌ HiddenLayer visualization failed for {model_name}: {e}")

    def create_manual_architecture_diagram(self, model: nn.Module, model_name: str) -> None:
        """Create a manual architecture diagram using matplotlib."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Get model layers
            layers = list(model.children())

            # Define layer positions
            y_positions = np.linspace(0.1, 0.9, len(layers) + 2)

            # Draw layers
            for i, (layer, y_pos) in enumerate(zip(layers, y_positions[1:-1])):
                layer_name = layer.__class__.__name__
                color = self._get_layer_color(layer_name)

                # Draw layer box
                rect = plt.Rectangle((0.1, y_pos - 0.03), 0.8, 0.06,
                                   facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)

                # Add layer name
                ax.text(0.5, y_pos, layer_name, ha='center', va='center',
                       fontsize=10, fontweight='bold')

                # Add layer details
                details = self._get_layer_details(layer)
                if details:
                    ax.text(0.5, y_pos - 0.02, details, ha='center', va='center',
                           fontsize=8, style='italic')

            # Add input and output
            ax.text(0.5, y_positions[0], 'Input\n(21 nucleotides)', ha='center', va='center',
                   fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
                   facecolor=self.colors['input'], edgecolor='black'))

            ax.text(0.5, y_positions[-1], 'Output\n(Efficiency Score)', ha='center', va='center',
                   fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
                   facecolor=self.colors['output'], edgecolor='black'))

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
            output_path = self.output_dir / f"{model_name}_manual.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Manual diagram saved: {output_path}")

        except Exception as e:
            print(f"❌ Manual visualization failed for {model_name}: {e}")

    def _get_layer_color(self, layer_name: str) -> str:
        """Get color for layer type."""
        if 'Conv' in layer_name:
            return self.colors['conv']
        elif 'Pool' in layer_name:
            return self.colors['pool']
        elif 'GRU' in layer_name or 'LSTM' in layer_name:
            return self.colors['rnn']
        elif 'Linear' in layer_name:
            return self.colors['linear']
        elif 'BatchNorm' in layer_name:
            return self.colors['batch_norm']
        elif 'Dropout' in layer_name:
            return self.colors['dropout']
        elif 'Embedding' in layer_name:
            return self.colors['embedding']
        else:
            return '#FFFFFF'

    def _get_layer_details(self, layer: nn.Module) -> str:
        """Get detailed information about a layer."""
        layer_name = layer.__class__.__name__

        if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
            return f"{layer.in_channels}→{layer.out_channels}"
        elif hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
            return f"{layer.in_features}→{layer.out_features}"
        elif hasattr(layer, 'hidden_size'):
            return f"hidden={layer.hidden_size}"
        elif hasattr(layer, 'num_features'):
            return f"features={layer.num_features}"
        elif hasattr(layer, 'p'):
            return f"p={layer.p}"
        else:
            return ""

    def create_comparison_diagram(self, models: Dict[str, nn.Module]) -> None:
        """Create a comparison diagram of all model architectures."""
        try:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()

            model_names = list(models.keys())

            for i, (model_name, model) in enumerate(models.items()):
                if i >= len(axes):
                    break

                ax = axes[i]
                layers = list(model.children())

                # Simple layer representation
                y_positions = np.linspace(0.1, 0.9, len(layers) + 2)

                # Draw simplified architecture
                for j, (layer, y_pos) in enumerate(zip(layers, y_positions[1:-1])):
                    layer_name = layer.__class__.__name__
                    color = self._get_layer_color(layer_name)

                    rect = plt.Rectangle((0.1, y_pos - 0.02), 0.8, 0.04,
                                       facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)

                    ax.text(0.5, y_pos, layer_name, ha='center', va='center',
                           fontsize=8, fontweight='bold')

                # Add input/output
                ax.text(0.5, y_positions[0], 'Input', ha='center', va='center',
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.2",
                       facecolor=self.colors['input'], edgecolor='black'))

                ax.text(0.5, y_positions[-1], 'Output', ha='center', va='center',
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.2",
                       facecolor=self.colors['output'], edgecolor='black'))

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(model_name, fontsize=12, fontweight='bold')
                ax.axis('off')

            # Hide unused subplots
            for i in range(len(model_names), len(axes)):
                axes[i].axis('off')

            plt.suptitle('ChromeCRISPR Model Architecture Comparison',
                        fontsize=16, fontweight='bold', y=0.95)
            plt.tight_layout()

            # Save the comparison
            output_path = self.output_dir / "model_architecture_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Comparison diagram saved: {output_path}")

        except Exception as e:
            print(f"❌ Comparison diagram failed: {e}")

    def generate_all_visualizations(self) -> None:
        """Generate all model architecture visualizations."""
        print("🎨 Generating ChromeCRISPR Model Architecture Visualizations...")
        print("=" * 60)

        # Create all models
        models = self.create_all_models()

        # Generate individual visualizations
        for model_name, model in models.items():
            print(f"\n📊 Visualizing {model_name} architecture...")

            # Set model to evaluation mode
            model.eval()

            # Generate different types of visualizations
            self.visualize_with_torchviz(model, model_name)
            self.visualize_with_hiddenlayer(model, model_name)
            self.create_manual_architecture_diagram(model, model_name)

        # Generate comparison diagram
        print(f"\n📊 Creating model architecture comparison...")
        self.create_comparison_diagram(models)

        # Create summary report
        self.create_visualization_report(models)

        print("\n" + "=" * 60)
        print("✅ All model architecture visualizations completed!")
        print(f"📁 Output directory: {self.output_dir}")
        print("\n📋 Available visualizations:")
        print("   • Individual model diagrams (torchviz, hiddenlayer, manual)")
        print("   • Model architecture comparison")
        print("   • Visualization summary report")

    def create_visualization_report(self, models: Dict[str, nn.Module]) -> None:
        """Create a summary report of all visualizations."""
        try:
            report_path = self.output_dir / "visualization_report.md"

            with open(report_path, 'w') as f:
                f.write("# ChromeCRISPR Model Architecture Visualizations\n\n")
                f.write("This report summarizes all model architecture visualizations generated for ChromeCRISPR.\n\n")

                f.write("## Generated Visualizations\n\n")

                for model_name in models.keys():
                    f.write(f"### {model_name}\n")
                    f.write(f"- `{model_name}_torchviz.png` - TorchViz computation graph\n")
                    f.write(f"- `{model_name}_hiddenlayer.png` - HiddenLayer architecture diagram\n")
                    f.write(f"- `{model_name}_manual.png` - Manual matplotlib diagram\n\n")

                f.write("## Comparison Diagrams\n\n")
                f.write("- `model_architecture_comparison.png` - Side-by-side comparison of all models\n\n")

                f.write("## Visualization Tools Used\n\n")
                f.write("1. **TorchViz** - Generates computation graphs showing data flow\n")
                f.write("2. **HiddenLayer** - Creates clean architecture diagrams\n")
                f.write("3. **Matplotlib** - Custom manual diagrams with detailed layer information\n\n")

                f.write("## Model Categories\n\n")
                f.write("### Base Models\n")
                f.write("- CNN: Convolutional Neural Network\n")
                f.write("- GRU: Gated Recurrent Unit\n")
                f.write("- LSTM: Long Short-Term Memory\n")
                f.write("- BiLSTM: Bidirectional LSTM\n\n")

                f.write("### Hybrid Models\n")
                f.write("- CNN-GRU: CNN + GRU combination\n")
                f.write("- CNN-LSTM: CNN + LSTM combination\n")
                f.write("- CNN-BiLSTM: CNN + BiLSTM combination\n\n")

                f.write("## Usage\n\n")
                f.write("These visualizations can be used in:\n")
                f.write("- Research papers and presentations\n")
                f.write("- Documentation and tutorials\n")
                f.write("- Model comparison and analysis\n")
                f.write("- Educational materials\n\n")

                f.write("## Authors\n\n")
                f.write("- Amir Daneshpajouh (amir_dp@sfu.ca)\n")
                f.write("- Megan F.A. (mfa69@sfu.ca)\n")
                f.write("- Kay Wiese (wiese@sfu.ca)\n")

            print(f"✅ Visualization report saved: {report_path}")

        except Exception as e:
            print(f"❌ Visualization report failed: {e}")


def main():
    """Main function to run the visualization script."""
    print("🎨 ChromeCRISPR Model Architecture Visualizer")
    print("=" * 50)

    # Create visualizer
    visualizer = ModelArchitectureVisualizer()

    # Generate all visualizations
    visualizer.generate_all_visualizations()

    print("\n🎉 Visualization complete! Check the docs/figures/model_architectures/ directory.")


if __name__ == "__main__":
    main()
