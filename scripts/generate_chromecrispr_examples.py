#!/usr/bin/env python3
"""
ChromeCRISPR Model Architecture Visualization Examples
Generates professional diagrams for all 20 ChromeCRISPR models using nn_visualizer.py
"""

import subprocess
import os
import sys

# ChromeCRISPR model architectures (all 20 models)
CHROMECRISPR_MODELS = {
    # Base Models
    "CNN": "Input->Conv2D(64)->Conv2D(128)->MaxPool(2x2)->Conv2D(256)->MaxPool(2x2)->Dense(128)->Dense(1)",
    "LSTM": "Input->LSTM(128)->LSTM(64)->Dense(128)->Dense(64)->Dense(1)",
    "GRU": "Input->GRU(128)->GRU(64)->Dense(128)->Dense(64)->Dense(1)",
    "BiLSTM": "Input->LSTM(128)->LSTM(128)->Dense(128)->Dense(64)->Dense(1)",
    "BiGRU": "Input->GRU(128)->GRU(128)->Dense(128)->Dense(64)->Dense(1)",
    
    # Hybrid Models
    "CNN-LSTM": "Input->Conv2D(64)->Conv2D(128)->MaxPool(2x2)->LSTM(128)->Dense(128)->Dense(1)",
    "CNN-GRU": "Input->Conv2D(64)->Conv2D(128)->MaxPool(2x2)->GRU(128)->Dense(128)->Dense(1)",
    "CNN-BiLSTM": "Input->Conv2D(64)->Conv2D(128)->MaxPool(2x2)->LSTM(128)->LSTM(128)->Dense(128)->Dense(1)",
    "CNN-BiGRU": "Input->Conv2D(64)->Conv2D(128)->MaxPool(2x2)->GRU(128)->GRU(128)->Dense(128)->Dense(1)",
    "LSTM-GRU": "Input->LSTM(128)->GRU(64)->Dense(128)->Dense(64)->Dense(1)",
    
    # Global Context Models
    "CNN-LSTM+GC": "Input->Conv2D(64)->Conv2D(128)->MaxPool(2x2)->LSTM(128)->Attention->Dense(128)->Dense(1)",
    "CNN-GRU+GC": "Input->Conv2D(64)->Conv2D(128)->MaxPool(2x2)->GRU(128)->Attention->Dense(128)->Dense(1)",
    "CNN-BiLSTM+GC": "Input->Conv2D(64)->Conv2D(128)->MaxPool(2x2)->LSTM(128)->LSTM(128)->Attention->Dense(128)->Dense(1)",
    "CNN-BiGRU+GC": "Input->Conv2D(64)->Conv2D(128)->MaxPool(2x2)->GRU(128)->GRU(128)->Attention->Dense(128)->Dense(1)",
    "LSTM-GRU+GC": "Input->LSTM(128)->GRU(64)->Attention->Dense(128)->Dense(64)->Dense(1)",
    
    # Deep Models
    "Deep-CNN": "Input->Conv2D(64)->Conv2D(128)->Conv2D(256)->Conv2D(512)->MaxPool(2x2)->Dense(256)->Dense(128)->Dense(1)",
    "Deep-LSTM": "Input->LSTM(256)->LSTM(128)->LSTM(64)->Dense(256)->Dense(128)->Dense(64)->Dense(1)",
    "Deep-GRU": "Input->GRU(256)->GRU(128)->GRU(64)->Dense(256)->Dense(128)->Dense(64)->Dense(1)",
    "Deep-BiLSTM": "Input->LSTM(256)->LSTM(256)->LSTM(128)->LSTM(128)->Dense(256)->Dense(128)->Dense(64)->Dense(1)",
    "Deep-BiGRU": "Input->GRU(256)->GRU(256)->GRU(128)->GRU(128)->Dense(256)->Dense(128)->Dense(64)->Dense(1)"
}

def run_visualizer(architecture, model_name, style="clean", palette="blue", figsize="18,5"):
    """Run the nn_visualizer.py script for a given architecture."""
    output_file = f"../architecture_diagrams/{model_name.lower().replace('+', '_plus_').replace('-', '_')}_{style}.png"
    
    cmd = [
        "python", "nn_visualizer.py",
        "-a", architecture,
        "-s", style,
        "-o", output_file,
        "-c", palette,
        "--figsize", figsize
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {model_name} ({style}) → {output_file}")
            return True
        else:
            print(f"❌ {model_name} ({style}) failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {model_name} ({style}) error: {e}")
        return False

def main():
    """Generate visualizations for all ChromeCRISPR models."""
    print("🎨 ChromeCRISPR Model Architecture Visualizations")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("../architecture_diagrams", exist_ok=True)
    
    # Generate clean style for all models
    print("\n📊 Generating clean block style visualizations...")
    success_count = 0
    
    for model_name, architecture in CHROMECRISPR_MODELS.items():
        if run_visualizer(architecture, model_name, "clean", "blue"):
            success_count += 1
    
    print(f"\n✅ Generated {success_count}/{len(CHROMECRISPR_MODELS)} clean style visualizations")
    
    # Generate 3D style for best models
    print("\n🎯 Generating 3D style for best performing models...")
    best_models = ["CNN-GRU+GC", "CNN-BiLSTM+GC", "CNN-LSTM+GC", "Deep-CNN"]
    
    for model_name in best_models:
        if model_name in CHROMECRISPR_MODELS:
            run_visualizer(CHROMECRISPR_MODELS[model_name], model_name, "3d", "purple")
    
    # Generate node style for comparison
    print("\n🔗 Generating node style for model comparison...")
    comparison_models = ["CNN", "LSTM", "GRU", "CNN-GRU+GC"]
    
    for model_name in comparison_models:
        if model_name in CHROMECRISPR_MODELS:
            run_visualizer(CHROMECRISPR_MODELS[model_name], model_name, "node", "green", "16,8")
    
    print("\n🎉 ChromeCRISPR visualization generation complete!")
    print("📁 All files saved in 'architecture_diagrams/' directory")
    print("\n📋 Generated files:")
    
    # List generated files
    for model_name in CHROMECRISPR_MODELS.keys():
        clean_file = f"../architecture_diagrams/{model_name.lower().replace('+', '_plus_').replace('-', '_')}_clean.png"
        if os.path.exists(clean_file):
            print(f"   • {os.path.basename(clean_file)}")
    
    print("\n💡 Usage examples:")
    print("   python scripts/nn_visualizer.py -a 'Input->Conv2D(64)->GRU(128)->Dense(1)' -s clean -o my_model.png")
    print("   python scripts/nn_visualizer.py -a 'Input->LSTM(128)->Dense(10)' -s 3d -c purple -o lstm_3d.png")
    print("   python scripts/nn_visualizer.py -a 'Input->Conv2D(32)->MaxPool(2x2)->Dense(100)' -s node -c green")

if __name__ == "__main__":
    main()
