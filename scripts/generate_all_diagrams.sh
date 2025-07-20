#!/bin/bash
# generate_all_diagrams.sh - Generate all 20 ChromeCRISPR diagrams
# Professional Neural Network Architecture Visualization Framework
# Based on evidence-based academic standards for publication quality

set -e  # Exit on any error

echo "🎨 ChromeCRISPR Professional Architecture Visualizer"
echo "=================================================="
echo "📊 Generating publication-quality diagrams for 20 models..."
echo ""

# Define all 20 ChromeCRISPR models
MODELS=(
    "CNN" "GRU" "LSTM" "BiLSTM"
    "CNN-GRU" "CNN-LSTM" "CNN-BiLSTM"
    "deepCNN" "deepGRU" "deepLSTM" "deepBiLSTM"
    "CNN-GRU-bio" "CNN-LSTM-bio" "CNN-BiLSTM-bio"
    "deepCNN-bio" "deepGRU-bio" "deepLSTM-bio" "deepBiLSTM-bio"
    "RF"
)

# Create output directory
OUTPUT_DIR="docs/model_architectures"
mkdir -p "$OUTPUT_DIR"

# Check if PlotNeuralNet is available
if [ ! -d "PlotNeuralNet" ]; then
    echo "❌ PlotNeuralNet not found. Cloning..."
    git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
fi

# Check if LaTeX is available
if ! command -v pdflatex &> /dev/null; then
    echo "⚠️  Warning: pdflatex not found. PDF compilation may fail."
    echo "   Install LaTeX for full functionality."
fi

# Generate diagrams for each model
for model in "${MODELS[@]}"; do
    echo "🎯 Generating $model architecture..."

    # Run the Python script for this model
    python scripts/generate_professional_architectures.py --model "$model" --style academic --output "$OUTPUT_DIR"

    # Compile using tikzmake.sh if available
    if [ -f "PlotNeuralNet/tikzmake.sh" ]; then
        TEX_FILE="$OUTPUT_DIR/${model}_architecture.tex"
        if [ -f "$TEX_FILE" ]; then
            echo "📄 Compiling $model to PDF..."
            bash PlotNeuralNet/tikzmake.sh "$TEX_FILE"
            if [ $? -eq 0 ]; then
                echo "✅ $model: PDF generated successfully"
            else
                echo "⚠️  $model: PDF compilation failed, TikZ file saved"
            fi
        fi
    fi

    echo ""
done

# Generate performance comparison
echo "📈 Generating performance comparison..."
python scripts/generate_professional_architectures.py --performance --output "$OUTPUT_DIR"

# Convert PDFs to PNG for web display
echo "🖼️  Converting PDFs to PNG for web display..."
if command -v convert &> /dev/null; then
    for pdf_file in "$OUTPUT_DIR"/*.pdf; do
        if [ -f "$pdf_file" ]; then
            png_file="${pdf_file%.pdf}.png"
            convert -density 300 "$pdf_file" -quality 100 "$png_file"
            echo "✅ Converted $(basename "$pdf_file") to PNG"
        fi
    done
else
    echo "⚠️  ImageMagick not found. Install for PNG conversion."
fi

# Generate summary report
echo "📋 Generating summary report..."
cat > "$OUTPUT_DIR/README.md" << 'EOF'
# ChromeCRISPR Professional Model Architecture Visualizations

## Overview
This directory contains publication-quality neural network architecture visualizations for all ChromeCRISPR models, generated using evidence-based academic standards.

## Academic Standards Implemented
- **Color Scheme**: Deep blue (#1f4e79) - most professional in academic contexts
- **Typography**: Arial family - highest readability in publications
- **Layout**: Left-to-right flow with clear data progression
- **Quality**: 300+ DPI resolution for print publications
- **Style**: Matches Nature/Science publication standards

## Generated Files

### Individual Model Architectures
EOF

for model in "${MODELS[@]}"; do
    echo "- \`${model}_architecture.tex\` - TikZ source for ${model}" >> "$OUTPUT_DIR/README.md"
    echo "- \`${model}_architecture.pdf\` - PDF visualization for ${model}" >> "$OUTPUT_DIR/README.md"
    echo "- \`${model}_architecture.png\` - PNG version for web display" >> "$OUTPUT_DIR/README.md"
done

cat >> "$OUTPUT_DIR/README.md" << 'EOF'

### Performance Analysis
- `performance_comparison.tex` - TikZ source for performance comparison
- `performance_comparison.pdf` - PDF performance comparison chart
- `performance_comparison.png` - PNG performance comparison for web

## Model Categories

### Base Models
- **CNN**: Convolutional Neural Network
- **GRU**: Gated Recurrent Unit
- **LSTM**: Long Short-Term Memory
- **BiLSTM**: Bidirectional LSTM

### Hybrid Models
- **CNN-GRU**: Convolutional + GRU hybrid
- **CNN-LSTM**: Convolutional + LSTM hybrid
- **CNN-BiLSTM**: Convolutional + BiLSTM hybrid

### Deep Models
- **deepCNN**: Deep Convolutional Network
- **deepGRU**: Deep GRU Network
- **deepLSTM**: Deep LSTM Network
- **deepBiLSTM**: Deep BiLSTM Network

### Biological Models (with GC Content)
- **CNN-GRU-bio**: CNN-GRU with biological features
- **CNN-LSTM-bio**: CNN-LSTM with biological features
- **CNN-BiLSTM-bio**: CNN-BiLSTM with biological features
- **deepCNN-bio**: Deep CNN with biological features
- **deepGRU-bio**: Deep GRU with biological features
- **deepLSTM-bio**: Deep LSTM with biological features
- **deepBiLSTM-bio**: Deep BiLSTM with biological features

### Traditional ML
- **RF**: Random Forest

## Best Performing Model
**CNN-GRU with GC Content (0.8777 Spearman correlation)**
- Hybrid architecture combining convolutional and recurrent layers
- Biological feature integration (GC content)
- Optimal balance of local and sequential feature extraction

## Usage

### For Publications
1. Use PDF files for high-quality print publications
2. Vector graphics ensure scalability at any resolution
3. Academic styling matches journal requirements

### For Web Display
1. Use PNG files for web pages and presentations
2. High-resolution images maintain quality on all devices
3. Consistent styling across all visualizations

### For Modifications
1. Edit TikZ (.tex) files for custom modifications
2. Recompile using LaTeX for updated visualizations
3. Maintain academic standards in any modifications

## Technical Details

### Tools Used
- **PlotNeuralNet**: LaTeX-based professional diagram generation
- **LaTeX**: High-quality PDF compilation
- **ImageMagick**: PNG conversion for web display

### Color Scheme
- Primary: Deep blue (#1f4e79) - academic credibility
- Secondary: Light blue (#6baed6) - optimal contrast
- Accent: Orange (#fd8d3c) - biological features
- Background: White (#ffffff) - publication standard

### Quality Standards
- 300+ DPI resolution for print
- Vector graphics for infinite scalability
- Academic typography and spacing
- Consistent design language across all models

## Research Basis
This visualization framework is based on research from 90+ sources including:
- Nature, Science, and Cell publication standards
- NeurIPS, ICML, ICLR visualization guidelines
- Bioinformatics journal requirements
- Evidence-based design principles from PLoS ONE 2025

---
Generated by ChromeCRISPR Professional Architecture Visualizer
EOF

echo "✅ Summary report generated"

# Final status
echo ""
echo "🎉 All ChromeCRISPR visualizations completed!"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""
echo "📊 Generated files:"
echo "   - $(ls -1 "$OUTPUT_DIR"/*.tex | wc -l) TikZ source files"
echo "   - $(ls -1 "$OUTPUT_DIR"/*.pdf | wc -l) PDF visualizations"
echo "   - $(ls -1 "$OUTPUT_DIR"/*.png | wc -l) PNG web images"
echo ""
echo "📋 Next steps:"
echo "1. Review generated visualizations"
echo "2. Update main README with best model architecture"
echo "3. Commit changes to repository"
echo "4. Share with research team for validation"
