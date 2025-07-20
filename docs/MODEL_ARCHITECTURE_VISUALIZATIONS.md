# ChromeCRISPR Model Architecture Visualizations

This document describes the model architecture visualizations available in the ChromeCRISPR repository and the tools used to generate them.

## 📊 Available Visualizations

### Generated Diagrams

The following model architecture diagrams have been generated and are available in `docs/figures/model_architectures/`:

1. **CNN_manual.png** - Convolutional Neural Network architecture
2. **GRU_manual.png** - Gated Recurrent Unit architecture
3. **LSTM_manual.png** - Long Short-Term Memory architecture
4. **BiLSTM_manual.png** - Bidirectional LSTM architecture
5. **CNN-GRU_manual.png** - Hybrid CNN-GRU architecture

### Visualization Types

Each model has been visualized using the following approaches:

#### 1. Manual Matplotlib Diagrams
- **Tool**: Custom matplotlib implementation
- **Features**:
  - Color-coded layer types
  - Detailed layer information
  - Clear data flow arrows
  - Professional appearance
- **Status**: Available for all models

#### 2. TorchViz Computation Graphs
- **Tool**: torchviz library
- **Features**:
  - Automatic computation graph generation
  - Shows data flow and operations
  - Detailed parameter information
- **Status**: ⚠️ **Partially working** (requires model adjustments)

#### 3. HiddenLayer Architecture Diagrams
- **Tool**: hiddenlayer library
- **Features**:
  - Clean architecture diagrams
  - Layer-by-layer visualization
  - Professional styling
- **Status**: ⚠️ **Not installed** (can be installed with `pip install hiddenlayer`)

## 🛠️ Visualization Tools

### Installed Tools

1. **Matplotlib** 
   - Used for manual architecture diagrams
   - Provides full control over visualization
   - Professional quality output

2. **TorchViz** 
   - Installed: `pip install torchviz`
   - Generates computation graphs
   - Shows detailed data flow

3. **HiddenLayer** ⚠️
   - Not currently installed
   - Install with: `pip install hiddenlayer`
   - Creates clean architecture diagrams

### Additional Tools Available

4. **TensorBoard** (Optional)
   - PyTorch integration available
   - Real-time training visualization
   - Model graph visualization

5. **Netron** (Optional)
   - Web-based model viewer
   - Supports multiple formats
   - Interactive exploration

## 🎨 Diagram Features

### Color Coding
- **Input/Output**: Light blue/red
- **Embedding**: Blue
- **Convolutional**: Orange
- **Recurrent (GRU/LSTM)**: Green
- **Linear**: Purple
- **Pooling**: Light red
- **Activation**: Yellow
- **Dropout**: Gray

### Layer Information
- Layer names and types
- Input/output dimensions
- Key parameters (kernel size, hidden size, etc.)
- Dropout rates

### Data Flow
- Clear arrows showing data progression
- Input: 21 nucleotides
- Output: Efficiency score (0-1)

## 📋 Model Architectures Visualized

### Base Models

#### 1. CNN (Convolutional Neural Network)
- **Architecture**: Embedding → Conv1d → ReLU → MaxPool → Conv1d → ReLU → AdaptiveAvgPool → Linear → Linear
- **Performance**: 0.792 Spearman correlation
- **Use Case**: Local pattern recognition in sequences

#### 2. GRU (Gated Recurrent Unit)
- **Architecture**: Embedding → GRU → Linear → ReLU → Dropout → Linear
- **Performance**: 0.837 Spearman correlation
- **Use Case**: Sequential data modeling

#### 3. LSTM (Long Short-Term Memory)
- **Architecture**: Embedding → LSTM → Linear → ReLU → Dropout → Linear
- **Performance**: 0.837 Spearman correlation
- **Use Case**: Long-range dependencies

#### 4. BiLSTM (Bidirectional LSTM)
- **Architecture**: Embedding → Bidirectional LSTM → Linear → ReLU → Dropout → Linear
- **Performance**: 0.843 Spearman correlation
- **Use Case**: Bidirectional sequence processing

### Hybrid Models

#### 5. CNN-GRU (Convolutional + GRU)
- **Architecture**: Embedding → Conv1d → ReLU → MaxPool → GRU → Linear → ReLU → Dropout → Linear
- **Performance**: 0.876 Spearman correlation ⭐ **BEST**
- **Use Case**: Local + global sequence features

## 🔧 Usage Instructions

### Generating Visualizations

```bash
# Run the visualization script
python scripts/visualize_model_architectures.py
```

### Installing Additional Tools

```bash
# Install hiddenlayer for additional diagrams
pip install hiddenlayer

# Install tensorboard for training visualization
pip install tensorboard

# Install netron for model exploration
pip install netron
```

### Viewing Diagrams

```bash
# Open the figures directory
open docs/figures/model_architectures/

# Or use a specific viewer
# For PNG files: any image viewer
# For interactive exploration: netron
```

## 📈 Visualization Quality

### Manual Diagrams
- **Resolution**: 300 DPI
- **Format**: PNG
- **Size**: 12x8 inches
- **Quality**: Publication-ready

### TorchViz Graphs
- **Format**: PNG
- **Detail Level**: High (computation graph)
- **Use Case**: Technical analysis

### HiddenLayer Diagrams
- **Format**: PNG
- **Style**: Clean, professional
- **Use Case**: Presentations, papers

## 🎯 Applications

### Research Papers
- Use manual diagrams for clear architecture presentation
- Include in methodology sections
- Reference in figure captions

### Presentations
- High-resolution PNG files
- Color-coded for easy understanding
- Professional appearance

### Documentation
- Clear architecture overview
- Layer-by-layer explanation
- Parameter details

### Education
- Step-by-step architecture understanding
- Visual learning aid
- Model comparison

## 🔄 Future Enhancements

### Planned Improvements
1. **Fix TorchViz integration** - Resolve tensor dimension issues
2. **Add HiddenLayer diagrams** - Install and configure
3. **Interactive visualizations** - Web-based exploration
4. **3D architecture diagrams** - Advanced visualization
5. **Training visualization** - Real-time monitoring

### Additional Tools to Consider
1. **PlotNeuralNet** - LaTeX-based diagrams
2. **Graphviz** - Graph visualization
3. **D3.js** - Interactive web visualizations
4. **Keras Visualization** - Alternative approach

## 📞 Support

For questions about model architecture visualizations:

- **Primary Contact**: Amir Daneshpajouh (amir_dp@sfu.ca)
- **All Authors**: {amir_dp, mfa69, wiese}@sfu.ca

## 📄 Citation

When using these visualizations in your work, please cite:

```bibtex
@article{chromecrispr2024,
  title={ChromeCRISPR: Deep Learning Framework for CRISPR Guide RNA Efficiency Prediction},
  author={Daneshpajouh, Amir and Megan, F. A. and Wiese, Kay},
  journal={Journal Name},
  year={2024}
}
```

---

**Note**: All visualizations are generated from the actual model implementations in the ChromeCRISPR repository, ensuring accuracy and consistency with the published research.
