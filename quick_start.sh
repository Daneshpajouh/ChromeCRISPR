#!/bin/bash

# ChromeCRISPR Quick Start Script
# This script sets up ChromeCRISPR and runs a quick demonstration

set -e  # Exit on any error

echo "🚀 ChromeCRISPR Quick Start"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.11+ is available
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.11+ required, found $PYTHON_VERSION"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python"
        else
            print_error "Python 3.11+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.11+"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "chromecrispr_env" ]; then
        $PYTHON_CMD -m venv chromecrispr_env
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source chromecrispr_env/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
    print_success "Dependencies installed"
}

# Create directories
create_directories() {
    print_status "Creating directories..."
    mkdir -p data/{raw,processed}
    mkdir -p results/{logs,plots,checkpoints,predictions}
    mkdir -p models
    mkdir -p logs
    print_success "Directories created"
}

# Generate sample data
generate_data() {
    print_status "Generating sample data..."
    $PYTHON_CMD scripts/generate_sample_data.py --n-samples 1000 --output data/processed/crispr_dataset.csv
    print_success "Sample data generated"
}

# Run quick test
run_test() {
    print_status "Running quick test..."

    # Test model creation
    $PYTHON_CMD -c "
import sys
sys.path.append('src')
from models.dynamic_model import DynamicModel
model = DynamicModel('cnn_gru', input_size=23, hidden_size=64, num_layers=1)
print('Model created successfully')
"

    # Test data loading
    $PYTHON_CMD -c "
import sys
sys.path.append('src')
from data.dataset import CRISPRDataset
dataset = CRISPRDataset('data/processed/crispr_dataset.csv', split='train')
print(f'Dataset loaded: {len(dataset)} samples')
"

    print_success "Quick test completed"
}

# Main execution
main() {
    echo ""
    print_status "Starting ChromeCRISPR setup..."

    # Check Python
    check_python

    # Create and activate virtual environment
    create_venv
    activate_venv

    # Install dependencies
    install_dependencies

    # Create directories
    create_directories

    # Generate sample data
    generate_data

    # Run quick test
    run_test

    echo ""
    print_success "ChromeCRISPR setup completed successfully!"
    echo ""
    echo "🎉 What's next?"
    echo "==============="
    echo "1. Train a model:"
    echo "   python scripts/train_model.py --config config/training_config.yaml"
    echo ""
    echo "2. Generate more data:"
    echo "   python scripts/generate_sample_data.py --n-samples 10000"
    echo ""
    echo "3. Explore model architectures:"
    echo "   ls model_architectures/"
    echo ""
    echo "4. Read documentation:"
    echo "   open docs/INSTALLATION.md"
    echo ""
    echo "Happy CRISPR prediction! 🧬"
}

# Run main function
main "$@"
