#!/bin/bash
# Setup script for creating and configuring the virtual environment

set -e  # Exit on error

echo "=== Setting up Virtual Environment for Reasoning Abstraction Project ==="

# Load required modules BEFORE creating venv
echo "Loading required modules..."
module load python/3.11.5
module load cuda/12.6
module load scipy-stack/2023b
module load arrow/21.0.0

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "Removing existing venv..."
    rm -rf venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv --system-site-packages

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Install datasets (HuggingFace)
echo "Installing datasets..."
pip install datasets

# Install local packages in editable mode
echo "Installing TransformerLens (editable)..."
pip install -e ../TransformerLens

echo "Installing pyvene (editable)..."
pip install -e ../pyvene

echo ""
echo "=== Virtual Environment Setup Complete! ==="
echo ""
echo "To activate the environment in the future, run:"
echo "  module load python/3.11.5 cuda/12.6 scipy-stack/2023b arrow/21.0.0"
echo "  source venv/bin/activate"
