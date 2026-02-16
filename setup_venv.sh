#!/bin/bash

# Setup script for RAG System
# Ensures correct Python version is used for virtual environment

set -e

echo "🔍 Checking Python version requirements..."

# Check if python3.13 is available
if command -v python3.13 &> /dev/null; then
    PYTHON_CMD="python3.13"
    PYTHON_VERSION=$(python3.13 --version 2>&1 | awk '{print $2}')
    echo "✅ Found Python 3.13: $PYTHON_VERSION"
else
    echo "❌ ERROR: python3.13 not found!"
    echo ""
    echo "Please install Python 3.13:"
    echo "  brew install python@3.13"
    echo ""
    exit 1
fi

# Check current Python version if venv exists
if [ -d "venv" ]; then
    if [ -f "venv/bin/python" ]; then
        VENV_PYTHON_VERSION=$(venv/bin/python --version 2>&1 | awk '{print $2}')
        echo "📦 Existing venv uses Python: $VENV_PYTHON_VERSION"
        
        # Check if it's Python 3.14
        if [[ "$VENV_PYTHON_VERSION" == "3.14"* ]]; then
            echo "⚠️  WARNING: Existing venv uses Python 3.14 (incompatible)"
            echo "   Removing old venv..."
            rm -rf venv
            echo "✅ Old venv removed"
        fi
    fi
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment with Python 3.13..."
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate and upgrade pip
echo "⬆️  Upgrading pip..."
source venv/bin/activate
python --version
pip install --upgrade pip -q

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete! Virtual environment is ready."
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"





