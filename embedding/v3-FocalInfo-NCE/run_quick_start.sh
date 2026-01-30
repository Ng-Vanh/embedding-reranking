#!/bin/bash

# Quick start script for Focal-InfoNCE training

echo "============================================================"
echo " Focal-InfoNCE for DOM Retrieval - Quick Start"
echo "============================================================"
echo ""

# Step 1: Test implementation
echo "[1/4] Testing Focal-InfoNCE implementation..."
conda activate /mnt/disk2/anhnv/rr/conda_py312
python test_focal_infonce.py

if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Please check the implementation."
    exit 1
fi

echo ""
echo "✅ Tests passed!"
echo ""

# Step 2: Run training
echo "[2/4] Starting training with Focal-InfoNCE..."
echo ""
read -p "Continue with full training? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train_focal_infonce.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Training failed!"
        exit 1
    fi
    
    echo ""
    echo "✅ Training completed!"
else
    echo "Skipping training."
fi

echo ""
echo "============================================================"
echo " Quick Start Complete"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Compare with InfoNCE baseline:"
echo "     python compare_losses.py"
echo ""
echo "  2. Run hyperparameter tuning:"
echo "     python hyperparameter_tuning.py"
echo ""
echo "  3. Visualize results:"
echo "     python visualize_analysis.py"
echo ""
echo "============================================================"
