#!/bin/bash

# Quick start script for Original Focal-InfoNCE training (v4)

echo "============================================================"
echo " Original Focal-InfoNCE (Paper) for DOM Retrieval - v4"
echo "============================================================"
echo ""

# Step 1: Test implementation
echo "[1/3] Testing Original Focal-InfoNCE implementation..."
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
echo "[2/3] Starting training with Original Focal-InfoNCE..."
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
echo "  1. Compare with v3 (modified version)"
echo "  2. Check tensorboard logs: tensorboard --logdir ./stage1_original_focal_infonce_bge-small/runs"
echo "  3. Analyze results"
echo ""
