#!/bin/bash

# Make script executable
chmod +x /mnt/disk2/anhnv/rr/stage1/v4-FocalInfo-NCE/run_quick_start.sh

echo "✅ v4-FocalInfo-NCE setup complete!"
echo ""
echo "Files created:"
echo "  ✓ focal_infonce_loss.py (paper implementation)"
echo "  ✓ train_focal_infonce.py"
echo "  ✓ test_focal_infonce.py"
echo "  ✓ README.md"
echo "  ✓ COMPARISON_v3_vs_v4.md"
echo "  ✓ QUICKSTART.md"
echo "  ✓ requirements.txt"
echo "  ✓ run_quick_start.sh"
echo ""
echo "Next steps:"
echo "  1. cd /mnt/disk2/anhnv/rr/stage1/v4-FocalInfo-NCE"
echo "  2. conda activate /mnt/disk2/anhnv/rr/conda_py312"
echo "  3. python test_focal_infonce.py"
echo "  4. python train_focal_infonce.py"
echo ""
echo "Documentation:"
echo "  - QUICKSTART.md     → Quick reference"
echo "  - README.md         → Full documentation"
echo "  - COMPARISON_v3_vs_v4.md → Detailed comparison"
echo ""
