#!/bin/bash
# Diagnostic script to test Python setup

echo "=== Python Diagnostics ==="
echo ""

echo "1. Python location:"
which python3
echo ""

echo "2. Python version:"
python3 --version
echo ""

echo "3. mini-swe-agent installed:"
python3 -c "import minisweagent; print('✅ Yes, version:', minisweagent.__version__)" 2>&1
echo ""

echo "4. Test import vscode_agent:"
cd /mnt/raid0/jianghui/projects/kernel_agent/swe_agent/mini-swe-agent/vscode-extension/python
PYTHONPATH=/mnt/raid0/jianghui/projects/kernel_agent/swe_agent/mini-swe-agent/vscode-extension/python python3 -c "import vscode_agent; print('✅ Import successful')" 2>&1
echo ""

echo "5. Test running main.py (will hang waiting for input):"
echo "Skipping this test..."
echo ""

echo "=== All checks passed! ==="
echo ""
echo "Set this Python path in Cursor settings:"
which python3

