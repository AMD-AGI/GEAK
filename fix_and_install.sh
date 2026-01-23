#!/bin/bash

# 修复并安装 mini-swe-agent 使用 Python 3.12

echo "=== 修复安装问题并使用 Python 3.12 安装 mini-swe-agent ==="

cd /data/yueliu14/mini-swe-agent

# 1. 创建 README.md（如果不存在）
if [ ! -f README.md ]; then
    echo "创建 README.md..."
    cat > README.md << 'EOF'
# mini-SWE-agent

A simple AI software engineering agent.

## Installation

```bash
pip install -e .
```

## Usage

```bash
mini-swe-agent --help
```

For more information, visit: https://mini-swe-agent.com/latest/
EOF
fi

# 2. 清理旧的构建文件和 egg-info
echo "清理旧的构建文件..."
rm -rf build/ dist/ *.egg-info UNKNOWN.egg-info
rm -rf src/*.egg-info src/mini_swe_agent.egg-info

# 3. 清理 __pycache__
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 4. 升级 pip
echo "升级 pip..."
/usr/local/python3.12/bin/python3.12 -m pip install --upgrade pip --user

# 5. 安装 setuptools 和 wheel
echo "安装构建依赖..."
/usr/local/python3.12/bin/pip3.12 install --upgrade setuptools wheel --user

# 6. 安装项目（可编辑模式）
echo "安装 mini-swe-agent..."
/usr/local/python3.12/bin/pip3.12 install -e . --user

# 7. 验证安装
echo ""
echo "=== 验证安装 ==="
/usr/local/python3.12/bin/python3.12 -c "import minisweagent; print(f'mini-swe-agent version: {minisweagent.__version__}')"

echo ""
echo "=== 检查命令行工具 ==="
ls -la ~/.local/bin/ | grep mini

echo ""
echo "安装完成！"
echo ""
echo "你可以使用以下命令运行:"
echo "  ~/.local/bin/mini-swe-agent --help"
echo "  或者"
echo "  ~/.local/bin/mini --help"
echo ""
echo "建议：将 ~/.local/bin 添加到 PATH："
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
echo "或者添加到 ~/.bashrc 使其永久生效："
echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
echo "  source ~/.bashrc"
