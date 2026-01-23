#!/bin/bash

# 安装 mini-swe-agent 使用 Python 3.12

echo "=== 使用 Python 3.12 安装 mini-swe-agent ==="

# 进入项目目录
cd /data/yueliu14/mini-swe-agent

# 升级 pip
echo "升级 pip..."
/usr/local/python3.12/bin/python3.12 -m pip install --upgrade pip

# 安装项目（可编辑模式）
echo "安装 mini-swe-agent..."
/usr/local/python3.12/bin/pip3.12 install -e .

# 验证安装
echo ""
echo "=== 验证安装 ==="
/usr/local/python3.12/bin/python3.12 -c "import minisweagent; print(f'mini-swe-agent version: {minisweagent.__version__}')"

echo ""
echo "=== 检查命令行工具 ==="
which mini-swe-agent
/usr/local/python3.12/bin/mini-swe-agent --help

echo ""
echo "安装完成！"
echo "你可以使用以下命令运行:"
echo "  /usr/local/python3.12/bin/mini-swe-agent"
echo "  或者"
echo "  /usr/local/python3.12/bin/mini"
