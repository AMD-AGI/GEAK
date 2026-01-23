#!/bin/bash

# 快速清理并安装

cd /data/yueliu14/mini-swe-agent

# 清理旧文件
echo "清理旧的构建文件..."
rm -rf build/ dist/ *.egg-info UNKNOWN.egg-info src/*.egg-info src/mini_swe_agent.egg-info

# 安装
echo "安装 mini-swe-agent..."
/usr/local/python3.12/bin/pip3.12 install -e . --user

echo ""
echo "完成！运行以下命令测试："
echo "  ~/.local/bin/mini --help"
