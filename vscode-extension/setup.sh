#!/bin/bash
# Setup script for mini-swe-agent VS Code extension

set -e

echo "🚀 Setting up mini-swe-agent VS Code extension..."

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

echo "✓ Node.js found: $(node --version)"

# Check for npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✓ npm found: $(npm --version)"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Install Node.js dependencies
echo ""
echo "📦 Installing Node.js dependencies..."
npm install

# Check if mini-swe-agent is installed
if ! python3 -c "import minisweagent" 2>/dev/null; then
    echo ""
    echo "⚠️  mini-swe-agent Python package not found."
    read -p "Do you want to install it from the parent directory? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing mini-swe-agent..."
        pip install -e ..
    else
        echo "⚠️  Please install mini-swe-agent manually: pip install mini-swe-agent"
    fi
else
    echo "✓ mini-swe-agent Python package found"
fi

# Compile TypeScript
echo ""
echo "🔨 Compiling TypeScript..."
npm run compile

# Check for API keys
echo ""
echo "🔑 Checking for API keys..."
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  No API keys found in environment."
    echo ""
    echo "Please set one of the following:"
    echo "  export OPENAI_API_KEY=your-key-here"
    echo "  export ANTHROPIC_API_KEY=your-key-here"
    echo ""
    echo "You can add this to your shell profile (.bashrc, .zshrc, etc.)"
else
    if [ ! -z "$OPENAI_API_KEY" ]; then
        echo "✓ OPENAI_API_KEY is set"
    fi
    if [ ! -z "$ANTHROPIC_API_KEY" ]; then
        echo "✓ ANTHROPIC_API_KEY is set"
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Open this folder in VS Code"
echo "  2. Press F5 to launch Extension Development Host"
echo "  3. Test the extension!"
echo ""
echo "For more information, see:"
echo "  - QUICKSTART.md for quick start guide"
echo "  - README.md for user documentation"
echo "  - DEVELOPMENT.md for development guide"

