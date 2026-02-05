#!/bin/bash
# GEAK-agent container entrypoint
# Sets up configuration and runs health checks

set -e

echo "🚀 GEAK-agent container initializing..."
echo ""

# Setup mini-swe-agent config from environment variables
mkdir -p /root/.config/mini-swe-agent

if [ -n "$AMD_LLM_API_KEY" ]; then
    cat > /root/.config/mini-swe-agent/.env << EOF
AMD_LLM_API_KEY='$AMD_LLM_API_KEY'
MSWEA_CONFIGURED='true'
EOF
    echo "✅ mini-swe-agent config created (model: amd/${GEAK_MODEL:-claude-sonnet-4.5})"
else
    echo "⚠️  AMD_LLM_API_KEY not set - LLM features won't work"
    echo "   Set it with: export AMD_LLM_API_KEY=your-key"
fi

# Run health checks
echo ""
echo "🔍 Running tool health checks..."

FAILED_CHECKS=0

# Check kernel-evolve (no LLM needed for 'strategies')
if kernel-evolve strategies balanced > /dev/null 2>&1; then
    echo "✅ kernel-evolve: OK"
else
    echo "❌ kernel-evolve: FAILED"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check kernel-ercs (specs doesn't need LLM)
if kernel-ercs specs > /dev/null 2>&1; then
    echo "✅ kernel-ercs: OK"
else
    echo "❌ kernel-ercs: FAILED"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check profile_kernel.py
if [ -f "/workspace/geak_agent/examples/profile_kernel.py" ]; then
    echo "✅ profile_kernel.py: Found"
else
    echo "❌ profile_kernel.py: Not found"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check geak command
if geak --help > /dev/null 2>&1; then
    echo "✅ geak (mini-swe-agent): OK"
else
    echo "❌ geak (mini-swe-agent): FAILED"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check Python environment
if python3 -c "from geak_agent.cli import main; from geakagent.optimizer import optimize_kernel" 2>/dev/null; then
    echo "✅ Python imports: OK"
else
    echo "❌ Python imports: FAILED"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Summary
echo ""
if [ $FAILED_CHECKS -eq 0 ]; then
    echo "✨ All checks passed! Container ready."
else
    echo "⚠️  $FAILED_CHECKS check(s) failed. Some tools may not work correctly."
fi
echo ""

# Execute whatever command was passed (or default CMD)
exec "$@"
