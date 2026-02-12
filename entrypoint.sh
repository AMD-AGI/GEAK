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

# Check kernel-profile (MetrixTool profiler)
if kernel-profile --help > /dev/null 2>&1; then
    echo "✅ kernel-profile: OK"
else
    echo "❌ kernel-profile: Not found"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check geak command
if geak --help > /dev/null 2>&1; then
    echo "✅ geak (mini-swe-agent): OK"
else
    echo "❌ geak (mini-swe-agent): FAILED"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check OpenEvolve installation
if python3 -c "from openevolve import OpenEvolve; print('OK')" > /dev/null 2>&1; then
    echo "✅ openevolve: OK"
else
    echo "❌ openevolve: Not importable"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check run_openevolve.py exists
GEAK_OE_ROOT="${GEAK_OE_ROOT:-/opt/geak-oe}"
if [ -f "${GEAK_OE_ROOT}/examples/geak_eval/run_openevolve.py" ]; then
    echo "✅ run_openevolve.py: OK (${GEAK_OE_ROOT})"
else
    echo "❌ run_openevolve.py: Not found at ${GEAK_OE_ROOT}/examples/geak_eval/"
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
