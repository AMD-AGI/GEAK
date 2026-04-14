#!/bin/bash
# GEAK-agent container entrypoint
# Sets up configuration and runs health checks

set -e

# Don't restrict GPUs at container level -- let geak --gpu-ids handle isolation
unset HIP_VISIBLE_DEVICES

echo "🚀 GEAK-agent container initializing..."
echo ""

# Editable mode: re-install packages from the mounted /workspace so that
# live host code is picked up instead of the baked-in site-packages copies.
if [ "${GEAK_EDITABLE}" = "1" ]; then
    echo "📝 Editable mode: re-installing packages from /workspace..."
    if ! make -C /workspace install-dev; then
        echo "❌ Editable install failed — container will use baked-in packages"
    else
        echo "✅ Editable installs complete"
    fi
    echo ""
fi

# Setup mini-swe-agent config from environment variables
mkdir -p /root/.config/mini-swe-agent

if [ -n "$AMD_LLM_API_KEY" ]; then
    cat > /root/.config/mini-swe-agent/.env << EOF
AMD_LLM_API_KEY='$AMD_LLM_API_KEY'
MSWEA_CONFIGURED='true'
EOF
    echo "✅ mini-swe-agent config created (model: amd/${GEAK_MODEL:-claude-opus-4.6})"
else
    echo "⚠️  AMD_LLM_API_KEY not set - LLM features won't work"
    echo "   Set it with: export AMD_LLM_API_KEY=your-key"
fi

# Run health checks
echo ""
echo "🔍 Running tool health checks..."

FAILED_CHECKS=0

# Check kernel-profile (MetrixTool profiler)
if kernel-profile --help > /dev/null 2>&1; then
    echo "✅ kernel-profile: OK"
else
    echo "❌ kernel-profile: Not found"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Check modular pipeline CLIs
for tool in resolve-kernel-url commandment validate-commandment \
            baseline-metrics task-generator; do
    if command -v "$tool" > /dev/null 2>&1; then
        echo "✅ ${tool}: OK"
    else
        echo "❌ ${tool}: Not found"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
done

# Check geak command
if geak --help > /dev/null 2>&1; then
    echo "✅ geak (mini-swe-agent): OK"
else
    echo "❌ geak (mini-swe-agent): FAILED"
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
