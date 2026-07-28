"""GEAK v4 bootstrap package.

GEAK v4 is not a conventional Python library — its Workflows run *inside*
Claude Code from a repo checkout. This package only exists so that

    pip install git+https://github.com/AMD-AGI/GEAK

can (a) install the Python runtime deps and (b) run a best-effort bootstrap
that clones the full repo locally and installs the Claude Code CLI.

Keep this module import-safe at build time: no third-party imports here.
"""

__version__ = "4.0.0"
