"""GEAK does not depend on Hyperloom.

Hyperloom may drive GEAK (it launches runs and hands them a seed config), and
comments may say so. The other direction is closed: no GEAK code, script or
agent-facing instruction may import, execute or require a Hyperloom checkout or
a file only a Hyperloom tool produces. The run report used to render through
Hyperloom's tools when ``HYPERLOOM_SRC`` pointed at a checkout; it is now built
entirely in this repository (``interface/geak_report.py``).
"""

import os
import re
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_THIS = os.path.abspath(__file__)

# Each is a way GEAK could reach into Hyperloom: locate a checkout, import or run
# one of its modules, or read the files its report tools write.
_REACH = re.compile(
    r"HYPERLOOM_SRC"
    r"|hyperloom\.inference_optimizer"
    r"|^\s*(?:import|from)\s+hyperloom\b"
    r"|dump_geak_call_report|render_geak_html_report"
    r"|GEAK_(?:LLM|HTML)_REPORT_CMD",
    re.M)
_EXTS = (".py", ".js", ".sh", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg", ".txt")
_SKIP_DIRS = {".git", "build", "dist", "node_modules", "__pycache__", "exp"}


class TestNoHyperloomDependency(unittest.TestCase):
    def test_nothing_in_geak_reaches_into_hyperloom(self):
        offenders = []
        for root, dirs, files in os.walk(_REPO):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info")]
            for name in files:
                path = os.path.join(root, name)
                if not name.endswith(_EXTS) or os.path.abspath(path) == _THIS:
                    continue
                try:
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    continue
                for m in _REACH.finditer(text):
                    line = text.count("\n", 0, m.start()) + 1
                    offenders.append("%s:%d: %s" % (os.path.relpath(path, _REPO), line, m.group(0).strip()))
        self.assertEqual(offenders, [], "GEAK must not depend on Hyperloom:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
