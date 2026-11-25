#!/usr/bin/env python3
"""Legacy Gluon entry; implementation lives in GEAK's shared kernel tools.

Full GEAK checkout required. For a relocated entry, set GEAK_ROOT to that checkout.
Only legacy CLI `plan` keeps the gfx950 reference default; shared APIs require arch.
"""
import importlib.util
import os
import runpy
import sys
from pathlib import Path


def _shared_root():
    override = os.environ.get('GEAK_ROOT')
    roots = [Path(override).expanduser()] if override else Path(__file__).resolve().parents
    for root in roots:
        shared = root / 'kernel_workflow/scripts/kernel_tools'
        if (shared / 'probe.py').is_file() and (shared / 'amd_resources.py').is_file():
            return root.resolve(), shared.resolve()
    raise ImportError('Gluon probe requires GEAK shared kernel tools. Use a full GEAK checkout '
                      'or set GEAK_ROOT=/path/to/GEAK (inside the container when applicable).')


try:
    _root, _shared = _shared_root()
except ImportError as exc:
    if __name__ == '__main__':
        sys.exit(str(exc))
    raise
sys.path.insert(0, str(_shared))
_target = _shared / 'probe.py'
if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['plan'] and not any(a == '--arch' or a.startswith('--arch=') for a in args) \
            and not any(a in ('-h', '--help') for a in args):
        arch = 'gfx950'
        context = os.environ.get('GEAK_HARDWARE_CONTEXT')
        if context:
            sys.path.insert(0, str(_root / 'scripts'))
            try:
                import hardware_context
                arch = hardware_context.load(context)['observed']['gfx']
            except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
                sys.exit(f'Invalid GEAK_HARDWARE_CONTEXT: {exc}')
        else:
            print('note: legacy Gluon plan uses the gfx950 reference; pass --arch for another target.',
                  file=sys.stderr)
        sys.argv.extend(['--arch', arch])
    runpy.run_path(str(_target), run_name='__main__')
else:
    _spec = importlib.util.spec_from_file_location(__name__, _target)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    # Also support importlib callers that retain the object passed to exec_module.
    globals().update({k: v for k, v in vars(_module).items() if not k.startswith('__')})
