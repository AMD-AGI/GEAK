#!/usr/bin/env python3
"""Compatibility entry point; new callers use amd_isa.py with an explicit --arch."""
if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv:
        from _amd_isa_db import main
        raise SystemExit(main(sys.argv))
    from amd_isa import main
    raise SystemExit(main(default_arch='gfx950'))
else:
    # Preserve imports of the historical database query helpers.
    import sys
    import _amd_isa_db
    sys.modules[__name__] = _amd_isa_db
