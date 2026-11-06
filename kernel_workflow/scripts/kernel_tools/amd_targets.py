"""AMD ISA query capabilities and installed-asset lookup, shared by all entry points."""
import os
from pathlib import Path

TARGETS = {
    'gfx950': {'encoding': 'gfx950-encoding.json.gz', 'layout': 'gfx950-mfma-layout.json',
               'facts': 'gfx950-isa-facts.json.gz', 'calculator': None},
    'cdna3': {'encoding': 'cdna3-encoding.json.gz', 'layout': None, 'facts': None, 'calculator': 'cdna3'},
    'rdna3': {'encoding': 'rdna3-encoding.json.gz', 'layout': None, 'facts': None, 'calculator': 'rdna3'},
    'rdna4': {'encoding': 'rdna4-encoding.json.gz', 'layout': None, 'facts': None, 'calculator': 'rdna4'},
    'cdna1': {'encoding': None, 'layout': None, 'facts': None, 'calculator': 'cdna1'},
    'cdna2': {'encoding': None, 'layout': None, 'facts': None, 'calculator': 'cdna2'},
}
ALIASES = {'cdna4': 'gfx950', 'gfx942': 'cdna3', 'mi300': 'cdna3', 'mi300x': 'cdna3',
           'mi325': 'cdna3', 'mi325x': 'cdna3', 'gfx1100': 'rdna3',
           'gfx1200': 'rdna4', 'gfx1201': 'rdna4', 'gfx90a': 'cdna2', 'gfx908': 'cdna1'}
MATRIX_COMMANDS = {'list', 'facts', 'layout', 'locate'}
ENCODING_COMMANDS = {'encoding', 'format', 'opcodes'}
FACT_COMMANDS = {'regs', 'limits', 'waitcnt', 'waits', 'mem', 'lds', 'tables', 'concepts'}
COMMANDS = MATRIX_COMMANDS | ENCODING_COMMANDS | FACT_COMMANDS | {'search', 'errata', 'modifiers', 'sparse', 'hazard'}


def canonical(arch):
    key = ALIASES.get(arch.lower(), arch.lower())
    if key not in TARGETS:
        raise ValueError(f'unknown AMD target {arch!r}; supported: {", ".join(sorted(TARGETS | ALIASES))}')
    return key


def calculator_path():
    for name in ('TILE_HW_SEED', 'TILE_HW_TOOLS', 'TILE_HW_CACHE'):
        root = os.environ.get(name)
        if root:
            path = Path(root) / 'amd_matrix_instruction_calculator/matrix_calculator.py'
            if path.is_file():
                return path
    return None


def capabilities(arch):
    import _hwdata
    key = canonical(arch)
    target = TARGETS[key]
    result = {}
    for query in sorted(COMMANDS):
        kind = ('layout' if query in MATRIX_COMMANDS | {'modifiers', 'sparse', 'hazard'}
                else 'encoding' if query in ENCODING_COMMANDS | {'search', 'errata'} else 'facts')
        database = target[kind]
        if database:
            available = _hwdata.find(database) is not None
            result[query] = {'status': 'available' if available else 'missing_asset',
                             'source': database, 'provider': 'committed_database'}
        elif query in MATRIX_COMMANDS and target['calculator']:
            path = calculator_path()
            result[query] = {'status': 'requires_calculator', 'asset_present': path is not None,
                             'source': str(path) if path else None, 'provider': 'matrix_calculator'}
        else:
            result[query] = {'status': 'not_extracted', 'source': None, 'provider': None}
    return {'requested_arch': arch, 'isa_family': key, 'queries': result,
            'note': 'Availability describes query data, not GPU numerical validation. Calculator dependencies are checked at invocation.'}
