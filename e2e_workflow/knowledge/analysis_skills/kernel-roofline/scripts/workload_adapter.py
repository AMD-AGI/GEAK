"""Build one timed callable. GPU imports and task imports occur only on demand."""
import importlib.util
import hashlib
import inspect
import os
import sys
from pathlib import Path


def load_module(path, name):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_path(value):
    path = getattr(getattr(value, '__code__', None), 'co_filename', '') or getattr(value, '__file__', '')
    if not path:
        try:
            path = inspect.getfile(value)
        except (TypeError, OSError):
            pass
    return str(Path(path).resolve()) if path else None


def arm_target(ut, arm):
    prefix = '_BASE' if arm == 'baseline' else '_CAND'
    # Module-based UTs use _BASEM/_CANDM. Select by timed/invoke signature.
    fn = getattr(ut, '_launch', None) or getattr(ut, '_timed', None) or getattr(ut, '_invoke', None)
    first = next(iter(inspect.signature(fn).parameters), '') if fn else ''
    names = [prefix+'M', prefix] if first in ('mod','module') else [prefix,prefix+'M']
    target = next((getattr(ut,n) for n in names if getattr(ut,n,None) is not None), None)
    if target is None:
        raise ValueError(f'UT has no target for {arm}; supply an explicit adapter')
    return target


def verify_arm(ut, arm, harness=None):
    target = arm_target(ut, arm)
    path = source_path(target)
    if harness and callable(getattr(harness, 'assert_arm', None)):
        harness.assert_arm(ut, arm)
    expected = {'baseline': {'baseline_src'}, 'candidate': {'kernel_src','candidate_src'}}[arm]
    opposite = {'baseline_src','kernel_src','candidate_src'} - expected
    parts = set(Path(path).parts) if path else set()
    if not parts.intersection(expected) or parts.intersection(opposite):
        raise ValueError(f'cannot verify selected {arm} implementation: {path}')
    return target, path


def timed_closure(ut, target, payload, mode='auto'):
    if mode == 'auto':
        entry = next((name for name in ('_launch','_timed') if callable(getattr(ut,name,None))), None)
        if entry is None:
            if callable(getattr(ut,'_gather',None)):
                raise ValueError('UT separates gather but has no known timed entry; supply adapter.build_case')
            entry = '_invoke' if callable(getattr(ut,'_invoke',None)) else None
        if entry is None:
            raise ValueError('cannot identify timed closure; supply adapter.build_case')
    elif mode == 'invoke':
        entry = '_invoke'
    else:
        raise ValueError('runner must be auto or invoke')
    fn = getattr(ut,entry)
    return lambda: fn(target,payload), entry


def build_case(args, torch):
    if args.adapter:
        module = load_module(args.adapter, 'krf_user_adapter')
        case = module.build_case(args.task, args.case, args.arm)
        if not isinstance(case, dict) or not callable(case.get('run')):
            raise ValueError('adapter.build_case must return {run: callable, metadata: dict, ...}')
        if not case.get('metadata', {}).get('operator_boundary'):
            raise ValueError('adapter must declare metadata.operator_boundary')
        # Keep the dictionary shared with adapter callbacks: validation may add
        # reference/error/output evidence before measurement creates its manifest.
        for key, value in (("task",args.task),("case",args.case),("arm",args.arm)):
            case['metadata'].setdefault(key,value)
        return case
    if not args.ut_root:
        raise ValueError('--ut-root or --adapter is required')
    task_dir = (Path(args.ut_root) / args.task).resolve()
    harness = load_module(args.harness, 'krf_legacy_loader') if args.harness else None
    if harness and callable(getattr(harness,'load_ut',None)):
        ut = harness.load_ut(task_dir)
    else:
        body = task_dir / 'unittest_ori.py'
        if not body.exists():
            body = task_dir / 'unittest.py'
        cwd, search = os.getcwd(), list(sys.path)
        try:
            os.chdir(task_dir); sys.path.insert(0,str(task_dir))
            ut = load_module(body,'krf_unit_test')
        finally:
            os.chdir(cwd); sys.path[:] = search
    meta = getattr(ut,'META',None) or getattr(ut,'meta',None)
    ci, data = next(((i,c) for i,c in enumerate(meta['cases']) if c['name']==args.case), (None,None))
    if data is None:
        raise ValueError(f'case {args.case} missing')
    generator = torch.Generator(device=ut.DEV).manual_seed(ut.SEED+ci)
    params = inspect.signature(ut._make_args).parameters
    extra = {'ci':ci}
    if 'wts' in params:
        extra['wts'] = ut._build_weights()
    inputs = ut._make_args(data,generator,**{k:v for k,v in extra.items() if k in params})
    target, src = verify_arm(ut,args.arm,harness)
    payload = ut._strip(inputs) if callable(getattr(ut,'_strip',None)) else inputs
    run, entry = timed_closure(ut,target,payload,args.runner)
    from ut_context import snapshot_inputs
    ut_source = source_path(ut)
    return {'run':run,'metadata':{'task':args.task,'case':args.case,'arm':args.arm,
        'case_metadata':data,'implementation_source':src,'timed_closure':entry,
        'ut_source':ut_source,'ut_source_sha256':hashlib.sha256(Path(ut_source).read_bytes()).hexdigest(),
        'input_metadata':snapshot_inputs(payload,torch),
        'implementation_sha256':hashlib.sha256(Path(src).read_bytes()).hexdigest(),
        'operator_boundary':f'UT.{entry}(selected implementation, case payload)',
        'legacy_runner':args.runner=='invoke','correctness':'not_run; use UT or adapter.validate'},
        'expected_dispatches':getattr(ut,'ROOFLINE_EXPECTED_DISPATCHES',{}).get(args.case)}
