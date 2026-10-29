"""CPU regressions for relocated rebinds and actual reference-harness case tables."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / 'workload/stages/applyback/build_overlay.py'
REGISTER = ROOT / 'workload/stages/applyback/reference-replay-ab/register_case.py'
SUMMARY = ROOT / 'workload/stages/applyback/reference-replay-ab/summarize.py'


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.mark.parametrize('nested', [False, True])
def test_register_preserves_reference_case_and_table_metadata(tmp_path, nested):
    harness = tmp_path / 'harness'
    baseline = {'description': 'same workload', 'server_args': {'tp': 8},
                'environment': {'FLAG': 'yes'}, 'custom_metadata': [1, {'keep': True}],
                'python_overlay': None}
    table = {'cases': {'baseline': baseline}, 'default_case': 'baseline', 'extra': 42} if nested else {'baseline': baseline}
    cases = harness / 'cases.json'
    write(cases, json.dumps(table))
    before = cases.read_bytes()
    overlay = tmp_path / 'source_overlay'
    write(overlay / 'entry.py', 'VALUE = 1\n')
    result = subprocess.run([sys.executable, str(REGISTER), '--harness', str(harness),
        '--case-file', 'cases.json', '--case', 'candidate', '--base-case', 'baseline',
        '--overlay', str(overlay), '--overlay-subdir', 'overlays/candidate', '--out', str(tmp_path/'registration.json')],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    after = json.loads(cases.read_text())
    entries = after['cases'] if nested else after
    assert entries['baseline'] == baseline
    assert entries['candidate']['custom_metadata'] == baseline['custom_metadata']
    assert entries['candidate']['environment'] == baseline['environment']
    assert entries['candidate']['python_overlay'] == 'overlays/candidate'
    assert (harness/'cases.json.before_candidate').read_bytes() == before
    if nested:
        assert after['default_case'] == 'baseline' and after['extra'] == 42
        assert 'candidate' not in after


def optimized_row(task):
    source = task / 'kernel_src'
    return {'kernel': task.name, 'verdict': 'pass', 'verified_speedup': 1.2,
            'task_meta_sha256': hashlib.sha256((task/'meta.json').read_bytes()).hexdigest(),
            'best_kernel_src': str(source), 'kernel_src_hashes': {
                p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob('*') if p.is_file()}}


def make_rebind_tasks(tmp_path, shared_env=False, config_env=False):
    tasks, installed = tmp_path/'tasks', tmp_path/'installed'
    rows, options = [], {}
    for number, name in enumerate(['first', 'second', 'jit'], 1):
        module = 'installed_' + name
        write(installed/(module+'.py'), 'def op(): return -1\nclass Backend: pass\n')
        task = tasks/name
        write(task/'kernel_src/compute.py', f'VALUE = {number * 11}\n')
        entry_name = name + '_entry'
        if name == 'first':
            source = ('from pathlib import Path\nimport os\n'
                "root = Path(os.environ['SHARED_ROOT' if 'SHARED_ROOT' in os.environ else 'FIRST_ROOT'])\n")
        else:
            # The same parent/kernel_src lookup used by real task-relative adapters.
            source = 'from pathlib import Path\nroot = Path(__file__).parent.parent / "kernel_src"\n'
        source += ('import importlib.util\n'
                   'spec = importlib.util.spec_from_file_location(__name__ + "_leaf", root / "compute.py")\n'
                   'leaf = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(leaf)\n'
                   'def op(): return leaf.VALUE\n'
                   'class Backend:\n    def op(self): return leaf.VALUE\n'
                   'class JIT:\n    def __init__(self, fn): self.fn = fn\n'
                   'jit = JIT(op)\n')
        write(task/'kernel_src'/(entry_name+'.py'), source)
        attr = {'first': 'Backend', 'second': 'op', 'jit': 'jit'}[name]
        target = module + ':' + ('op' if name == 'jit' else attr)
        bind = {'kind': 'rebind', 'target': target, 'impl_module': entry_name,
                'impl_attr': attr, 'file': 'kernel_src/' + entry_name + '.py'}
        if name == 'first':
            if config_env:
                options[name] = {'source_env': {'FIRST_ROOT': '{kernel_src}'}}
            else:
                bind['requires_env'] = {'SHARED_ROOT' if shared_env else 'FIRST_ROOT': '<task>/kernel_src'}
        if shared_env and name == 'second':
            options[name] = {'source_env': {'SHARED_ROOT': '{kernel_src}'}}
        write(task/'meta.json', json.dumps({'candidate_bind': bind, 'target_callable': target}))
        rows.append(optimized_row(task))
    write(tmp_path/'optimized.json', json.dumps({'kernels': rows}))
    write(tmp_path/'settings.json', json.dumps({'binding_options': options}))
    return tasks, installed


def run_builder(tmp_path, tasks, installed):
    env = dict(os.environ, PYTHONPATH=str(installed))
    env.pop('FIRST_ROOT', None); env.pop('SHARED_ROOT', None)
    return subprocess.run([sys.executable, str(BUILDER), '--optimized', str(tmp_path/'optimized.json'),
        '--tasks-dir', str(tasks), '--overlay', str(tmp_path/'overlay'), '--manifest', str(tmp_path/'applied.json'),
        '--settings', str(tmp_path/'settings.json'), '--geak-root', str(ROOT)],
        env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('config_env', [False, True])
def test_rebind_classes_functions_and_jit_keep_companions_after_relocation(tmp_path, config_env):
    tasks, installed = make_rebind_tasks(tmp_path, config_env=config_env)
    process = run_builder(tmp_path, tasks, installed)
    assert process.returncode == 0, process.stdout + process.stderr
    applied = json.loads((tmp_path/'applied.json').read_text())
    assert applied['all_shadowed'] and applied['serving_rank_source_proof_required']
    assert all(entry['bundled_file'] for entry in applied['applied'])
    relocated = tmp_path/'harness/overlays/accepted'
    shutil.copytree(tmp_path/'overlay', relocated)
    shutil.rmtree(tasks); shutil.rmtree(tmp_path/'overlay')
    env = dict(os.environ, PYTHONPATH=str(relocated) + os.pathsep + str(installed))
    env.pop('FIRST_ROOT', None); env.pop('SHARED_ROOT', None)
    result = subprocess.run([sys.executable, '-c',
        'import json,os,installed_first,installed_second,installed_jit; '
        'print(json.dumps([installed_first.Backend().op(),installed_second.op(),installed_jit.op.fn(),os.environ["FIRST_ROOT"]]))'],
        env=env, cwd=tmp_path, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values[:3] == [11, 22, 33]
    assert Path(values[3]).is_relative_to(relocated)


def test_conflicting_source_root_env_is_rejected(tmp_path):
    tasks, installed = make_rebind_tasks(tmp_path, shared_env=True)
    result = run_builder(tmp_path, tasks, installed)
    assert result.returncode != 0
    assert 'conflicting task source environment' in result.stderr


@pytest.mark.parametrize('config_env', [False, True])
def test_module_source_hooks_precede_imports_and_survive_relocation(tmp_path, config_env):
    tasks, installed = tmp_path/'tasks', tmp_path/'installed'
    options, rows = {}, []
    for number, name in enumerate(['first', 'second'], 1):
        module = 'module_' + name
        key = name.upper() + '_MODULE_ROOT'
        write(installed/(module + '.py'), 'def op(): return -1\n')
        task = tasks/name
        write(task/'kernel_src/compute.py', f'VALUE = {number * 17}\n')
        source = ('from pathlib import Path\nimport importlib.util, os\n'
                  'assert "FIRST_MODULE_ROOT" in os.environ and "SECOND_MODULE_ROOT" in os.environ\n'
                  f'root = Path(os.environ[{key!r}])\n'
                  'assert root == Path(__file__).resolve().parent\n'
                  'spec = importlib.util.spec_from_file_location(__name__ + "_leaf", root / "compute.py")\n'
                  'leaf = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(leaf)\n'
                  'runtime = root.parent / "run/native_build"\nruntime.mkdir(parents=True, exist_ok=True)\n'
                  '(runtime / "import-marker").write_text("runtime output")\n'
                  'def op(): return leaf.VALUE\n')
        write(task/'kernel_src/entry.py', source)
        bind = {'kind': 'module', 'module': module, 'file': 'kernel_src/entry.py'}
        if config_env:
            options[name] = {'source_env': {key: '{kernel_src}'}}
        else:
            bind['requires_env'] = {key: '<task>/kernel_src'}
        write(task/'meta.json', json.dumps({'candidate_bind': bind, 'target_callable': module + ':op'}))
        rows.append(optimized_row(task))
    write(tmp_path/'optimized.json', json.dumps({'kernels': rows}))
    write(tmp_path/'settings.json', json.dumps({'binding_options': options}))
    process = run_builder(tmp_path, tasks, installed)
    assert process.returncode == 0, process.stdout + process.stderr
    applied = json.loads((tmp_path/'applied.json').read_text())
    assert applied['all_shadowed']
    overlay = tmp_path/'overlay'
    assert (overlay/'sitecustomize.py').read_text().count('resolve source hooks before patched imports') == 1
    for entry in applied['applied']:
        assert not (overlay/'_patched'/entry['kernel']/'run').exists()
        for name, expected in entry['kernel_src_hashes'].items():
            assert hashlib.sha256((overlay/'_patched'/entry['kernel']/'kernel_src'/name).read_bytes()).hexdigest() == expected
    relocated = tmp_path/'harness/overlays/accepted'
    shutil.copytree(overlay, relocated)
    shutil.rmtree(tasks); shutil.rmtree(overlay)
    env = dict(os.environ, PYTHONPATH=str(relocated) + os.pathsep + str(installed),
               FIRST_MODULE_ROOT=str(tasks/'first/kernel_src'), SECOND_MODULE_ROOT=str(tasks/'second/kernel_src'))
    result = subprocess.run([sys.executable, '-c',
        'import json,os,module_first,module_second; '
        'print(json.dumps([module_first.op(),module_second.op(),module_first.__file__,module_second.__file__, '
        'os.environ["FIRST_MODULE_ROOT"],os.environ["SECOND_MODULE_ROOT"]]))'],
        env=env, cwd=tmp_path, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values[:2] == [17, 34]
    assert all(Path(value).is_relative_to(relocated) for value in values[2:])
    for name, expected in applied['source_inventory'].items():
        assert hashlib.sha256((relocated/name).read_bytes()).hexdigest() == expected


def summary_module():
    spec = importlib.util.spec_from_file_location('reference_replay_summary', SUMMARY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_write(path, value):
    write(path, json.dumps(value))


def summary_fixture(tmp_path, native=False):
    module = summary_module()
    overlay = tmp_path/'overlay'
    source_name = '_patched/task/kernel_src/compute.cu' if native else '_patched/task/kernel_src/compute.py'
    write(overlay/source_name, 'source bytes\n')
    digest = module.sha256(overlay/source_name)
    source_hashes = {source_name: digest}
    target = 'fixture:operation'
    applied = {'all_shadowed': True, 'source_inventory': source_hashes,
        'applied': [{'target_callable': target, 'bundled_source_hashes': source_hashes,
                     'native_proof_required': native}]}
    settings = {'overlay_receipt': {'file': 'run_identity.json', 'hashes_path': 'server.python_overlay_files',
                                    'root_path': 'server.python_overlay'},
        'native_validation': {'file': 'aiperf_artifacts/profile_export_aiperf.json',
            'required_values': {'metadata.submission_valid': True, 'was_cancelled': False, 'error_summary': []},
            'aggregate_required_values': {'request_accounting.records_error_dropped': 0},
            'completion_receipt': {'file': 'run_identity.json', 'required_values': {'client_exit_code': 0}},
            'duration_coverage': {'path': 'metadata.metric_duration_coverage'}},
        'serving_source_proof': {'file': 'source_proof.json', 'world_size': 2}}
    write(tmp_path/'producer.py', '# CPU fixture; actual worker receipts are supplied only by test fixtures.\n')
    settings['serving_source_proof'].update({'producer_file': str(tmp_path/'producer.py'),
                                            'producer_sha256': module.sha256(tmp_path/'producer.py')})
    module.validate_settings(settings)
    for tag, throughput in [('stock', 100), ('one', 110), ('two', 90), ('three', 115)]:
        run = tmp_path/'runs'/tag
        arm_overlay = tmp_path/'arm_overlays'/tag
        shutil.copytree(overlay, arm_overlay)
        json_write(run/(tag+'.json'), {'request_metrics': {'throughput': {
            'output': {'tokens_per_second': throughput}, 'duration_seconds': 3607}},
            'request_accounting': {'records_error_dropped': 0}})
        json_write(run/'aiperf_artifacts/profile_export_aiperf.json', {'metadata': {
            'submission_valid': True, 'metric_duration_coverage': [{'phase_name': 'profiling',
            'expected_duration_seconds': 3600, 'required_ratio': 0.95, 'ttft_ratio': 1., 'inter_token_latency_ratio': 1.}]},
            'was_cancelled': False, 'error_summary': []})
        json_write(run/'run_identity.json', {'client_exit_code': 0, 'server': {
            'python_overlay': str(arm_overlay),
            'python_overlay_files': [{'path': source_name, 'sha256': digest}]}})
        ranks = []
        for rank in range(2):
            log = run/f'rank{rank}.log'
            write(log, f'CPU fixture for rank {rank}, not real serving proof\n')
            binding = {'target_callable': target, 'invocations': 1, 'source_hashes': source_hashes,
                       'resolved_file': source_name, 'leaf_source_files': [source_name]}
            if native:
                library = run/'fixture.so'; write(library, 'CPU fixture library bytes\n')
                provenance = run/'build.json'; json_write(provenance, {'fixture': True})
                binding['native_libraries'] = [{'path': str(library), 'sha256': module.sha256(library),
                    'loaded': True, 'namespace': 'fixture_native', 'source_hashes': source_hashes,
                    'build_provenance_ref': {'path': 'build.json', 'sha256': module.sha256(provenance)}}]
            ranks.append({'rank': rank, 'pid': 100+rank, 'post_lazy_load': True, 'observers_removed': True,
                'log_ref': {'path': log.name, 'sha256': module.sha256(log)}, 'bindings': [binding]})
        json_write(run/'source_proof.json', {'contract': 'serving_source_proof/v1', 'run_tag': tag,
                                            'overlay_root': str(arm_overlay), 'ranks': ranks})
    json_write(tmp_path/'settings.json', settings)
    json_write(tmp_path/'applied.json', applied)
    return module, overlay, applied, settings


def run_summary(tmp_path, tags=('one', 'two', 'three')):
    return subprocess.run([sys.executable, str(SUMMARY), '--settings', str(tmp_path/'settings.json'),
        '--overlay', str(tmp_path/'overlay'), '--applied', str(tmp_path/'applied.json'),
        '--reference-run', str(tmp_path/'runs/stock/stock.json'),
        '--out-reference', str(tmp_path/'reference.json'), '--out-summary', str(tmp_path/'summary.json'),
        '--expected-repeats', '3', '--duration-s', '3600', '--runs',
        *[str(tmp_path/'runs'/tag) for tag in tags]],
        text=True, capture_output=True)


def test_summary_keeps_every_valid_arm_including_slow_repeats(tmp_path):
    summary_fixture(tmp_path)
    result = run_summary(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['output_token_throughput'] == 110
    assert summary['legs_tok_s'] == [110, 90, 115]
    assert summary['overlay_loaded_by_every_leg']


def test_repeated_receipt_is_not_an_independent_arm(tmp_path):
    summary_fixture(tmp_path)
    result = run_summary(tmp_path, tags=('one', 'one', 'three'))
    assert result.returncode != 0
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert not summary['independent_arms']
    assert 'output_token_throughput' not in summary
    assert not (tmp_path/'reference.json').exists()


@pytest.mark.parametrize('problem', ['missing_proof', 'missing_rank', 'observer', 'hash', 'native_invalid',
    'client_exit', 'profiling_errors', 'short_duration', 'short_coverage', 'basis', 'missing_throughput',
    'bool_accounting', 'bad_duration_row', 'wrong_arm_root', 'changed_source'])
def test_summary_refuses_unfounded_or_partial_arms(tmp_path, problem):
    summary_fixture(tmp_path)
    run = tmp_path/'runs/two'
    if problem == 'missing_proof':
        (run/'source_proof.json').unlink()
    elif problem in ('missing_rank', 'observer', 'hash', 'wrong_arm_root'):
        path = run/'source_proof.json'; data = json.loads(path.read_text())
        if problem == 'missing_rank': data['ranks'].pop()
        if problem == 'observer': data['ranks'][0]['observers_removed'] = False
        if problem == 'hash': data['ranks'][0]['bindings'][0]['source_hashes'] = {}
        if problem == 'wrong_arm_root': data['overlay_root'] = str(tmp_path/'arm_overlays/one')
        json_write(path, data)
    elif problem == 'client_exit':
        path = run/'run_identity.json'; data = json.loads(path.read_text()); data['client_exit_code'] = 1; json_write(path, data)
    elif problem in ('native_invalid', 'short_coverage', 'bad_duration_row'):
        path = run/'aiperf_artifacts/profile_export_aiperf.json'; data = json.loads(path.read_text())
        if problem == 'native_invalid': data['metadata']['submission_valid'] = False
        elif problem == 'short_coverage': data['metadata']['metric_duration_coverage'][0]['ttft_ratio'] = 0.8
        else: data['metadata']['metric_duration_coverage'] = [None]
        json_write(path, data)
    elif problem == 'changed_source':
        write(tmp_path/'arm_overlays/two/_patched/task/kernel_src/compute.py', 'changed\n')
    else:
        path = run/'two.json'; data = json.loads(path.read_text())
        if problem == 'profiling_errors': data['request_accounting']['records_error_dropped'] = 1
        if problem == 'bool_accounting': data['request_accounting']['records_error_dropped'] = False
        if problem == 'short_duration': data['request_metrics']['throughput']['duration_seconds'] = 1200
        if problem == 'basis': data['metric_basis'] = 'request_throughput'
        if problem == 'missing_throughput': del data['request_metrics']['throughput']['output']
        json_write(path, data)
    result = run_summary(tmp_path)
    assert result.returncode != 0
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert 'output_token_throughput' not in summary
    assert not (tmp_path/'reference.json').exists()


def test_native_proof_requires_library_and_build_linkage(tmp_path):
    module, overlay, applied, settings = summary_fixture(tmp_path, native=True)
    run = tmp_path/'runs/one'
    assert module.leg(run, module.overlay_hashes(overlay), settings, 3600, overlay, applied)['admissible']
    path = run/'source_proof.json'; proof = json.loads(path.read_text())
    proof['ranks'][0]['bindings'][0]['native_libraries'][0]['source_hashes'] = {}
    json_write(path, proof)
    assert not module.leg(run, module.overlay_hashes(overlay), settings, 3600, overlay, applied)['admissible']


def test_runtime_outputs_do_not_redefine_prelaunch_source_inventory(tmp_path):
    summary_fixture(tmp_path)
    write(tmp_path/'arm_overlays/one/_patched/task/run/native_build/cache/generated.cpp', 'runtime output\n')
    result = run_summary(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_per_arm_registration_makes_real_source_copies_without_prior_builds(tmp_path):
    tasks, installed = make_rebind_tasks(tmp_path)
    result = run_builder(tmp_path, tasks, installed)
    assert result.returncode == 0, result.stderr
    accepted = tmp_path/'overlay'
    harness = tmp_path/'harness'
    json_write(harness/'cases.json', {'cases': {'baseline': {'python_overlay': None}}})
    for tag in ('one', 'two'):
        result = subprocess.run([sys.executable, str(REGISTER), '--harness', str(harness),
            '--case-file', 'cases.json', '--case', tag, '--base-case', 'baseline',
            '--overlay', str(accepted), '--overlay-subdir', 'overlays/'+tag,
            '--out', str(tmp_path/(tag+'.registration.json'))], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        arm = harness/'overlays'/tag
        env = dict(os.environ, PYTHONPATH=str(arm)+os.pathsep+str(installed))
        probe = subprocess.run([sys.executable, '-c', 'import os,installed_first; print(os.environ["FIRST_ROOT"])'],
                               env=env, cwd=tmp_path, text=True, capture_output=True)
        assert probe.returncode == 0, probe.stderr
        assert Path(probe.stdout.strip()).is_relative_to(arm)
        assert not arm.is_symlink()
        if tag == 'one':
            write(arm/'_patched/first/run/native_build/only_first.bin', 'runtime output\n')
    assert not (harness/'overlays/two/_patched/first/run/native_build').exists()
    assert not (accepted/'_patched/first/run/native_build').exists()
    assert (harness/'overlays/one/_patched/first/run/native_build/only_first.bin').exists()


def test_missing_or_changed_producer_blocks_preflight(tmp_path):
    module, _, _, settings = summary_fixture(tmp_path)
    settings['serving_source_proof'].pop('producer_file')
    with pytest.raises(ValueError, match='producer_file'):
        module.validate_settings(settings)
    settings['serving_source_proof']['producer_file'] = str(tmp_path/'producer.py')
    write(tmp_path/'producer.py', '# changed after review\n')
    with pytest.raises(ValueError, match='reviewed SHA'):
        module.validate_settings(settings)


def test_binding_checks_cannot_contaminate_accepted_source(tmp_path):
    tasks, installed = make_rebind_tasks(tmp_path)
    entry = tasks/'first/kernel_src/first_entry.py'
    entry.write_text(entry.read_text() + '\noutput = Path(__file__).parent.parent / "run/native_build/check.txt"\n'
                     'output.parent.mkdir(parents=True, exist_ok=True)\noutput.write_text(str(__file__))\n')
    report = json.loads((tmp_path/'optimized.json').read_text())
    report['kernels'][0] = optimized_row(tasks/'first')
    json_write(tmp_path/'optimized.json', report)
    result = run_builder(tmp_path, tasks, installed)
    assert result.returncode == 0, result.stderr
    applied = json.loads((tmp_path/'applied.json').read_text())
    assert not (tmp_path/'overlay/_patched/first/run/native_build').exists()
    assert (Path(applied['binding_check_overlay'])/'_patched/first/run/native_build/check.txt').exists()


def test_reviewed_producer_is_installed_but_not_run_in_binding_checks(tmp_path):
    tasks, installed = make_rebind_tasks(tmp_path)
    producer = tmp_path/'producer.py'
    write(producer, 'from pathlib import Path\nimport os\n'
          '(Path(os.environ["GEAK_SERVING_SOURCE_PROOF_RUN_DIR"])/"producer_ran").write_text(__file__)\n')
    settings = json.loads((tmp_path/'settings.json').read_text())
    settings['serving_source_proof'] = {'file':'source_proof.json', 'world_size':2,
        'producer_file':str(producer), 'producer_sha256':hashlib.sha256(producer.read_bytes()).hexdigest()}
    json_write(tmp_path/'settings.json', settings)
    result = run_builder(tmp_path, tasks, installed)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path/'producer_ran').exists()
    assert (tmp_path/'overlay/_geak_source_obligations.json').is_file()
    env = dict(os.environ, PYTHONPATH=str(tmp_path/'overlay')+os.pathsep+str(installed),
               GEAK_SERVING_SOURCE_PROOF_RUN_DIR=str(tmp_path))
    result = subprocess.run([sys.executable, '-c', 'import installed_first'], env=env,
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert Path((tmp_path/'producer_ran').read_text()).is_relative_to(tmp_path/'overlay')
