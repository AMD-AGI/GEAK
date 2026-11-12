"""CPU object reconstruction and recorded-signature replay, with no GPU imports."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

SHARED = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SHARED / filename)
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


cw = load('compile_watch_object_test', 'compile_watch.py')
lr = load('leg_runner_object_test', 'leg_runner.py')


def fake_torch():
    module = types.ModuleType('torch')
    module.Tensor = type('Tensor', (), {})
    module.cuda = types.SimpleNamespace(is_available=lambda: False)
    module.saved = []
    module.save = lambda value, path: module.saved.append((value, path))
    return module


def call_record(args, positional=None):
    return {'id': 'call-0', 'callable': 'test_live_object:Layer.forward',
            'kernel': '_native', 'args': args,
            'positional': list(args) if positional is None else positional}


class ObjectReplay(unittest.TestCase):
    def setUp(self):
        self.torch = fake_torch()
        self.patcher = mock.patch.dict(sys.modules, {'torch': self.torch})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_bound_method_keeps_self_and_recorded_signature(self):
        class Layer:
            def __init__(self, scale):
                self.scale = scale

            def forward(self, x, *, bias):
                return self.scale * x + bias

        call = call_record({'self': {'__object__': 'Layer'}, 'x': 7, 'bias': 3},
                           positional=['self', 'x'])
        seen = []

        def build(name, descriptor, *, call, seed):
            seen.append((name, descriptor, call, seed))
            return Layer(seed)

        args, kwargs, tensors, notes = cw.materialize_call(call, [], seed=11,
                                                         make_server_object=build)
        self.assertEqual(Layer.forward(*args, **kwargs), 80)
        self.assertEqual(kwargs, {'bias': 3})
        self.assertEqual(tensors, {})
        self.assertEqual(seen, [('self', {'__object__': 'Layer'}, call, 11)])
        self.assertIn('cases.make_server_object', notes[0])

    def test_nonself_object_and_nested_kwargs_preserve_structural_values(self):
        cfg = {'__object__': 'Config'}
        call = call_record({'x': 9,
                            '*tail': {'__seq__': 'tuple', 'items': [2, cfg]},
                            '**options': {'labels': {'__seq__': 'list', 'items': [1, 4]},
                                          'cfg': cfg}}, positional=['x'])
        seen = []

        def build(name, descriptor, *, call, seed):
            seen.append(name)
            return types.SimpleNamespace(value=seed)

        args, kwargs, _, _ = cw.materialize_call(call, [], seed=17, make_server_object=build)
        self.assertEqual(args[:2], [9, 2])
        self.assertEqual(args[2].value, 17)
        self.assertEqual(kwargs['cfg'].value, 17)
        self.assertEqual(kwargs['labels'], [1, 4])
        self.assertEqual(seen, ['*tail[1]', 'cfg'])

    def test_default_refuses_self_and_other_object_descriptors(self):
        for name in ('self', 'cfg'):
            with self.subTest(name=name):
                call = call_record({name: {'__object__': 'SomeObject'}})
                with self.assertRaisesRegex(cw.Unreplayable, 'cases.make_server_object'):
                    cw.materialize_call(call, [])
        nested = call_record({'xs': {'__seq__': 'tuple', 'items': [{'__object__': 'Cfg'}]}})
        with self.assertRaisesRegex(cw.Unreplayable, r'xs\[0\]'):
            cw.materialize_call(nested, [])

    def test_builtin_literals_replay_without_custom_object_hook(self):
        descriptor = {'__object__': 'builtins.frozenset', 'repr': "frozenset({'decode'})"}
        call = call_record({'enabled': descriptor})
        args, kwargs, tensors, notes = cw.materialize_call(call, [])
        self.assertEqual(args, [frozenset({'decode'})])
        self.assertEqual((kwargs, tensors, notes), ({}, {}, []))

    def test_builtin_literals_and_custom_object_hook_coexist(self):
        call = call_record({'self': {'__object__': 'Layer'},
                            'enabled': {'__object__': 'builtins.frozenset',
                                        'repr': "frozenset({'decode'})"}})
        instance = object()
        hook = mock.Mock(return_value=instance)
        args, _, _, _ = cw.materialize_call(call, [], make_server_object=hook)
        self.assertIs(args[0], instance)
        self.assertEqual(args[1], frozenset({'decode'}))
        hook.assert_called_once()
        self.assertEqual(hook.call_args.args[0], 'self')

    def test_hook_cannot_mutate_recorded_call_or_ordinary_arguments(self):
        call = call_record({'cfg': {'__object__': 'Cfg'}, 'scalar': 5,
                            'values': {'__seq__': 'list', 'items': [2, 8]}})
        original = copy.deepcopy(call)

        def build(name, descriptor, *, call, seed):
            descriptor['__object__'] = 'Changed'
            call['args']['scalar'] = -1
            call['args']['values']['items'][0] = -1
            call['positional'].reverse()
            return object()

        args, _, _, _ = cw.materialize_call(call, [], make_server_object=build)
        self.assertEqual(call, original)
        self.assertEqual(args[1:], [5, [2, 8]])

    def test_hook_cannot_supply_missing_structural_arguments(self):
        call = call_record({'cfg': {'__object__': 'Cfg'}, 'ids': {'__missing__': 1}})
        hook = mock.Mock()
        with self.assertRaisesRegex(cw.Unreplayable, 'ids was not bound'):
            cw.materialize_call(call, [], make_server_object=hook)
        hook.assert_not_called()

    def test_hook_explicitly_declines_reconstruction(self):
        call = call_record({'cfg': {'__object__': 'Cfg'}})
        for result in (None, NotImplemented):
            with self.subTest(result=result):
                with self.assertRaisesRegex(cw.Unreplayable, 'did not rebuild argument cfg'):
                    cw.materialize_call(call, [], make_server_object=lambda *a, **kw: result)

    def test_leg_loads_cases_hook_and_passes_original_call_index_seed(self):
        class Layer:
            def forward(self, x, *, bias):
                return self.scale * x + bias

        live = types.ModuleType('test_live_object')
        live.Layer = Layer
        records = []
        for index in range(3):
            record = call_record({'self': {'__object__': 'Layer'}, 'x': index + 1, 'bias': 4},
                                 positional=['self', 'x'])
            record['id'] = 'call-' + str(index)
            records.append(record)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'server_calls.json').write_text(json.dumps({'calls': records}))
            (path / 'cases.py').write_text(
                'from test_live_object import Layer\n'
                'seen = []\n'
                'def make_server_object(name, descriptor, *, call, seed):\n'
                '    seen.append((name, descriptor, call["id"], seed))\n'
                '    instance = Layer()\n'
                '    instance.scale = seed\n'
                '    return instance\n')
            args = types.SimpleNamespace(sigs='call-2', out=str(path / 'out.pt'), seed=30)
            meta = {'target_callable': 'test_live_object:Layer.forward',
                    'server_evidence': {'server_calls': 'server_calls.json'}}
            with mock.patch.dict(sys.modules, {'test_live_object': live, 'compile_watch': cw}), \
                    mock.patch.object(cw, 'call_data', return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()):
                lr._server_calls(args, directory, meta)
                seen = sys.modules['cases'].seen
            self.assertEqual(seen, [('self', {'__object__': 'Layer'}, 'call-2', 32)])
            result = self.torch.saved[-1][0]
            self.assertEqual(list(result), ['call-2'])
            self.assertEqual(result['call-2']['status'], 'ok')
            self.assertEqual(result['call-2']['outputs']['return'], 100)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch not installed; run in CPU torch environment')
class TensorOwnership(unittest.TestCase):
    def test_object_hook_does_not_change_tensor_generation_or_recorded_structure(self):
        tensors = []
        budget = [cw.EVENT_DATA_CAP]
        values = {'x': torch.ones((64, 64), dtype=torch.float32),
                  'ids': torch.tensor([1, 4, 2], dtype=torch.int64)}
        descriptors = {name: cw._describe(t, tensors, budget, False, name=name)
                       for name, t in values.items()}
        descriptors['x'].pop('gen', None)
        basic = call_record(descriptors)
        with_object = copy.deepcopy(basic)
        with_object['args']['cfg'] = {'__object__': 'Config'}
        hook = mock.Mock(return_value=object())
        plain = cw.materialize_call(basic, tensors, seed=23)[2]
        actual = cw.materialize_call(with_object, tensors, seed=23, make_server_object=hook)[2]
        self.assertEqual(hook.call_count, 1)
        self.assertEqual(hook.call_args.args[0], 'cfg')
        for name in values:
            self.assertTrue(torch.equal(plain[name], actual[name]))
            self.assertEqual(plain[name].stride(), actual[name].stride())
        self.assertTrue(torch.equal(actual['ids'], values['ids']))


if __name__ == '__main__':
    unittest.main()
