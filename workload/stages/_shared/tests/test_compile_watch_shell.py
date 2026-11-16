"""CPU replay of packed FP4/E8M0 shell tensors without shell arithmetic or copy_."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

spec = importlib.util.spec_from_file_location(
    'compile_watch_shell', Path(__file__).resolve().parents[1] / 'compile_watch.py')
cw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cw)


def descriptor(dtype, shape=(17, 8), stride=(12, 1), offset=3, storage_bytes=256, **extra):
    result = {'__tensor__': 1, 'shape': list(shape), 'stride': list(stride),
              'offset': offset, 'dtype': 'torch.' + dtype, 'device': 'cpu',
              'nbytes': math.prod(shape), 'storage_nbytes': storage_bytes, 'data': None}
    result.update(extra)
    return result


def recorded_call(fp4, scale):
    return {'positional': ['quantized'], 'args': {
        'quantized': {'__seq__': 'tuple', 'items': [fp4, scale]}}}


@unittest.skipUnless(torch is not None and all(hasattr(torch, n) for n in cw.SHELL_BYTE_DTYPES),
                     'requires CPU torch with packed shell dtypes')
class ShellReplay(unittest.TestCase):
    def setUp(self):
        self.fp4 = descriptor('float4_e2m1fn_x2', sample_error='index_select_cuda unsupported')
        self.scale = descriptor('float8_e8m0fnu', gen={
            'kind': 'normal', 'mean': .975685, 'std': .388974})

    def test_fallback_seed_parity_valid_codes_and_exact_layout(self):
        call = recorded_call(self.fp4, self.scale)
        original = copy.deepcopy(call)
        first = cw.materialize_call(call, [], seed=17)
        again = cw.materialize_call(call, [], seed=17)
        other = cw.materialize_call(call, [], seed=18)
        self.assertEqual(call, original)
        for i, desc in enumerate((self.fp4, self.scale)):
            a, b, c = (result[0][0][i] for result in (first, again, other))
            self.assertEqual(a.dtype, getattr(torch, desc['dtype'].removeprefix('torch.')))
            self.assertEqual(list(a.shape), desc['shape'])
            self.assertEqual(list(a.stride()), desc['stride'])
            self.assertEqual(a.storage_offset(), desc['offset'])
            self.assertEqual(a.untyped_storage().nbytes(), desc['storage_nbytes'])
            self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
            self.assertFalse(torch.equal(a.view(torch.uint8), c.view(torch.uint8)))
        raw = first[0][0][0].view(torch.uint8)
        codes = torch.stack((raw & 15, raw >> 4))
        magnitude = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])[codes.long() & 7]
        decoded = torch.where((codes & 8) != 0, -magnitude, magnitude)
        self.assertTrue(bool(torch.isfinite(decoded).all()))
        self.assertTrue(bool((decoded > 0).any()) and bool((decoded < 0).any()))
        exponents = first[0][0][1].view(torch.uint8)
        self.assertFalse(bool((exponents == 255).any()))
        scales = torch.ldexp(torch.ones(exponents.shape, dtype=torch.float64), exponents.int() - 127)
        self.assertTrue(bool((scales > 0).all()) and bool(torch.isfinite(scales).all()))
        self.assertTrue(torch.equal(scales.log2(), scales.log2().round()))
        self.assertTrue(any('no recorded generation distribution' in note for note in first[3]))
        self.assertTrue(any('recorded normal summary projected' in note for note in first[3]))

    def test_fp4_encodes_known_values_low_then_high_and_ties_to_even(self):
        values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                               -.5, -6., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 99.])
        raw = cw._shell_payload_bytes(values, 'float4_e2m1fn_x2')
        self.assertEqual(raw.tolist(), [0x10, 0x32, 0x54, 0x76, 0xf9,
                                        0x20, 0x42, 0x64, 0x76])

    def test_e8m0_projects_zero_and_negative_tails_to_positive_finite_scales(self):
        values = torch.tensor([0., -1., .5, 1., 2., 4., 2. ** -127, 2. ** 127])
        raw = cw._shell_payload_bytes(values, 'float8_e8m0fnu')
        self.assertEqual(raw.tolist(), [0, 127, 126, 127, 128, 129, 0, 254])

    def test_recorded_contents_use_bytes_without_changing_layout(self):
        original_copy = torch.Tensor.copy_

        def safe_copy(target, source, *args, **kwargs):
            self.assertNotIn(str(target.dtype).removeprefix('torch.'), cw.SHELL_BYTE_DTYPES)
            return original_copy(target, source, *args, **kwargs)

        for dtype in cw.SHELL_BYTE_DTYPES:
            with self.subTest(dtype=dtype), mock.patch.object(torch.Tensor, 'copy_', safe_copy):
                data = torch.arange(24, dtype=torch.uint8).reshape(4, 6).view(getattr(torch, dtype))
                desc = descriptor(dtype, shape=(4, 6), stride=(9, 1), offset=5,
                                  storage_bytes=48, data=0)
                call = {'args': {'x': desc}, 'positional': ['x']}
                result = cw.materialize_call(call, [data])[0][0]
                self.assertTrue(torch.equal(result.view(torch.uint8), data.view(torch.uint8)))
                self.assertEqual(result.stride(), (9, 1))
                self.assertEqual(result.storage_offset(), 5)
                self.assertEqual(result.untyped_storage().nbytes(), 48)

    def test_recorded_broadcast_retains_stride_zero_and_offset(self):
        for dtype in cw.SHELL_BYTE_DTYPES:
            with self.subTest(dtype=dtype):
                src = torch.tensor([20, 40, 60, 80], dtype=torch.uint8).expand(3, 4)
                data = src.view(getattr(torch, dtype))
                desc = descriptor(dtype, shape=(3, 4), stride=(0, 1), offset=2,
                                  storage_bytes=10, data=0)
                call = {'args': {'x': desc}, 'positional': ['x']}
                result = cw.materialize_call(call, [data])[0][0]
                self.assertEqual(result.stride(), (0, 1))
                self.assertEqual(result.storage_offset(), 2)
                self.assertTrue(torch.equal(result.view(torch.uint8), src))

    def test_recorded_shell_samples_are_resampled_as_bytes(self):
        for dtype in cw.SHELL_BYTE_DTYPES:
            with self.subTest(dtype=dtype):
                data = torch.tensor([16, 32, 48, 64], dtype=torch.uint8).view(getattr(torch, dtype))
                desc = descriptor(dtype, gen={'kind': 'resample', 'sample': 0})
                call = {'args': {'x': desc}, 'positional': ['x']}
                result = cw.materialize_call(call, [data], seed=3)[0][0].view(torch.uint8)
                self.assertEqual(set(result.flatten().tolist()), {16, 32, 48, 64})

    def test_nan_shell_sample_is_not_silently_synthesized(self):
        data = torch.tensor([127, 255], dtype=torch.uint8).view(torch.float8_e8m0fnu)
        desc = descriptor('float8_e8m0fnu', gen={'kind': 'resample', 'sample': 0})
        with self.assertRaisesRegex(cw.Unreplayable, 'NaN bytes'):
            cw.materialize_call({'args': {'scale': desc}, 'positional': ['scale']}, [data])


if __name__ == '__main__':
    unittest.main()
