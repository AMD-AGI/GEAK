import torch
import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def debug_kernel(
    predictions_ptr: fx.Tensor,
    targets_ptr: fx.Tensor,
    output_ptr: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    
    predictions_ptr = fx.rocdl.make_buffer_tensor(predictions_ptr)
    targets_ptr = fx.rocdl.make_buffer_tensor(targets_ptr)
    
    tPred = fx.logical_divide(predictions_ptr, fx.make_layout(block_dim, 1))
    tTarg = fx.logical_divide(targets_ptr, fx.make_layout(block_dim, 1))
    tOut = fx.logical_divide(output_ptr, fx.make_layout(block_dim, 1))
    
    tPred = fx.slice(tPred, (None, bid))
    tTarg = fx.slice(tTarg, (None, bid))
    tOut = fx.slice(tOut, (None, bid))
    
    tPred = fx.logical_divide(tPred, fx.make_layout(1, 1))
    tTarg = fx.logical_divide(tTarg, fx.make_layout(1, 1))
    tOut = fx.logical_divide(tOut, fx.make_layout(1, 1))
    
    RegTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
    
    copyAtomBuf = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    
    rPred = fx.memref_alloca(RegTy, fx.make_layout(1, 1))
    rTarg = fx.memref_alloca(RegTy, fx.make_layout(1, 1))
    rOut = fx.memref_alloca(RegTy, fx.make_layout(1, 1))
    
    fx.copy_atom_call(copyAtomBuf, fx.slice(tPred, (None, tid)), rPred)
    fx.copy_atom_call(copyAtomBuf, fx.slice(tTarg, (None, tid)), rTarg)
    
    pred_val = fx.memref_load_vec(rPred)
    targ_val = fx.memref_load_vec(rTarg)
    
    mul_result = fx.arith.mulf(pred_val, targ_val)
    
    fx.memref_store_vec(mul_result, rOut)
    fx.copy_atom_call(copyAtom, rOut, fx.slice(tOut, (None, tid)))


@flyc.jit
def run_debug(
    predictions_t: fx.Tensor,
    targets_t: fx.Tensor,
    output_t: fx.Tensor,
    n: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 256
    grid_x = (n + block_dim - 1) // block_dim
    
    debug_kernel(
        predictions_t, targets_t, output_t, block_dim
    ).launch(grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream)


# Test
torch.manual_seed(42)
batch_size = 128
input_size = 128

predictions = torch.randn(batch_size, input_size).cuda()
targets = torch.randint(0, 2, (batch_size,)).float().cuda() * 2 - 1

targets_broadcast = targets.unsqueeze(1).expand(batch_size, input_size).clone()

predictions_flat = predictions.reshape(-1).contiguous()
targets_flat = targets_broadcast.reshape(-1).contiguous()
n_elements = predictions_flat.numel()

output = torch.empty_like(predictions_flat)

t_pred = flyc.from_dlpack(predictions_flat).mark_layout_dynamic(leading_dim=0, divisibility=4)
t_targ = flyc.from_dlpack(targets_flat).mark_layout_dynamic(leading_dim=0, divisibility=4)
t_out = flyc.from_dlpack(output).mark_layout_dynamic(leading_dim=0, divisibility=4)

run_debug(t_pred, t_targ, t_out, n_elements, stream=torch.cuda.Stream())
torch.cuda.synchronize()

# Compare
pytorch_mul = predictions * targets.unsqueeze(1)
pytorch_mul_flat = pytorch_mul.reshape(-1)

print("FlyDSL mul result (first 10):", output[:10])
print("PyTorch mul result (first 10):", pytorch_mul_flat[:10])
print("Are they close?", torch.allclose(output, pytorch_mul_flat))
print("Max diff:", torch.max(torch.abs(output - pytorch_mul_flat)))
