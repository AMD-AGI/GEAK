import torch
import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def hinge_loss_elementwise_kernel(
    predictions_ptr: fx.Tensor,
    targets_ptr: fx.Tensor,
    output_ptr: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    
    predictions_ptr = fx.rocdl.make_buffer_tensor(predictions_ptr)
    targets_ptr = fx.rocdl.make_buffer_tensor(targets_ptr)
    
    tile_elems = block_dim * vec_width
    
    tPred = fx.logical_divide(predictions_ptr, fx.make_layout(tile_elems, 1))
    tTarg = fx.logical_divide(targets_ptr, fx.make_layout(tile_elems, 1))
    tOut = fx.logical_divide(output_ptr, fx.make_layout(tile_elems, 1))
    
    tPred = fx.slice(tPred, (None, bid))
    tTarg = fx.slice(tTarg, (None, bid))
    tOut = fx.slice(tOut, (None, bid))
    
    tPred = fx.logical_divide(tPred, fx.make_layout(vec_width, 1))
    tTarg = fx.logical_divide(tTarg, fx.make_layout(vec_width, 1))
    tOut = fx.logical_divide(tOut, fx.make_layout(vec_width, 1))
    
    copy_bits = vec_width * 32
    RegTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register)
    
    copyAtomBuf = fx.make_copy_atom(fx.rocdl.BufferCopy(copy_bits), fx.Float32)
    copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)
    
    rPred = fx.memref_alloca(RegTy, fx.make_layout(vec_width, 1))
    rTarg = fx.memref_alloca(RegTy, fx.make_layout(vec_width, 1))
    rOut = fx.memref_alloca(RegTy, fx.make_layout(vec_width, 1))
    
    fx.copy_atom_call(copyAtomBuf, fx.slice(tPred, (None, tid)), rPred)
    fx.copy_atom_call(copyAtomBuf, fx.slice(tTarg, (None, tid)), rTarg)
    
    pred_val = fx.memref_load_vec(rPred)
    targ_val = fx.memref_load_vec(rTarg)
    
    mul_result = fx.arith.mulf(pred_val, targ_val)
    
    one_vec = fx.arith.constant_vector(1.0, fx.T.vector(vec_width, fx.T.f32()))
    sub_result = fx.arith.subf(one_vec, mul_result)
    
    zero_vec = fx.arith.constant_vector(0.0, fx.T.vector(vec_width, fx.T.f32()))
    clamped = fx.arith.maximumf(zero_vec, sub_result)
    
    fx.memref_store_vec(clamped, rOut)
    fx.copy_atom_call(copyAtom, rOut, fx.slice(tOut, (None, tid)))


@flyc.jit
def run_hinge_loss_elementwise(
    predictions_t: fx.Tensor,
    targets_t: fx.Tensor,
    output_t: fx.Tensor,
    n: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 256
    vec_width = 1
    tile_elems = block_dim * vec_width
    grid_x = (n + tile_elems - 1) // tile_elems
    
    hinge_loss_elementwise_kernel(
        predictions_t, targets_t, output_t, block_dim, vec_width
    ).launch(grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        batch_size, input_size = predictions.shape
        
        targets_broadcast = targets.unsqueeze(0).expand(batch_size, input_size).clone()
        
        predictions_flat = predictions.reshape(-1).contiguous()
        targets_flat = targets_broadcast.reshape(-1).contiguous()
        n_elements = predictions_flat.numel()
        
        output = torch.empty_like(predictions_flat)
        
        t_pred = flyc.from_dlpack(predictions_flat).mark_layout_dynamic(leading_dim=0, divisibility=4)
        t_targ = flyc.from_dlpack(targets_flat).mark_layout_dynamic(leading_dim=0, divisibility=4)
        t_out = flyc.from_dlpack(output).mark_layout_dynamic(leading_dim=0, divisibility=4)
        
        run_hinge_loss_elementwise(
            t_pred, t_targ, t_out, n_elements,
            stream=torch.cuda.Stream()
        )
        
        torch.cuda.synchronize()
        
        return torch.mean(output)
