import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir._mlir_libs._mlir.ir import VectorType


@flyc.kernel
def hinge_loss_kernel(
    predictions_ptr: fx.Tensor,
    targets_ptr: fx.Tensor,
    output_ptr: fx.Tensor,
    input_size: fx.Constexpr[int],
    n_elements: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
):
    """
    Fused kernel that computes: clamp(1 - predictions * targets, min=0)
    predictions: (batch_size, input_size) flattened
    targets: (batch_size,)
    output: (batch_size, input_size) flattened
    
    Each thread processes one element.
    """
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    
    # Make buffer tensors for efficient reads
    predictions_ptr = fx.rocdl.make_buffer_tensor(predictions_ptr)
    targets_ptr = fx.rocdl.make_buffer_tensor(targets_ptr)
    
    # Tile the tensors - one element per thread
    tPred = fx.logical_divide(predictions_ptr, fx.make_layout(block_dim, 1))
    tTarget = fx.logical_divide(targets_ptr, fx.make_layout(block_dim, 1))
    tOut = fx.logical_divide(output_ptr, fx.make_layout(block_dim, 1))
    
    # Slice by block
    tPred = fx.slice(tPred, (None, bid))
    tTarget = fx.slice(tTarget, (None, bid))
    tOut = fx.slice(tOut, (None, bid))
    
    # Further divide for per-thread access
    tPred = fx.logical_divide(tPred, fx.make_layout(1, 1))
    tTarget = fx.logical_divide(tTarget, fx.make_layout(1, 1))
    tOut = fx.logical_divide(tOut, fx.make_layout(1, 1))
    
    # Create register storage for scalar values
    RegTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
    
    copyAtomBuf = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    
    rPred = fx.memref_alloca(RegTy, fx.make_layout(1, 1))
    rTarget = fx.memref_alloca(RegTy, fx.make_layout(1, 1))
    rOut = fx.memref_alloca(RegTy, fx.make_layout(1, 1))
    
    # Load prediction and target (both are now flattened and aligned)
    fx.copy_atom_call(copyAtomBuf, fx.slice(tPred, (None, tid)), rPred)
    fx.copy_atom_call(copyAtomBuf, fx.slice(tTarget, (None, tid)), rTarget)
    
    # Load values as vectors (vector<1xf32>)
    pred_val = fx.memref_load_vec(rPred)
    target_val = fx.memref_load_vec(rTarget)
    
    # Compute: clamp(1 - pred * target, min=0)
    # prod = pred * target
    prod = fx.arith.mulf(pred_val, target_val)
    
    # diff = 1 - prod
    # Create vector constants
    vec_type = VectorType.get([1], fx.T.f32())
    one = fx.arith.constant_vector(1.0, vec_type)
    diff = fx.arith.subf(one, prod)
    
    # clamp to min=0: max(0, diff)
    zero = fx.arith.constant_vector(0.0, vec_type)
    result = fx.arith.maximumf(diff, zero)
    
    # Store result
    fx.memref_store_vec(result, rOut)
    
    # Write back to global memory
    fx.copy_atom_call(copyAtom, rOut, fx.slice(tOut, (None, tid)))


@flyc.jit
def run_hinge_loss_kernel(
    predictions_t: fx.Tensor,
    targets_t: fx.Tensor,
    output_t: fx.Tensor,
    input_size: fx.Int32,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    const_input: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 256
    # Compute grid size - round up to ensure we cover all elements
    grid_x = (n + block_dim - 1) // block_dim
    
    hinge_loss_kernel(
        predictions_t, targets_t, output_t,
        const_input, const_n, block_dim
    ).launch(grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        """
        predictions: (batch_size, input_size)
        targets: (batch_size,)
        Returns: scalar mean of hinge loss
        """
        batch_size, input_size = predictions.shape
        n_elements = batch_size * input_size
        
        # Broadcast targets to match predictions shape (column-wise broadcasting)
        # targets is (batch_size,), we need to broadcast it to (batch_size, input_size)
        # PyTorch broadcasts along the last dimension
        # Use unsqueeze and expand to create the broadcasted tensor
        targets_expanded = targets.unsqueeze(0).expand(batch_size, input_size).contiguous()
        
        # Flatten both for the kernel
        predictions_flat = predictions.reshape(-1).contiguous()
        targets_flat = targets_expanded.reshape(-1).contiguous()
        
        # Allocate output tensor for element-wise results
        output = torch.empty_like(predictions_flat)
        
        # Convert to FlyDSL tensors
        pred_t = flyc.from_dlpack(predictions_flat).mark_layout_dynamic(leading_dim=0, divisibility=4)
        target_t = flyc.from_dlpack(targets_flat).mark_layout_dynamic(leading_dim=0, divisibility=4)
        output_t = flyc.from_dlpack(output).mark_layout_dynamic(leading_dim=0, divisibility=4)
        
        # Run fused kernel
        run_hinge_loss_kernel(
            pred_t, target_t, output_t,
            input_size, n_elements,
            n_elements + 1, input_size + 1,
            stream=torch.cuda.Stream()
        )
        
        # Synchronize
        torch.cuda.synchronize()
        
        # Compute mean using PyTorch
        return torch.mean(output)
