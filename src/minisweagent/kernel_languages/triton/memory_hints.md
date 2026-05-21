# Triton — memory (KB) key-parameter patterns

Regex patterns the formatter uses to extract "key parameters" from
Triton kernels and experience records.  Moved out of
``memory/cross_session/formatter.py::_PARAM_PATTERNS`` so the
formatter stays language-agnostic.  One pattern per line:

    LABEL   REGEX

Lines starting with `#` are comments. The formatter parses this file
and emits `[label: <matched text>]` lines in the memory context block.

    num_warps           num_warps\s*=\s*\d+
    num_stages          num_stages\s*=\s*\d+
    BLOCK_SIZE          BLOCK_(?:T|M|N|K|D|D_HALF)\s*=\s*\d+
    XBLOCK              XBLOCK\s*[:=]\s*\d+
    dtype_cast          \.to\s*\(\s*tl\.(?:float16|float32|bfloat16|int32|int64)\s*\)
    bitcast             (?:bitcast\s*=\s*True|tl\.(?:int32|uint32)\s*,\s*bitcast)
    tl_arange_dtype     tl\.arange\s*\([^)]*\)\s*\.to\s*\(\s*tl\.(?:int32|int64)\s*\)
    prealloc_output     (?:torch\.empty|torch\.empty_like|output_tensor\.copy_|prealloc)
    tl_constexpr_dtype  tl\.constexpr\s*=\s*tl\.(?:int32|int64|float32|float16)
    cuda_graph          (?:torch\.cuda\.CUDAGraph|cudagraph|cuda_graph)
