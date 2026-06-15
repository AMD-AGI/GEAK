"""No-graph-wrapper FlyDSL seam: bind the BARE fused fp8 core (no host-side
graph_replay capture, which is illegal inside sglang's decode CUDA-graph)."""
import importlib, importlib.abc, importlib.util, sys
_TARGET="aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale"
_ALIAS="aiter.ops.triton.gemm_a8w8_blockscale"
_ATTR="gemm_a8w8_blockscale"
_FP8_UTILS="sglang.srt.layers.quantization.fp8_utils"
_FP8_NAMES=("triton_gemm_a8w8_blockscale","gemm_a8w8_blockscale_bpreshuffle")
_state={"a":False,"f":False}
_cache={"fn":None}
def _impl():
    if _cache["fn"] is not None: return _cache["fn"]
    from gemm_a8w8_blockscale_flydsl import gemm_a8w8_blockscale as core
    _cache["fn"]=core
    return core
def _patch_aiter(mod):
    fn=_impl(); setattr(mod,_ATTR,fn)
    al=sys.modules.get(_ALIAS)
    if al is not None and al is not mod: setattr(al,_ATTR,fn)
    if not _state["a"]:
        _state["a"]=True
        print("[flydsl-overlay-nograph] bound BARE FlyDSL core over aiter triton symbol",flush=True)
def _patch_fp8(mod):
    present=[n for n in _FP8_NAMES if hasattr(mod,n)]
    if not present: return
    fn=_impl()
    for n in present: setattr(mod,n,fn)
    if not _state["f"]:
        _state["f"]=True
        print(f"[flydsl-overlay-nograph] rebound fp8_utils {present} -> BARE FlyDSL core",flush=True)
class _WL(importlib.abc.Loader):
    def __init__(s,i,a): s._i=i; s._a=a
    def create_module(s,spec): return s._i.create_module(spec)
    def exec_module(s,m):
        s._i.exec_module(m)
        try: s._a(m)
        except Exception as e: print(f"[flydsl-overlay-nograph] post-exec fail: {e!r}",flush=True)
class _F(importlib.abc.MetaPathFinder):
    _H={_TARGET:_patch_aiter,_FP8_UTILS:_patch_fp8}
    def find_spec(s,fn,path=None,target=None):
        a=s._H.get(fn)
        if a is None: return None
        try: sys.meta_path.remove(s)
        except ValueError: pass
        try: spec=importlib.util.find_spec(fn)
        except Exception: spec=None
        finally: sys.meta_path.insert(0,s)
        if spec is None or spec.loader is None: return None
        spec.loader=_WL(spec.loader,a); return spec
_ea=sys.modules.get(_TARGET) or sys.modules.get(_ALIAS)
if _ea is not None: _patch_aiter(_ea)
_ef=sys.modules.get(_FP8_UTILS)
if _ef is not None: _patch_fp8(_ef)
if _ea is None or _ef is None: sys.meta_path.insert(0,_F())
