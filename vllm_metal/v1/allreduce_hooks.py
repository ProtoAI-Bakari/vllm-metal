"""ProtoAI-Bakari--tp_allreduce_hooks: TP all-reduce for MLX row-parallel layers.

MLX nn.Module does not support PyTorch-style register_forward_hook().
Instead, we wrap __call__ on the row-parallel Linear projections
(o_proj and down_proj) to inject all-reduce after their matmul output.

Architecture (mlx_lm Llama TransformerBlock.__call__):

    r = self.self_attn(self.input_layernorm(x), mask, cache)
        # Inside Attention.__call__:
        #   ... q/k/v_proj (column-parallel, no sync needed) ...
        #   ... attention computation ...
        #   return self.o_proj(output)  <-- ROW-PARALLEL, produces PARTIAL SUM
    h = x + r                           <-- residual add (needs FULL sum)

    r = self.mlp(self.post_attention_layernorm(h))
        # Inside MLP.__call__:
        #   ... gate_proj, up_proj (column-parallel, no sync needed) ...
        #   return self.down_proj(swiglu(...))  <-- ROW-PARALLEL, produces PARTIAL SUM
    out = h + r                         <-- residual add (needs FULL sum)

We patch o_proj.__call__ and down_proj.__call__ so the all-reduced result
flows directly into the residual add. This is more surgical than patching
the entire Attention/MLP module -- it targets exactly where the partial sum
is produced.

Usage from model_runner.py:
    from allreduce_hooks import install_tp_allreduce_hooks, remove_tp_allreduce_hooks
    install_tp_allreduce_hooks(model, tp_rank, tp_size, backend="udp", ...)
"""

import logging
import os
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All-reduce wrapper factory
# ---------------------------------------------------------------------------

def _make_gloo_allreduce_wrapper(original_call, layer_idx, proj_name):
    """Wrap a Linear.__call__ with GLOO-based all-reduce.

    Steps:
        1. Call original Linear forward (matmul) -> partial sum
        2. mx.eval to materialize the lazy MLX array
        3. MLX -> numpy (near-zero-copy on Apple Silicon unified memory)
        4. torch.distributed.all_reduce via GLOO
        5. numpy -> MLX array
    """
    import numpy as np
    import torch
    import torch.distributed as tdist

    _call_count = [0]

    def _wrapped_call(*args, **kwargs):
        partial = original_call(*args, **kwargs)

        mx.eval(partial)

        original_shape = partial.shape
        original_dtype = partial.dtype

        # MLX -> numpy -> torch (zero-copy where possible)
        np_arr = np.array(partial, copy=False)
        t = torch.from_numpy(np_arr.astype(np.float32))

        tdist.all_reduce(t, op=tdist.ReduceOp.SUM)

        # torch -> MLX
        reduced = mx.array(t.numpy()).astype(original_dtype)

        _call_count[0] += 1
        if _call_count[0] <= 2 or _call_count[0] % 100 == 0:
            logger.debug(
                "gloo all-reduce layer=%d %s call=%d shape=%s",
                layer_idx, proj_name, _call_count[0], original_shape,
            )

        return reduced

    return _wrapped_call


def _make_udp_allreduce_wrapper(original_call, layer_idx, proj_name, ar_group):
    """Wrap a Linear.__call__ with UDP ring all-reduce.

    Uses the custom udp_allreduce backend for fast all-reduce over UDP
    without the overhead of GLOO/TCP.

    Steps:
        1. Call original Linear forward (matmul) -> partial sum
        2. mx.eval to materialize
        3. MLX -> numpy via mlx_bridge (handles bf16 properly)
        4. UDP ring all-reduce in-place
        5. numpy -> MLX via mlx_bridge
    """
    from udp_allreduce.mlx_bridge import mx_to_numpy, numpy_to_mx

    _call_count = [0]

    def _wrapped_call(*args, **kwargs):
        partial = original_call(*args, **kwargs)

        mx.eval(partial)

        original_shape = partial.shape
        original_dtype_str = str(partial.dtype)

        np_partial = mx_to_numpy(partial)

        # In-place ring all-reduce (sum) over UDP
        ar_group.all_reduce_(np_partial, op="sum")

        reduced = numpy_to_mx(
            np_partial,
            target_dtype_str=original_dtype_str,
            shape=original_shape,
        )

        _call_count[0] += 1
        if _call_count[0] <= 2 or _call_count[0] % 100 == 0:
            logger.debug(
                "udp all-reduce layer=%d %s call=%d shape=%s dtype=%s",
                layer_idx, proj_name, _call_count[0],
                original_shape, original_dtype_str,
            )

        return reduced

    return _wrapped_call


def _make_mx_distributed_wrapper(original_call, layer_idx, proj_name, group=None):
    """Wrap a Linear.__call__ with mx.distributed.all_sum.

    Uses MLX's native distributed backend (requires MPI). Included for
    completeness / future use.
    """
    _call_count = [0]

    def _wrapped_call(*args, **kwargs):
        partial = original_call(*args, **kwargs)
        reduced = mx.distributed.all_sum(partial, group=group)

        _call_count[0] += 1
        if _call_count[0] <= 2 or _call_count[0] % 100 == 0:
            logger.debug(
                "mx.distributed all-reduce layer=%d %s call=%d shape=%s",
                layer_idx, proj_name, _call_count[0], partial.shape,
            )

        return reduced

    return _wrapped_call


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------

# Known attribute names for submodules across architectures
_ATTN_ATTR_NAMES = ("self_attn", "attention", "attn")
_MLP_ATTR_NAMES = ("mlp", "feed_forward", "ffn")

# Row-parallel projections that produce partial sums requiring all-reduce
_ATTN_ROW_PROJ_NAMES = ("o_proj", "out_proj", "wo")
_MLP_ROW_PROJ_NAMES = ("down_proj", "w2", "down")


def _find_layers(model) -> list:
    """Walk common model structures to find transformer layers."""
    for attr_path in [
        "model.layers",         # mlx_lm Llama: Model -> LlamaModel -> layers
        "layers",               # direct access
        "transformer.layers",   # some architectures
        "transformer.h",        # GPT-style
    ]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        except (AttributeError, TypeError):
            continue
    return []


def _find_proj(parent_module, proj_names):
    """Find the first matching projection by name on a parent module."""
    for name in proj_names:
        proj = getattr(parent_module, name, None)
        if proj is not None and hasattr(proj, "__call__"):
            return name, proj
    return None, None


def _find_submodule(layer, attr_names):
    """Get the first matching submodule from a list of attribute names."""
    for name in attr_names:
        mod = getattr(layer, name, None)
        if mod is not None:
            return name, mod
    return None, None


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def install_tp_allreduce_hooks(
    model,
    tp_rank: int,
    tp_size: int,
    backend: str = "udp",
    ar_group: Any = None,
    mx_group: Any = None,
) -> int:
    """ProtoAI-Bakari--install_tp_allreduce: wrap o_proj/down_proj with all-reduce.

    Must be called AFTER weight sharding (_shard_weights_tp) so that each
    rank's row-parallel projections produce partial sums.

    Args:
        model: The loaded MLX model (e.g., mlx_lm Model instance).
        tp_rank: This worker's tensor-parallel rank.
        tp_size: Total number of tensor-parallel ranks.
        backend: One of 'udp', 'gloo', 'mx_distributed'.
        ar_group: For 'udp' backend -- the AllReduceGroup instance.
        mx_group: For 'mx_distributed' backend -- the mx.distributed.Group.

    Returns:
        Number of projections patched (expect 2 * num_layers for standard models).
    """
    if tp_size <= 1:
        logger.debug("TP size <= 1, no all-reduce hooks needed")
        return 0

    layers = _find_layers(model)
    if not layers:
        logger.error(
            "Cannot find transformer layers in model %s -- "
            "no all-reduce hooks installed",
            type(model).__name__,
        )
        return 0

    # Select wrapper factory based on backend
    if backend == "udp":
        if ar_group is None:
            raise ValueError(
                "UDP backend requires ar_group (AllReduceGroup instance)"
            )
        def make_wrapper(orig, li, pn):
            return _make_udp_allreduce_wrapper(orig, li, pn, ar_group)
    elif backend in ("gloo", "gloo_fast"):
        def make_wrapper(orig, li, pn):
            return _make_gloo_allreduce_wrapper(orig, li, pn)
    elif backend == "mx_distributed":
        def make_wrapper(orig, li, pn):
            return _make_mx_distributed_wrapper(orig, li, pn, group=mx_group)
    else:
        raise ValueError(f"Unknown all-reduce backend: {backend}")

    n_hooked = 0

    for layer_idx, layer in enumerate(layers):
        # --- Attention o_proj ---
        # Attention.__call__ ends with: return self.o_proj(output)
        # Patching o_proj.__call__ means the all-reduced value flows into
        # TransformerBlock's residual add: h = x + r
        _, attn = _find_submodule(layer, _ATTN_ATTR_NAMES)
        if attn is not None:
            proj_name, proj = _find_proj(attn, _ATTN_ROW_PROJ_NAMES)
            if proj is not None:
                original_call = proj.__call__
                proj.__call__ = make_wrapper(original_call, layer_idx, proj_name)
                # Stash for removal / debugging
                proj._tp_original_call = original_call
                proj._tp_hooked = True
                n_hooked += 1

        # --- MLP down_proj ---
        # MLP.__call__ ends with: return self.down_proj(swiglu(...))
        # Patching down_proj.__call__ means the all-reduced value flows into
        # TransformerBlock's residual add: out = h + r
        _, mlp = _find_submodule(layer, _MLP_ATTR_NAMES)
        if mlp is not None:
            proj_name, proj = _find_proj(mlp, _MLP_ROW_PROJ_NAMES)
            if proj is not None:
                original_call = proj.__call__
                proj.__call__ = make_wrapper(original_call, layer_idx, proj_name)
                proj._tp_original_call = original_call
                proj._tp_hooked = True
                n_hooked += 1

    # For Llama 8B (32 layers), expect 64 hooks: 32 o_proj + 32 down_proj
    expected = len(layers) * 2
    if n_hooked != expected:
        logger.warning(
            "All-reduce hooks: installed %d, expected %d "
            "(some layers may have non-standard naming)",
            n_hooked, expected,
        )
    else:
        logger.info(
            "All-reduce hooks installed: %d hooks across %d layers "
            "(o_proj + down_proj per layer), backend=%s, rank=%d/%d",
            n_hooked, len(layers), backend, tp_rank, tp_size,
        )

    return n_hooked


def remove_tp_allreduce_hooks(model) -> int:
    """Remove all-reduce patches, restoring original __call__ methods.

    Returns:
        Number of projections unpatched.
    """
    layers = _find_layers(model)
    n_removed = 0

    for layer in layers:
        # Check attention and MLP submodules
        for attr_names, proj_names in [
            (_ATTN_ATTR_NAMES, _ATTN_ROW_PROJ_NAMES),
            (_MLP_ATTR_NAMES, _MLP_ROW_PROJ_NAMES),
        ]:
            _, parent = _find_submodule(layer, attr_names)
            if parent is None:
                continue
            _, proj = _find_proj(parent, proj_names)
            if proj is not None and getattr(proj, "_tp_hooked", False):
                proj.__call__ = proj._tp_original_call
                del proj._tp_original_call
                proj._tp_hooked = False
                n_removed += 1

    logger.info("Removed %d all-reduce hooks", n_removed)
    return n_removed
