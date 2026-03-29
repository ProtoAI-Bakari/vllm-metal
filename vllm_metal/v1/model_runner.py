# SPDX-License-Identifier: Apache-2.0
"""Metal Model Runner for vLLM v1 engine.

Optimized for performance with:
- True batched decode using BatchKVCache for O(1) forward passes per batch
- Async evaluation pipeline for pipelined computation
- Pre-allocated input buffers to reduce allocation overhead
- Rust-based token state management for efficient batch operations
- Global model cache for fast repeated loads
- Content hash prefix caching for shared prompt reuse
"""

import hashlib
import math
import os
import time
from array import array
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeAlias

import numpy as np
import mlx.core as mx
import torch
from mlx_lm import load as mlx_lm_load
from mlx_lm import stream_generate
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)

# mlx_vlm for vision-language models
from mlx_vlm import load as mlx_vlm_load
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_metal.config import get_config
from vllm_metal.paged_attention_common import (
    OffsetCache,
    clear_context,
    prepare_decode,
    prepare_prefill,
)
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.utils import get_model_download_path

logger = init_logger(__name__)

# Global model cache for fast repeated loads
_model_cache: dict[str, tuple[Any, Any]] = {}  # model_name -> (model, tokenizer)
_model_cache_lock = Lock()

# Try to import MLX distributed for tensor-parallel sharding
# Note: mlx.distributed does not exist as a standalone module in mlx 0.30+;
# the distributed API lives at mlx.core.distributed (accessed as mx.distributed).
try:
    import mlx.core as _mlx_core
    dist = _mlx_core.distributed
    from mlx.nn.layers.distributed import shard_inplace, shard_linear

    HAS_MLX_DISTRIBUTED = True
except (ImportError, AttributeError):
    HAS_MLX_DISTRIBUTED = False

# Try to import Rust extension for high-performance token state management
try:
    from vllm_metal._rs import RequestStateManager as RustRequestStateManager

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    logger.debug("Rust extension not available, using Python fallback")

# Configuration for batched operations
_MIN_BATCH_SIZE_FOR_BATCHING = 2  # Minimum requests to use BatchKVCache
_MAX_BATCH_SIZE = 64  # Maximum batch size for decode

# Performance tuning
_CACHE_CLEAR_INTERVAL = 50  # Clear cache every N finished requests

# Prefix cache configuration — enabled by setting VLLM_METAL_PREFIX_CACHE
# in the environment (any value; unset to disable).


def _prefix_cache_enabled() -> bool:
    """Check whether prefix caching is enabled via environment variable."""
    return "VLLM_METAL_PREFIX_CACHE" in os.environ


_PREFIX_CACHE_ENABLED = _prefix_cache_enabled()
_PREFIX_CACHE_DEFAULT_FRACTION = 0.05  # 5% of MLX working set


def _get_prefix_cache_max_bytes() -> int:
    """Get prefix cache memory limit based on MLX recommended working set."""
    fraction_str = os.environ.get("VLLM_METAL_PREFIX_CACHE_FRACTION", "")
    if fraction_str:
        try:
            fraction = float(fraction_str)
            if not math.isfinite(fraction) or fraction <= 0 or fraction > 1:
                logger.warning(
                    "VLLM_METAL_PREFIX_CACHE_FRACTION=%r out of range (0, 1], "
                    "using default %.2f",
                    fraction_str,
                    _PREFIX_CACHE_DEFAULT_FRACTION,
                )
                fraction = _PREFIX_CACHE_DEFAULT_FRACTION
        except ValueError:
            logger.warning(
                "Invalid VLLM_METAL_PREFIX_CACHE_FRACTION=%r, using default %.2f",
                fraction_str,
                _PREFIX_CACHE_DEFAULT_FRACTION,
            )
            fraction = _PREFIX_CACHE_DEFAULT_FRACTION
    else:
        fraction = _PREFIX_CACHE_DEFAULT_FRACTION

    fallback_bytes = 8 * 1024 * 1024 * 1024  # 8 GB
    try:
        device_info = mx.metal.device_info()
        total = int(device_info.get("max_recommended_working_set_size", 0))
    except (AttributeError, RuntimeError):
        total = 0

    if total == 0:
        total = fallback_bytes
        logger.warning("Could not get MLX working set size, using 8GB fallback")

    max_bytes = int(total * fraction)
    logger.info(
        "Prefix cache: %.1fGB limit (%.1f%% of %.1fGB MLX working set)",
        max_bytes / (1024 * 1024 * 1024),
        fraction * 100,
        total / (1024 * 1024 * 1024),
    )
    return max_bytes


def _compute_prefix_hash(token_ids: list[int]) -> bytes:
    """Compute content hash for a token sequence."""
    h = hashlib.sha256()
    h.update(array("I", token_ids).tobytes())
    return h.digest()


def _compute_entry_bytes(cache_state: list[tuple[mx.array, mx.array] | None]) -> int:
    """Compute memory usage of a cache entry in bytes."""
    total = 0
    for pair in cache_state:
        if pair is not None:
            total += pair[0].nbytes + pair[1].nbytes
    return total


@dataclass
class CachedPrefix:
    """Cached KV state for a token prefix.

    cache_state contains (k, v) tuples for KVCache layers, or None for
    ArraysCache layers in hybrid models.
    """

    token_ids: list[int]
    cache_state: list[tuple[mx.array, mx.array] | None]
    size_bytes: int = 0
    ref_count: int = 0


class PrefixCacheManager:
    """Manager for prefix KV cache reuse with memory-based eviction."""

    def __init__(self, max_bytes: int | None = None):
        self._cache: dict[bytes, CachedPrefix] = {}
        self._max_bytes = (
            max_bytes if max_bytes is not None else _get_prefix_cache_max_bytes()
        )
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0

    def lookup(self, token_ids: list[int]) -> CachedPrefix | None:
        """Look up cached prefix by token IDs."""
        prefix_hash = _compute_prefix_hash(token_ids)
        cached = self._cache.get(prefix_hash)
        if cached is not None:
            self._hits += 1
            cached.ref_count += 1
            logger.debug(
                "Prefix cache HIT: %d hits, %d misses, rate=%.1f%%",
                self._hits,
                self._misses,
                self.hit_rate * 100,
            )
            return cached
        self._misses += 1
        logger.debug(
            "Prefix cache MISS: %d hits, %d misses, rate=%.1f%%",
            self._hits,
            self._misses,
            self.hit_rate * 100,
        )
        return None

    def _evict_until_fits(self, needed_bytes: int) -> None:
        """Evict entries until we have room for needed_bytes."""
        while self._current_bytes + needed_bytes > self._max_bytes and self._cache:
            min_hash, min_entry = min(self._cache.items(), key=lambda x: x[1].ref_count)
            self._current_bytes -= min_entry.size_bytes
            del self._cache[min_hash]
            logger.debug(
                "Prefix cache eviction: freed %.1fMB",
                min_entry.size_bytes / (1024 * 1024),
            )

    def insert(self, token_ids: list[int], cache: list[KVCache]) -> None:
        """Insert a prefix cache entry with memory-based eviction.

        Only KVCache layers are cached. ArraysCache layers are skipped (stored as
        None) for hybrid model compatibility.
        """
        prefix_hash = _compute_prefix_hash(token_ids)
        if prefix_hash in self._cache:
            return

        cache_state = []
        for layer_cache in cache:
            if isinstance(layer_cache, KVCache):
                k = layer_cache.state[0]
                v = layer_cache.state[1]
                cache_state.append((mx.array(k), mx.array(v)))
            else:
                cache_state.append(None)

        entry_bytes = _compute_entry_bytes(cache_state)

        # Skip if single entry exceeds memory limit
        if entry_bytes > self._max_bytes:
            logger.debug(
                "Prefix cache skip: entry %.1fMB exceeds limit %.1fGB",
                entry_bytes / (1024 * 1024),
                self._max_bytes / (1024 * 1024 * 1024),
            )
            return

        self._evict_until_fits(entry_bytes)

        self._cache[prefix_hash] = CachedPrefix(
            token_ids=list(token_ids),
            cache_state=cache_state,
            size_bytes=entry_bytes,
            ref_count=1,
        )
        self._current_bytes += entry_bytes

    def restore_cache(
        self, cached: CachedPrefix, model: Any, is_vlm: bool
    ) -> list["AnyCache"]:
        """Restore a cached prefix to a fresh KVCache.

        Only KVCache layers are restored. RotatingKVCache / ArraysCache layers
        remain in their fresh state.
        """
        cache_model = (
            model.language_model
            if is_vlm and hasattr(model, "language_model")
            else model
        )
        cache = make_prompt_cache(cache_model)
        for i, layer_cache in enumerate(cache):
            if i < len(cached.cache_state) and cached.cache_state[i] is not None:
                if isinstance(layer_cache, KVCache):
                    k, v = cached.cache_state[i]
                    layer_cache.state = [mx.array(k), mx.array(v)]
        return cache

    @property
    def hit_rate(self) -> float:
        """Return prefix cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Return prefix cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "cached_entries": len(self._cache),
            "current_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
        }


# Type alias for any per-layer cache type supported by the model.
#
# Notes:
# - Some models (e.g. gpt_oss) use `RotatingKVCache` for sliding-window attention.
# - Hybrid models use `ArraysCache` for non-attention state.
AnyCache: TypeAlias = KVCache | RotatingKVCache | ArraysCache


def _merge_arrays_caches(caches: list[ArraysCache]) -> ArraysCache:
    """Merge per-request ArraysCache objects into a single batched ArraysCache.

    This mirrors the behavior of `mlx_lm.models.cache.ArraysCache.merge` but is
    implemented here for compatibility with older mlx-lm versions that do not
    provide `merge()` / `extract()`.
    """
    if not caches:
        raise ValueError("caches must be non-empty")

    num_entries = len(caches[0].state)
    batch_size = len(caches)

    merged = ArraysCache(num_entries)
    for entry_idx in range(num_entries):
        values = [cache.state[entry_idx] for cache in caches]
        template = next((value for value in values if value is not None), None)
        if template is None:
            continue

        shape = list(template.shape)
        shape[0] = batch_size
        merged_state = mx.zeros(tuple(shape), template.dtype)
        for batch_idx, value in enumerate(values):
            if value is None:
                continue
            merged_state[batch_idx : batch_idx + 1] = value

        merged[entry_idx] = merged_state

    return merged


def _extract_arrays_cache(batch_cache: ArraysCache, idx: int) -> ArraysCache:
    """Extract a single request's ArraysCache from a batched ArraysCache."""
    state = batch_cache.state
    extracted = ArraysCache(len(state))
    extracted.state = [
        None if value is None else value[idx : idx + 1] for value in state
    ]
    return extracted


def _merge_rotating_kv_caches(
    caches: list[RotatingKVCache],
) -> BatchRotatingKVCache:
    """Merge per-request RotatingKVCache objects into a single BatchRotatingKVCache.

    This mirrors ``BatchRotatingKVCache.merge`` but pre-computes the temporal-ordered
    keys/values, trims them to ``len(cache)`` (the effective sliding-window length),
    and uses that length for the copy width.  The upstream implementation in
    mlx-lm <= 0.29.1 uses ``c.offset`` which can exceed the underlying array size
    after the cache has rotated, causing a broadcast shape error.

    ProtoAI-Bakari--rotating_kv_cache_offset_fix: upstream compatibility shim
    for mlx-lm <= 0.29.1 offset overflow (ml-explore/mlx-lm#738).
    Remove once vllm-metal can depend on a corrected mlx-lm release that has
    been verified end-to-end with gpt-oss models.
    """
    if not caches:
        raise ValueError("caches must be non-empty")

    if any(c.keys is None or c.values is None for c in caches):
        raise ValueError(
            "Cannot merge unpopulated RotatingKVCache (keys/values is None)"
        )

    if not all(c.max_size == caches[0].max_size for c in caches):
        raise ValueError(
            "BatchRotatingKVCache can only merge caches with the same maximum size"
        )

    # Pre-compute temporal-ordered keys/values and trim to the effective
    # sliding-window length.  ``_temporal_order`` may return an array larger
    # than ``len(cache)`` when the internal buffer has not been trimmed yet
    # (e.g. after a large prefill), so we trim via ``_trim`` to preserve
    # the ``keep`` prefix semantics used by RotatingKVCache internally.
    ordered: list[tuple[mx.array, mx.array]] = []
    for c in caches:
        effective_len = len(c)  # min(offset, max_size)
        ordered_keys = c._temporal_order(c.keys)
        ordered_values = c._temporal_order(c.values)
        if ordered_keys.shape[2] > effective_len:
            trim_size = ordered_keys.shape[2] - effective_len
            ordered_keys = c._trim(trim_size, ordered_keys)
            ordered_values = c._trim(trim_size, ordered_values)
        else:
            ordered_keys = ordered_keys[..., :effective_len, :]
            ordered_values = ordered_values[..., :effective_len, :]
        ordered.append((ordered_keys, ordered_values))

    lengths = [k.shape[2] for k, _ in ordered]
    max_length = max(lengths)
    padding = [max_length - length for length in lengths]
    batch_size = len(caches)
    n_heads = max(k.shape[1] for k, _ in ordered)
    k_dim = max(k.shape[3] for k, _ in ordered)
    v_dim = max(v.shape[3] for _, v in ordered)
    dtype = next(iter(k.dtype for k, _ in ordered))

    keys = mx.zeros((batch_size, n_heads, max_length, k_dim), dtype=dtype)
    values = mx.zeros((batch_size, n_heads, max_length, v_dim), dtype=dtype)
    for i, (pad, (k, v)) in enumerate(zip(padding, ordered, strict=True)):
        n = k.shape[2]
        keys[i : i + 1, :, pad : pad + n] = k
        values[i : i + 1, :, pad : pad + n] = v

    cache = BatchRotatingKVCache(caches[0].max_size, padding)
    cache.keys = keys
    cache.values = values
    cache.offset = mx.array([c.offset for c in caches])
    cache._idx = keys.shape[2]
    cache._offset = keys.shape[2]

    return cache


def _mlx_greedy_sample(logits: mx.array) -> mx.array:
    """Native MLX greedy sampling - avoids PyTorch round-trip.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)

    Returns:
        Token IDs of shape (batch_size,)
    """
    return mx.argmax(logits, axis=-1)


def _create_request_generator(
    device: torch.device,
    sampling_params: SamplingParams,
) -> torch.Generator | None:
    """Create a per-request generator for seeded sampling.

    vLLM uses a per-request generator only when an explicit seed is provided.
    For unseeded sampling, vLLM relies on the global RNG state.
    """
    if sampling_params.seed is None:
        return None
    if sampling_params.temperature < 1e-5:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(sampling_params.seed)
    return generator


@dataclass
class SamplerOutput:
    """Output from the sampler."""

    token_ids: list[int]
    logprobs: list[float] | None = None


@dataclass
class RequestState:
    """State for an ongoing request with KV cache."""

    token_ids: list[int]
    # Length of the original prompt (prefix) within `token_ids`.
    # vLLM applies repetition penalties to both prompt+output tokens, but applies
    # presence/frequency penalties only to generated (output) tokens.
    prompt_len: int
    cache: list[AnyCache]  # Per-layer caches (KVCache, RotatingKVCache, or ArraysCache)
    sampling_params: SamplingParams  # Sampling parameters for this request
    generator: torch.Generator | None = None
    generated_tokens: int = 0


def _merge_kv_caches(
    caches_list: list[list[AnyCache]],
) -> list[BatchKVCache | BatchRotatingKVCache | ArraysCache]:
    """Merge multiple per-request caches into batched caches.

    Args:
        caches_list: List of per-request caches, each is a list of per-layer caches

    Returns:
        List of batched caches, one per layer
    """
    if not caches_list:
        return []

    num_layers = len(caches_list[0])
    merged: list[BatchKVCache | BatchRotatingKVCache | ArraysCache] = []

    for layer_idx in range(num_layers):
        layer_caches = [caches[layer_idx] for caches in caches_list]
        if isinstance(layer_caches[0], ArraysCache):
            arrays_caches: list[ArraysCache] = []
            for cache in layer_caches:
                if not isinstance(cache, ArraysCache):
                    raise TypeError(
                        "Mixed cache types in a single layer: expected ArraysCache"
                    )
                arrays_caches.append(cache)
            batch_cache = _merge_arrays_caches(arrays_caches)
        elif isinstance(layer_caches[0], RotatingKVCache):
            rotating_caches: list[RotatingKVCache] = []
            for cache in layer_caches:
                if not isinstance(cache, RotatingKVCache):
                    raise TypeError(
                        "Mixed cache types in a single layer: expected RotatingKVCache"
                    )
                rotating_caches.append(cache)
            batch_cache = _merge_rotating_kv_caches(rotating_caches)
        elif isinstance(layer_caches[0], KVCache):
            kv_caches: list[KVCache] = []
            for cache in layer_caches:
                if not isinstance(cache, KVCache):
                    raise TypeError(
                        "Mixed cache types in a single layer: expected KVCache"
                    )
                kv_caches.append(cache)
            batch_cache = BatchKVCache.merge(kv_caches)
        else:
            cache_type = type(layer_caches[0]).__name__
            raise TypeError(f"Unsupported cache type for batching: {cache_type}")
        merged.append(batch_cache)

    return merged


def _extract_kv_cache(
    batch_caches: list[BatchKVCache | BatchRotatingKVCache | ArraysCache], idx: int
) -> list[AnyCache]:
    """Extract a single request's cache from batched caches.

    Args:
        batch_caches: List of batched caches, one per layer
        idx: Index of the request in the batch

    Returns:
        List of caches for the request, one per layer
    """
    extracted: list[AnyCache] = []
    for cache in batch_caches:
        if isinstance(cache, ArraysCache):
            extracted.append(_extract_arrays_cache(cache, idx))
        else:
            extracted.append(cache.extract(idx))
    return extracted


class MetalModelRunner:
    """Model runner for MLX-based inference on Metal.

    Implements the vLLM v1 model runner interface for Apple Silicon.
    Uses true batched decode with BatchKVCache for efficient parallel processing.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """Initialize model runner.

        Args:
            vllm_config: vLLM configuration
            device: PyTorch device (CPU for Metal interop)
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = bool(self.scheduler_config.async_scheduling)
        self.device = device
        self.metal_config = get_config()

        self.model: Any = None
        self.tokenizer: Any = None
        self.model_args: dict[str, Any] = {}
        self._is_vlm: bool = False  # Will be set during model loading

        # Request state cache for incremental decoding
        self._request_states: dict[str, RequestState] = {}

        # Rust-based token state manager (optional, for batch operations)
        self._rust_state_manager: Any = None
        if _RUST_AVAILABLE:
            self._rust_state_manager = RustRequestStateManager()

        # Pre-allocated buffer for decode input tokens
        self._max_batch_size = _MAX_BATCH_SIZE

        # vLLM Sampler for token sampling with temperature, top_k, top_p support
        self._sampler = Sampler()

        # Track finished requests for lazy cache clearing
        self._finished_request_count = 0

        # vLLM v1 async scheduling calls sample_tokens after execute_model.
        # Keep the latest execution output so sample_tokens can return it.
        self._pending_output: ModelRunnerOutput | None = None

        # Prefix cache for shared prompt reuse
        self._prefix_cache: PrefixCacheManager | None = None
        if _PREFIX_CACHE_ENABLED:
            self._prefix_cache = PrefixCacheManager()

        # Paged attention state (set by worker when enabled)
        self._paged_kv_cache: Any = None  # MPSPagedKVCache, set by worker
        self._paged_block_size: int = 0
        self._paged_request_seq_lens: dict[str, int] = {}  # req_id → seq_len

        # Tensor parallel rank/size for weight sharding
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.tp_rank = 0  # Updated in load_model after distributed init
        logger.info("ProtoAI-Bakari--init: tp_size=%d from parallel_config", self.tp_size)

    def _is_vlm_model(self) -> bool:
        """Check if the model is a vision-language model (VLM).

        Returns:
            True if the model is multimodal/VLM, False otherwise
        """
        # Check vLLM's multimodal detection
        if hasattr(self.model_config, "is_multimodal_model"):
            return self.model_config.is_multimodal_model
        return False

    def load_model(self) -> None:
        """Load the model using MLX with caching for fast repeated loads.

        Uses mlx_vlm for vision-language models and mlx_lm for text-only models.
        """
        model_name = get_model_download_path(self.model_config.model)
        is_vlm = self._is_vlm_model()

        logger.info(f"Loading model: {model_name} (VLM: {is_vlm})")
        start_time = time.time()

        # Check global cache first for fast repeated loads
        with _model_cache_lock:
            if model_name in _model_cache:
                self.model, self.tokenizer = _model_cache[model_name]
                load_time = time.time() - start_time
                logger.info(
                    f"Model loaded from cache in {load_time:.3f}s: {model_name}"
                )
                self._extract_model_args()
                self._resolve_model_dims()
                return

        # Load model using appropriate backend
        if is_vlm:
            logger.info("Using mlx-vlm for vision-language model")
            self.model, self.tokenizer = mlx_vlm_load(model_name)
            self._is_vlm = True
        else:
            # Load model and tokenizer using mlx_lm for text-only models
            self.model, self.tokenizer = mlx_lm_load(
                model_name,
                tokenizer_config={
                    "trust_remote_code": self.model_config.trust_remote_code
                },
            )
            self._is_vlm = False

        # Cache for future loads
        with _model_cache_lock:
            _model_cache[model_name] = (self.model, self.tokenizer)

        self._extract_model_args()
        self._resolve_model_dims()

        # Apply tensor-parallel sharding if multi-device.
        # _shard_model_tp() uses MLX distributed (replaces layers with sharded
        # variants).  _shard_weights_tp() manually slices weight tensors for
        # Ray/torch.distributed TP.  Only ONE path should run — if MLX
        # distributed sharding succeeds, manual weight slicing would double-
        # shard (size/tp/tp = garbage).
        #
        # BUG FIX (claude-vlm 2026-03-28): When MLX distributed is importable
        # but the ring backend isn't initialized for multi-node, dist.init()
        # returns size=1 and _shard_model_tp() skips sharding. Must fall back
        # to manual _shard_weights_tp() in this case.
        # Always use manual weight sharding — it's proven correct with UDP
        # allreduce. MLX shard_linear bundles sharding+allreduce and needs
        # perfect ring init synchronization between workers, which is fragile
        # with Ray's async worker startup. Manual sharding + hooks is reliable.
        self._shard_weights_tp()

        # Install all-reduce hooks for row-parallel partial sums.
        # The hooks will try native mx.distributed.all_sum() first (if the
        # ring backend initialized), falling back to UDP.
        self._install_tp_allreduce_hooks()

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s: {model_name}")

    def _extract_model_args(self) -> None:
        """Extract model configuration from loaded model.

        Handles both text-only models and VLMs (which have nested text_config).
        """
        if hasattr(self.model, "args"):
            # mlx-lm models (Qwen, Llama, etc.)
            self.model_args = vars(self.model.args)
        elif hasattr(self.model, "config"):
            config = self.model.config
            if self._is_vlm and hasattr(config, "text_config"):
                # VLMs with nested text config (LLaVA, Pixtral via mlx-vlm)
                text_config = config.text_config
                if hasattr(text_config, "to_dict"):
                    self.model_args = text_config.to_dict()
                else:
                    self.model_args = {
                        k: getattr(text_config, k)
                        for k in dir(text_config)
                        if not k.startswith("_")
                        and not callable(getattr(text_config, k))
                    }
            elif hasattr(config, "to_dict"):
                # Standard HuggingFace config objects
                self.model_args = config.to_dict()
            else:
                self.model_args = vars(config)
        else:
            raise ValueError(
                "Cannot extract model config: model has neither .args nor "
                ".config attribute."
            )
        if self.metal_config.debug:
            logger.info(f"Model args: {self.model_args}")

    def _resolve_model_dims(self) -> None:
        """Extract and validate model dimensions from ``self.model_args``.

        Must be called after ``_extract_model_args()``.  Stores validated
        dimensions as instance attributes so that every consumer reads from
        one canonical source instead of repeating fallback chains.

        Raises:
            ValueError: If any critical dimension cannot be determined.
        """
        args = self.model_args

        num_layers = args.get("num_hidden_layers") or args.get("n_layers")
        num_attention_heads = args.get("num_attention_heads")
        num_kv_heads = (
            args.get("num_key_value_heads")
            or args.get("n_kv_heads")
            or num_attention_heads
        )
        hidden_size = args.get("hidden_size")
        head_dim = args.get("head_dim") or (
            hidden_size // num_attention_heads
            if hidden_size and num_attention_heads
            else None
        )

        # Fail fast if critical dims are missing
        missing = []
        if not num_layers:
            missing.append("num_layers (num_hidden_layers / n_layers)")
        if not num_kv_heads:
            missing.append("num_kv_heads (num_key_value_heads / n_kv_heads)")
        if not head_dim:
            missing.append("head_dim")
        if missing:
            raise ValueError(
                f"Cannot resolve model dimensions: {', '.join(missing)}. "
                f"Available keys: {sorted(args.keys())}"
            )

        self.num_layers: int = int(num_layers)
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads: int = int(num_kv_heads)
        self.hidden_size = hidden_size
        self.head_dim: int = int(head_dim)

    def _shard_model_tp(self) -> None:
        """Apply tensor-parallel sharding to model weights using MLX distributed.

        Converts linear layers to distributed variants:
        - Column-parallel (all-to-sharded): QKV projections, gate/up MLP
        - Row-parallel (sharded-to-all): output projection, down MLP
        """
        if not HAS_MLX_DISTRIBUTED:
            logger.warning(
                "mlx.distributed not available, skipping TP sharding"
            )
            return

        group = dist.init()
        tp_size = group.size()
        rank = group.rank()

        if tp_size <= 1:
            logger.info("TP size is 1, skipping sharding")
            return

        logger.info(
            "Applying MLX tensor-parallel sharding: tp_size=%d, rank=%d",
            tp_size,
            rank,
        )

        model = self.model
        n_sharded = 0

        # Find the transformer layers - try common attribute paths
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "transformer") and hasattr(
            model.transformer, "layers"
        ):
            layers = model.transformer.layers

        if layers is None:
            logger.error("Cannot find transformer layers for TP sharding")
            return

        for i, layer in enumerate(layers):
            attn = None
            mlp = None

            # Find attention module
            for attn_name in ("self_attn", "attention", "attn"):
                if hasattr(layer, attn_name):
                    attn = getattr(layer, attn_name)
                    break

            # Find MLP module
            for mlp_name in ("mlp", "feed_forward", "ffn"):
                if hasattr(layer, mlp_name):
                    mlp = getattr(layer, mlp_name)
                    break

            if attn is not None:
                # Column-parallel: Q, K, V projections
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    if hasattr(attn, proj_name):
                        setattr(
                            attn,
                            proj_name,
                            shard_linear(
                                getattr(attn, proj_name),
                                "all-to-sharded",
                                group=group,
                            ),
                        )
                        n_sharded += 1
                # Fused QKV
                if hasattr(attn, "qkv_proj"):
                    attn.qkv_proj = shard_linear(
                        attn.qkv_proj,
                        "all-to-sharded",
                        segments=3,
                        group=group,
                    )
                    n_sharded += 1
                # Row-parallel: output projection
                for proj_name in ("o_proj", "out_proj", "dense"):
                    if hasattr(attn, proj_name):
                        shard_inplace(
                            getattr(attn, proj_name),
                            "sharded-to-all",
                            group=group,
                        )
                        n_sharded += 1
                        break

            if mlp is not None:
                # Column-parallel: gate and up projections
                for proj_name in ("gate_proj", "w1", "gate"):
                    if hasattr(mlp, proj_name):
                        setattr(
                            mlp,
                            proj_name,
                            shard_linear(
                                getattr(mlp, proj_name),
                                "all-to-sharded",
                                group=group,
                            ),
                        )
                        n_sharded += 1
                        break
                for proj_name in ("up_proj", "w3", "up"):
                    if hasattr(mlp, proj_name):
                        setattr(
                            mlp,
                            proj_name,
                            shard_linear(
                                getattr(mlp, proj_name),
                                "all-to-sharded",
                                group=group,
                            ),
                        )
                        n_sharded += 1
                        break
                # Fused gate_up
                if hasattr(mlp, "gate_up_proj"):
                    mlp.gate_up_proj = shard_linear(
                        mlp.gate_up_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    n_sharded += 1
                # Row-parallel: down projection
                for proj_name in ("down_proj", "w2", "down"):
                    if hasattr(mlp, proj_name):
                        shard_inplace(
                            getattr(mlp, proj_name),
                            "sharded-to-all",
                            group=group,
                        )
                        n_sharded += 1
                        break

        # Adjust head counts for tensor parallelism
        if hasattr(self, "num_attention_heads") and self.num_attention_heads:
            self.num_attention_heads = self.num_attention_heads // tp_size
        if hasattr(self, "num_kv_heads"):
            self.num_kv_heads = self.num_kv_heads // tp_size

        logger.info(
            "TP sharding complete: %d layers sharded across %d devices",
            n_sharded,
            tp_size,
        )

    def _shard_weights_tp(self) -> None:
        """Post-load weight sharding for tensor parallelism.

        After mlx_lm loads the full model, slice each linear layer's weights
        to keep only this rank's portion. This reduces memory by 1/tp_size.
        """
        logger.info("ProtoAI-Bakari--_shard_weights_tp: ENTERED, tp_size=%d", self.tp_size)
        if self.tp_size <= 1:
            logger.info("ProtoAI-Bakari--_shard_weights_tp: SKIPPED (tp_size=%d <= 1)", self.tp_size)
            return

        # Get rank from torch.distributed (initialized by Ray/GLOO)
        try:
            import torch.distributed as tdist
            if tdist.is_initialized():
                self.tp_rank = tdist.get_rank()
            else:
                logger.warning("torch.distributed not initialized, cannot shard weights")
                return
        except Exception as e:
            logger.warning("Cannot get TP rank: %s", e)
            return

        logger.info("Sharding weights: tp_rank=%d, tp_size=%d", self.tp_rank, self.tp_size)

        # Find transformer layers
        layers = None
        model = self.model
        for attr_path in ['model.layers', 'layers', 'transformer.layers']:
            obj = model
            try:
                for part in attr_path.split('.'):
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue

        if layers is None:
            logger.error("Cannot find transformer layers for TP sharding")
            return

        n_sharded = 0
        tp = self.tp_size
        rank = self.tp_rank

        def _shard_col(proj, tp, rank):
            """Column-parallel: slice dim=0 (output features). Handles quantized scales/biases."""
            w = proj.weight
            chunk = w.shape[0] // tp
            proj.weight = mx.array(w[rank * chunk:(rank + 1) * chunk])
            if hasattr(proj, 'bias') and proj.bias is not None:
                proj.bias = mx.array(proj.bias[rank * chunk:(rank + 1) * chunk])
            if hasattr(proj, 'scales') and proj.scales is not None:
                s = proj.scales
                s_chunk = s.shape[0] // tp
                proj.scales = mx.array(s[rank * s_chunk:(rank + 1) * s_chunk])
            if hasattr(proj, 'biases') and proj.biases is not None:
                b = proj.biases
                b_chunk = b.shape[0] // tp
                proj.biases = mx.array(b[rank * b_chunk:(rank + 1) * b_chunk])

        def _shard_row(proj, tp, rank):
            """Row-parallel: slice dim=1 (input features). Handles quantized scales/biases."""
            w = proj.weight
            chunk = w.shape[1] // tp
            proj.weight = mx.array(w[:, rank * chunk:(rank + 1) * chunk])
            # Row-parallel: scales/biases slice on dim=1 too
            if hasattr(proj, 'scales') and proj.scales is not None:
                s = proj.scales
                s_chunk = s.shape[1] // tp
                proj.scales = mx.array(s[:, rank * s_chunk:(rank + 1) * s_chunk])
            if hasattr(proj, 'biases') and proj.biases is not None:
                b = proj.biases
                b_chunk = b.shape[1] // tp
                proj.biases = mx.array(b[:, rank * b_chunk:(rank + 1) * b_chunk])
            # Row-parallel bias fix: each rank computes partial = x_shard @ W.T + bias.
            # After allreduce sum, bias gets multiplied by tp_size.
            # Fix: only rank 0 keeps the bias; others zero it out.
            if rank > 0 and hasattr(proj, 'bias') and proj.bias is not None:
                proj.bias = mx.zeros_like(proj.bias)

        for layer in layers:
            # Column-parallel: slice output dim (dim=0 of weight)
            for module_name in ['self_attn', 'attention', 'attn']:
                attn = getattr(layer, module_name, None)
                if attn is None:
                    continue
                for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                    proj = getattr(attn, proj_name, None)
                    if proj is not None and hasattr(proj, 'weight'):
                        _shard_col(proj, tp, rank)
                        n_sharded += 1
                # Fused QKV
                qkv = getattr(attn, 'qkv_proj', None)
                if qkv is not None and hasattr(qkv, 'weight'):
                    w = qkv.weight
                    # Fused QKV: split into 3 segments, shard each, recombine
                    seg_size = w.shape[0] // 3
                    chunk = seg_size // tp
                    segments = []
                    for s in range(3):
                        seg = w[s * seg_size:(s + 1) * seg_size]
                        segments.append(seg[rank * chunk:(rank + 1) * chunk])
                    qkv.weight = mx.concatenate(segments, axis=0)
                    n_sharded += 1
                break

            # Row-parallel: slice input dim (dim=1 of weight) for output proj
            for module_name in ['self_attn', 'attention', 'attn']:
                attn = getattr(layer, module_name, None)
                if attn is None:
                    continue
                for proj_name in ['o_proj', 'out_proj']:
                    proj = getattr(attn, proj_name, None)
                    if proj is not None and hasattr(proj, 'weight'):
                        _shard_row(proj, tp, rank)
                        n_sharded += 1
                        break
                break

            # MLP column-parallel: gate and up
            for mlp_name in ['mlp', 'feed_forward', 'ffn']:
                mlp = getattr(layer, mlp_name, None)
                if mlp is None:
                    continue
                for proj_name in ['gate_proj', 'w1', 'gate']:
                    proj = getattr(mlp, proj_name, None)
                    if proj is not None and hasattr(proj, 'weight'):
                        _shard_col(proj, tp, rank)
                        n_sharded += 1
                        break
                for proj_name in ['up_proj', 'w3', 'up']:
                    proj = getattr(mlp, proj_name, None)
                    if proj is not None and hasattr(proj, 'weight'):
                        _shard_col(proj, tp, rank)
                        n_sharded += 1
                        break
                # Fused gate_up
                fused = getattr(mlp, 'gate_up_proj', None)
                if fused is not None and hasattr(fused, 'weight'):
                    w = fused.weight
                    seg_size = w.shape[0] // 2
                    chunk = seg_size // tp
                    gate_shard = w[:seg_size][rank * chunk:(rank + 1) * chunk]
                    up_shard = w[seg_size:][rank * chunk:(rank + 1) * chunk]
                    fused.weight = mx.concatenate([gate_shard, up_shard], axis=0)
                    n_sharded += 1
                # Row-parallel: down proj
                for proj_name in ['down_proj', 'w2', 'down']:
                    proj = getattr(mlp, proj_name, None)
                    if proj is not None and hasattr(proj, 'weight'):
                        _shard_row(proj, tp, rank)
                        n_sharded += 1
                        break
                break

        # Adjust head counts on the runner
        if hasattr(self, 'num_kv_heads') and self.num_kv_heads is not None:
            self.num_kv_heads = self.num_kv_heads // tp
        if hasattr(self, 'num_heads') and self.num_heads is not None:
            self.num_heads = self.num_heads // tp

        # Patch the model's internal config so attention reshape uses sharded head counts
        # MLX Llama stores args on model.args and n_heads/n_kv_heads on each attention module
        model_args = getattr(model, 'args', None)
        if model_args is not None:
            if hasattr(model_args, 'num_attention_heads'):
                model_args.num_attention_heads = model_args.num_attention_heads // tp
            if hasattr(model_args, 'num_key_value_heads'):
                model_args.num_key_value_heads = model_args.num_key_value_heads // tp
            logger.info("Patched model.args: num_attention_heads=%s, num_key_value_heads=%s",
                        getattr(model_args, 'num_attention_heads', '?'),
                        getattr(model_args, 'num_key_value_heads', '?'))

        # Patch each attention module's n_heads/n_kv_heads so Q/K/V reshape correctly
        for layer in layers:
            for module_name in ['self_attn', 'attention', 'attn']:
                attn = getattr(layer, module_name, None)
                if attn is None:
                    continue
                if hasattr(attn, 'n_heads'):
                    attn.n_heads = attn.n_heads // tp
                if hasattr(attn, 'n_kv_heads'):
                    attn.n_kv_heads = attn.n_kv_heads // tp
                if hasattr(attn, 'num_heads'):
                    attn.num_heads = attn.num_heads // tp
                if hasattr(attn, 'num_kv_heads'):
                    attn.num_kv_heads = attn.num_kv_heads // tp
                # gpt-oss uses num_attention_heads / num_key_value_heads
                if hasattr(attn, 'num_attention_heads'):
                    attn.num_attention_heads = attn.num_attention_heads // tp
                if hasattr(attn, 'num_key_value_heads'):
                    attn.num_key_value_heads = attn.num_key_value_heads // tp
                # Recalculate num_key_value_groups after head count changes
                if (hasattr(attn, 'num_key_value_groups') and
                        hasattr(attn, 'num_attention_heads') and
                        hasattr(attn, 'num_key_value_heads')):
                    attn.num_key_value_groups = attn.num_attention_heads // attn.num_key_value_heads
                # Resize model-specific tensors that depend on head count (e.g. gpt-oss sinks)
                if hasattr(attn, 'sinks') and attn.sinks is not None:
                    new_heads = getattr(attn, 'n_heads', getattr(attn, 'num_heads', getattr(attn, 'num_attention_heads', None)))
                    if new_heads is not None and attn.sinks.shape[0] != new_heads:
                        attn.sinks = mx.zeros((new_heads,), dtype=attn.sinks.dtype)
                break

        # Force MLX to free the sliced-away memory
        mx.eval(mx.array([0]))

        logger.info("Weight sharding complete: %d projections sharded, tp_rank=%d/%d", n_sharded, rank, tp)

    def _install_tp_allreduce_hooks(self) -> None:
        """Install forward hooks on row-parallel layers to all-reduce partial sums.

        In tensor parallelism, column-parallel layers (q/k/v_proj, gate/up_proj)
        shard the output dimension -- each rank computes a slice independently.
        Row-parallel layers (o_proj, down_proj) shard the input dimension and
        produce PARTIAL sums that must be all-reduced across ranks before the
        residual connection adds them back to the hidden state.

        Without this all-reduce, each rank sees only its own partial sum,
        leading to garbage output.

        ProtoAI-Bakari--tp_allreduce_inject: wrap __call__ on each o_proj and
        down_proj Linear to append UDP all-reduce after matmul. MLX has no
        forward-hook API like PyTorch, so we wrap __call__ directly.
        """
        logger.info("ProtoAI-Bakari--_install_tp_allreduce_hooks: ENTERED, tp_size=%d", self.tp_size)
        if self.tp_size <= 1:
            logger.info("ProtoAI-Bakari--_install_tp_allreduce_hooks: SKIPPED (tp_size=%d <= 1)", self.tp_size)
            return

        import sys
        # ProtoAI-Bakari--udp_allreduce_path: add udp_allreduce to path on remote nodes
        for udp_path in ["/Users/z", "/Users/z/udp_allreduce/..", "/home/z/AGENT"]:
            if udp_path not in sys.path:
                sys.path.insert(0, udp_path)
        try:
            from udp_allreduce import AllReduceGroup
            from udp_allreduce.config import DEFAULT_PEERS
        except Exception as e:
            logger.error("Failed to import udp_allreduce: %s", e)
            return
        # Initialize the UDP all-reduce group for this rank.
        # ProtoAI-Bakari--ray_peer_autodetect: Ray workers only propagate env
        # vars listed in vllm/ray/ray_env.py (envs.environment_variables).
        # VLLM_UDP_AR_PEERS is not in that list, so workers fall back to
        # DEFAULT_PEERS (sys4-7 hardcoded) even when the driver sets the var.
        # Fix: auto-detect live peers from Ray cluster topology so no env var
        # propagation is needed at all.  Env vars remain as a manual override.
        peer_ips = None

        # 1. Prefer explicit env-var override (set on driver AND worker).
        env_peers = os.environ.get("VLLM_UDP_AR_PEERS") or os.environ.get("UDP_AR_PEERS")
        if env_peers:
            peer_ips = [p.strip() for p in env_peers.split(",") if p.strip()]
            logger.info(
                "UDP all-reduce peers from env var: %s", peer_ips
            )

        # 2. Auto-detect from GLOO distributed group — gives actual rank-to-IP
        #    mapping. This is correct even when Ray assigns workers to
        #    arbitrary nodes (sorted Ray IPs were wrong: rank 1 on sys7 but
        #    peers[1] was sys5).
        if not peer_ips:
            try:
                import torch.distributed as _tdist
                if _tdist.is_initialized():
                    # Gather each rank's IP via GLOO all_gather
                    # Detect IP: VLLM_HOST_IP > GLOO interface > UDP connect
                    my_ip = os.environ.get("VLLM_HOST_IP", "")
                    if not my_ip or my_ip == "127.0.0.1":
                        import subprocess, sys as _sys
                        ifname = os.environ.get("GLOO_SOCKET_IFNAME", "en0")
                        if _sys.platform == "darwin":
                            try:
                                my_ip = subprocess.check_output(
                                    ["ipconfig", "getifaddr", ifname],
                                    text=True, timeout=2
                                ).strip()
                            except Exception:
                                pass
                        if not my_ip or my_ip == "127.0.0.1":
                            import socket as _sock
                            try:
                                s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
                                s.connect(("8.8.8.8", 80))
                                my_ip = s.getsockname()[0]
                                s.close()
                            except Exception:
                                my_ip = "127.0.0.1"
                    # Use all_gather_object for rank-ordered IP list
                    ip_list = [None] * _tdist.get_world_size()
                    _tdist.all_gather_object(ip_list, my_ip)
                    peer_ips = ip_list[:self.tp_size]
                    logger.info(
                        "UDP all-reduce peers from GLOO rank-to-IP mapping: %s",
                        peer_ips,
                    )
            except Exception as _e:
                logger.warning("GLOO peer auto-detect failed: %s", _e)

        # 2b. Fallback: Ray topology (may give wrong rank-to-IP mapping)
        if not peer_ips:
            try:
                import ray as _ray
                if _ray.is_initialized():
                    nodes = _ray.nodes()
                    peer_ips = sorted([
                        n["NodeManagerAddress"]
                        for n in nodes
                        if n.get("Alive", False)
                    ])
                    logger.info(
                        "UDP all-reduce peers from Ray topology (sorted, may be wrong): %s",
                        peer_ips,
                    )
                else:
                    logger.warning(
                        "Ray not initialized; cannot auto-detect peers"
                    )
            except Exception as _e:
                logger.warning("Ray peer auto-detect failed: %s", _e)

        # 3. Last resort: compiled-in DEFAULT_PEERS constant.
        if not peer_ips:
            peer_ips = list(DEFAULT_PEERS)
            logger.warning(
                "UDP all-reduce peers falling back to DEFAULT_PEERS: %s — "
                "set VLLM_UDP_AR_PEERS or ensure Ray is initialized to fix this",
                peer_ips,
            )

        # Robust rank detection: self.tp_rank may still be 0 if
        # _shard_weights_tp skipped (e.g. torch.distributed not init at that
        # point). Re-check now — workers have had more time to init.
        effective_rank = self.tp_rank
        try:
            import torch.distributed as tdist
            if tdist.is_initialized():
                effective_rank = tdist.get_rank()
                if effective_rank != self.tp_rank:
                    logger.warning(
                        "tp_rank mismatch: self.tp_rank=%d, tdist.get_rank()=%d — using %d",
                        self.tp_rank, effective_rank, effective_rank,
                    )
                    self.tp_rank = effective_rank
        except Exception:
            pass

        ar_group = AllReduceGroup(
            rank=effective_rank,
            world_size=self.tp_size,
            peers=peer_ips[:self.tp_size],
        )
        # Store on self so it persists and can be cleaned up
        self._allreduce_group = ar_group

        _ar_msg = (
            f"[UDP-AR] all-reduce group initialized: rank={effective_rank}, "
            f"world_size={self.tp_size}, peers={peer_ips[:self.tp_size]}"
        )
        print(_ar_msg, flush=True)
        logger.info(_ar_msg)

        # Find transformer layers
        model = self.model
        layers = None
        for attr_path in ['model.layers', 'layers', 'transformer.layers']:
            obj = model
            try:
                for part in attr_path.split('.'):
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue

        if layers is None:
            logger.error("Cannot find transformer layers for all-reduce hooks")
            return

        # ProtoAI-Bakari--allreduce_wrapper_class: Python's __call__ dunder is
        # resolved on the TYPE not the instance. Setting proj.__call__ = wrapper
        # does NOT intercept proj(x). We must replace the module attribute on the
        # parent with a wrapper class whose TYPE defines __call__.
        #
        # ProtoAI-Bakari--profiler: set VLLM_PROFILE_AR=1 to enable per-phase
        # timing.  The branch is resolved once at class-definition time so there
        # is zero per-call overhead when profiling is off.
        _profile_ar_enabled = os.environ.get("VLLM_PROFILE_AR", "0") == "1"

        if _profile_ar_enabled:
            # Use the canonical ARProfiler from allreduce_hooks when available;
            # fall back to a self-contained inline version otherwise.
            try:
                from allreduce_hooks import ARProfiler as _ARProfilerCls
            except ImportError:
                import collections as _col
                import math as _math2

                class _ARProfilerCls:  # type: ignore[no-redef]
                    """Inline fallback ARProfiler for model_runner standalone path."""
                    _singleton = None
                    PHASE_NAMES = ("A:mx_eval", "B:mx_to_np", "C:all_reduce", "D:np_to_mx")

                    def __init__(self):
                        self._total = [0.0, 0.0, 0.0, 0.0]
                        self._count = 0
                        self._ring = _col.deque(maxlen=200)

                    @classmethod
                    def instance(cls):
                        if cls._singleton is None:
                            cls._singleton = cls()
                        return cls._singleton

                    def record(self, phases):
                        for i, v in enumerate(phases):
                            self._total[i] += v
                        self._count += 1
                        self._ring.append(phases)

                    @classmethod
                    def dump_profile(cls):
                        p = cls.instance()
                        n = p._count
                        if n == 0:
                            print("[ARProfiler] No calls recorded.")
                            return
                        total_wall = sum(p._total)
                        print(f"\n{'='*60}")
                        print(f"[ARProfiler]  AllReduce phase breakdown  ({n} calls)")
                        print(f"{'='*60}")
                        for i, name in enumerate(cls.PHASE_NAMES):
                            avg_ms = (p._total[i] / n) * 1000.0
                            pct = (p._total[i] / total_wall * 100.0) if total_wall else 0.0
                            print(f"  {name:<18}  avg={avg_ms:.4f} ms  {pct:.1f}%")
                        avg_call_ms = total_wall / n * 1000.0
                        print(f"  Total/call  : {avg_call_ms:.4f} ms")
                        print(f"  Est/token(64): {avg_call_ms * 64:.2f} ms  "
                              f"({1000.0 / (avg_call_ms * 64):.1f} TPS AR ceiling)")
                        if p._ring:
                            totals = [sum(r) * 1000.0 for r in p._ring]
                            mean_v = sum(totals) / len(totals)
                            sd = _math2.sqrt(
                                sum((x - mean_v) ** 2 for x in totals) / len(totals)
                            )
                            print(f"  Ring({len(p._ring)}): "
                                  f"min={min(totals):.3f}  mean={mean_v:.3f}  "
                                  f"max={max(totals):.3f}  sd={sd:.3f} ms")
                        print(f"{'='*60}\n")

            _ar_profiler_instance = _ARProfilerCls.instance()
        else:
            _ARProfilerCls = None          # type: ignore[assignment]
            _ar_profiler_instance = None

        # ---- Check if MLX native distributed is available and properly
        #      initialized for multi-node.  If so, use mx.distributed.all_sum()
        #      instead of the UDP allreduce (zero numpy, zero GIL, zero copies).
        _use_native_mlx_allreduce = False
        _mlx_dist_group = None
        if HAS_MLX_DISTRIBUTED:
            try:
                # CRITICAL: All ranks must call dist.init() simultaneously.
                # Model loading takes variable time across ranks, so rank 0
                # might call dist.init() while rank 1 is still loading weights.
                # The ring backend's TCP handshake requires both ranks to be
                # listening. Use a GLOO barrier to synchronize before init.
                import torch.distributed as _tdist_sync
                if _tdist_sync.is_initialized():
                    logger.info("GLOO barrier before MLX distributed init...")
                    _tdist_sync.barrier()
                    logger.info("GLOO barrier passed, all ranks ready for MLX ring init")
                _mlx_dist_group = dist.init()
                if _mlx_dist_group.size() >= self.tp_size:
                    _use_native_mlx_allreduce = True
                    logger.info(
                        "Using MLX native distributed all_sum for allreduce "
                        "(group size=%d, tp_size=%d)",
                        _mlx_dist_group.size(), self.tp_size,
                    )
                else:
                    logger.info(
                        "MLX distributed init returned size=%d (need %d), "
                        "falling back to UDP. Check MLX_RANK=%s MLX_HOSTFILE=%s",
                        _mlx_dist_group.size(), self.tp_size,
                        os.environ.get("MLX_RANK", "NOT SET"),
                        os.environ.get("MLX_HOSTFILE", "NOT SET"),
                    )
            except Exception as _e:
                logger.warning("MLX distributed init failed: %s", _e)
        if not _use_native_mlx_allreduce:
            logger.info("Using UDP allreduce (MLX distributed not available for multi-node)")

        class AllReduceLinearWrapper:
            """Wrapper that forwards to original Linear then all-reduces.

            This replaces the projection module on the parent (e.g., attn.o_proj)
            so that Python's type-based __call__ dispatch invokes our wrapper.

            When MLX native distributed is available (ring backend over TCP/TB4),
            uses mx.distributed.all_sum() — zero numpy conversion, zero GIL.
            Falls back to UDP allreduce via numpy bridge otherwise.
            """

            # Shared profiler reference.  None when profiling is disabled.
            _profiler = _ar_profiler_instance
            # Whether to use native MLX allreduce (set once, used in __call__)
            _native = _use_native_mlx_allreduce
            _mlx_group = _mlx_dist_group

            def __init__(self, original_proj, layer_idx, proj_name, group):
                self._original = original_proj
                self._layer_idx = layer_idx
                self._proj_name = proj_name
                self._group = group  # UDP group (fallback)
                self._call_count = 0
                if not AllReduceLinearWrapper._native:
                    # Only import numpy bridge if using UDP path
                    from udp_allreduce.mlx_bridge import mx_to_numpy, numpy_to_mx
                    self._mx_to_numpy = mx_to_numpy
                    self._numpy_to_mx = numpy_to_mx
                # Forward attribute access to original (for weight, scales, etc.)
                for attr in dir(original_proj):
                    if not attr.startswith('_') and attr not in ('weight', 'scales', 'biases'):
                        try:
                            setattr(self, attr, getattr(original_proj, attr))
                        except (AttributeError, TypeError):
                            pass

            if _profile_ar_enabled:
                def __call__(self, *args, **kwargs):
                    # Step 1: compute partial sum via original linear forward
                    partial = self._original(*args, **kwargs)

                    # Phase A: mx.async_eval dispatches Metal cmd buffer (non-blocking)
                    # Actual sync happens in _mx_to_numpy when np.array() reads the buffer
                    _t0 = time.perf_counter()
                    mx.async_eval(partial)
                    _t1 = time.perf_counter()

                    original_shape = partial.shape
                    original_dtype_str = str(partial.dtype)

                    # Phase B: MLX -> numpy bridge
                    np_partial = self._mx_to_numpy(partial)
                    _t2 = time.perf_counter()

                    # Phase C: all-reduce network call (bf16-safe)
                    if np_partial.dtype == np.uint16:
                        flat_u16 = np_partial.ravel()
                        buf = np.frombuffer(
                            (flat_u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32
                        ).copy().reshape(np_partial.shape)
                        self._group.all_reduce_(buf, op="sum")
                        u32_view = np.frombuffer(buf.ravel().tobytes(), dtype=np.uint32)
                        np_partial = (u32_view >> 16).astype(np.uint16).reshape(np_partial.shape)
                    else:
                        self._group.all_reduce_(np_partial, op="sum")
                    _t3 = time.perf_counter()

                    # Phase D: numpy -> MLX bridge
                    reduced = self._numpy_to_mx(
                        np_partial, target_dtype_str=original_dtype_str, shape=original_shape
                    )
                    _t4 = time.perf_counter()

                    phases = (_t1 - _t0, _t2 - _t1, _t3 - _t2, _t4 - _t3)
                    AllReduceLinearWrapper._profiler.record(phases)

                    self._call_count += 1
                    if self._call_count <= 2:
                        logger.debug(
                            "allreduce[profiled]: layer=%d %s call=%d shape=%s "
                            "phases_ms=(%.3f,%.3f,%.3f,%.3f)",
                            self._layer_idx, self._proj_name,
                            self._call_count, original_shape,
                            phases[0]*1e3, phases[1]*1e3, phases[2]*1e3, phases[3]*1e3,
                        )

                    return reduced
            else:
                def __call__(self, *args, **kwargs):
                    # Step 1: compute partial sum via original linear forward
                    partial = self._original(*args, **kwargs)

                    # === NATIVE MLX DISTRIBUTED PATH ===
                    # Zero numpy, zero GIL, zero copies — direct Metal buffer allreduce
                    if AllReduceLinearWrapper._native:
                        reduced = mx.distributed.all_sum(
                            partial, group=AllReduceLinearWrapper._mlx_group
                        )
                        self._call_count += 1
                        return reduced

                    # === UDP ALLREDUCE FALLBACK PATH ===
                    # Step 2: dispatch Metal cmd buffer (non-blocking)
                    mx.async_eval(partial)

                    original_shape = partial.shape
                    original_dtype_str = str(partial.dtype)

                    # Step 3: MLX -> numpy (near-zero-copy on unified memory)
                    np_partial = self._mx_to_numpy(partial)

                    # Bug #1 fix: bf16 is viewed as uint16 in numpy -- summing
                    # uint16 bit patterns produces garbage. Cast to float32 for
                    # the actual arithmetic, then cast back after all-reduce.
                    if np_partial.dtype == np.uint16:
                        flat_u16 = np_partial.ravel()
                        buf = np.frombuffer(
                            (flat_u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32
                        ).copy().reshape(np_partial.shape)
                        self._group.all_reduce_(buf, op="sum")
                        u32_view = np.frombuffer(buf.ravel().tobytes(), dtype=np.uint32)
                        np_partial = (u32_view >> 16).astype(np.uint16).reshape(np_partial.shape)
                    else:
                        self._group.all_reduce_(np_partial, op="sum")

                    # Step 5: numpy -> MLX
                    reduced = self._numpy_to_mx(np_partial, target_dtype_str=original_dtype_str,
                                          shape=original_shape)

                    self._call_count += 1
                    if self._call_count <= 2:
                        logger.debug(
                            "allreduce: layer=%d %s call=%d shape=%s",
                            self._layer_idx, self._proj_name,
                            self._call_count, original_shape,
                        )

                    return reduced

            @classmethod
            def dump_profile(cls):
                """Print per-phase profiling report.  No-op when VLLM_PROFILE_AR!=1."""
                if cls._profiler is not None:
                    cls._profiler.dump_profile()
                else:
                    print("[AllReduceLinearWrapper] Profiling disabled. "
                          "Set VLLM_PROFILE_AR=1 to enable.")

            def __getattr__(self, name):
                # Delegate attribute access to the original module
                return getattr(self._original, name)

        n_hooked = 0

        for layer_idx, layer in enumerate(layers):
            # Hook attention output projection (o_proj / out_proj)
            for module_name in ['self_attn', 'attention', 'attn']:
                attn = getattr(layer, module_name, None)
                if attn is None:
                    continue
                for proj_name in ['o_proj', 'out_proj']:
                    proj = getattr(attn, proj_name, None)
                    if proj is not None:
                        wrapper = AllReduceLinearWrapper(proj, layer_idx, proj_name, ar_group)
                        setattr(attn, proj_name, wrapper)
                        n_hooked += 1
                        break
                break

            # Hook MLP down projection (down_proj / w2 / down)
            for mlp_name in ['mlp', 'feed_forward', 'ffn']:
                mlp = getattr(layer, mlp_name, None)
                if mlp is None:
                    continue
                for proj_name in ['down_proj', 'w2', 'down']:
                    proj = getattr(mlp, proj_name, None)
                    if proj is not None:
                        wrapper = AllReduceLinearWrapper(proj, layer_idx, proj_name, ar_group)
                        setattr(mlp, proj_name, wrapper)
                        n_hooked += 1
                        break
                break

        # For dense models (Llama), expect 2 hooks/layer (o_proj + down_proj = 64 for 32 layers).
        # For MoE models (gpt-oss), MLP uses SwitchGLU with no down_proj — only o_proj hooked
        # (1 hook/layer = 36 for 36 layers). Both patterns are correct.
        n_layers = len(layers)
        if n_hooked == n_layers * 2:
            logger.info(
                "All-reduce hooks installed: %d hooks across %d layers "
                "(o_proj + down_proj per layer), rank=%d/%d",
                n_hooked, n_layers, self.tp_rank, self.tp_size,
            )
        elif n_hooked == n_layers:
            logger.info(
                "All-reduce hooks installed: %d hooks across %d layers "
                "(o_proj only — MoE MLP has no row-parallel down_proj), rank=%d/%d",
                n_hooked, n_layers, self.tp_rank, self.tp_size,
            )
        elif n_hooked > 0:
            logger.warning(
                "All-reduce hooks: installed %d across %d layers "
                "(expected %d or %d — check layer naming)",
                n_hooked, n_layers, n_layers, n_layers * 2,
            )
        else:
            logger.error(
                "All-reduce hooks: 0 installed across %d layers — "
                "no row-parallel projections found", n_layers,
            )

    def _extract_logits(self, model_output: Any) -> mx.array:
        """Extract logits from model output.

        Handles both mlx-lm (returns array directly) and mlx-vlm
        (returns LanguageModelOutput with .logits attribute).

        Args:
            model_output: Output from model forward pass

        Returns:
            Logits array
        """
        if hasattr(model_output, "logits"):
            # mlx-vlm returns LanguageModelOutput
            return model_output.logits
        # mlx-lm returns logits directly
        return model_output

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get KV cache specification.

        Returns:
            Dictionary mapping attention layer names to KV cache specs
        """
        block_size = self.metal_config.block_size

        # Create a spec for each layer
        specs: dict[str, KVCacheSpec] = {}
        for layer_idx in range(self.num_layers):
            layer_name = f"layers.{layer_idx}.self_attn"
            specs[layer_name] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                dtype=torch.float16,
            )

        return specs

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Accept KV cache config from engine (no-op for MLX path).

        MLX manages its own KV cache via make_prompt_cache().
        This method exists to satisfy the engine's initialization protocol.
        """
        logger.info(
            "KV cache config received: %d blocks (MLX manages cache internally)",
            kv_cache_config.num_blocks,
        )

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of a single cache block in bytes.

        Returns:
            Block size in bytes
        """
        block_size = self.metal_config.block_size

        # Each block stores key and value for all layers
        # Block memory = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
        dtype_size = 2  # float16
        return (
            2
            * self.num_layers
            * block_size
            * self.num_kv_heads
            * self.head_dim
            * dtype_size
        )

    def warm_up(self) -> None:
        """Warm up the model with a dummy forward pass.

        When paged attention is enabled, also loads the HF Metal kernel and
        runs a tiny ``reshape_and_cache`` to force Metal library creation.
        This catches Metal language-version incompatibilities at startup
        rather than during the first real inference request.
        """
        if self.model is None:
            logger.warning("Model not loaded, skipping warm-up")
            return

        logger.info("Warming up model...")

        # Run a small dummy inference (standard MLX path)
        try:
            dummy_tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
            output = self.model(dummy_tokens)
            logits = self._extract_logits(output)
            mx.eval(logits)
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

        # Paged attention kernel warm-up: load kernel + smoke-test Metal ops
        if hasattr(self, "_paged_kv_cache") and self._paged_kv_cache is not None:
            self._warm_up_paged_attention_kernel()

    def _warm_up_paged_attention_kernel(self) -> None:
        """Load the HF paged-attention kernel and verify Metal ops work.

        Forces ``newLibraryWithData`` inside the .so by running a single-token
        ``reshape_and_cache`` against layer 0 of the already-allocated cache.
        If the embedded metallib targets a Metal language version unsupported
        by this OS, the error surfaces here instead of mid-inference.
        """
        import platform

        from vllm_metal.metal_kernel_backend.kernel_loader import (
            get_paged_attention_ops,
        )

        cache = self._paged_kv_cache

        logger.info("Warming up paged attention Metal kernel...")

        try:
            ops = get_paged_attention_ops()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load paged-attention Metal kernel: {e}. "
                f"macOS version: {platform.mac_ver()[0]}"
            ) from e

        # Smoke-test: single-token reshape_and_cache on layer 0
        try:
            dummy_k = torch.zeros(
                1,
                cache.num_kv_heads,
                cache.head_dim,
                dtype=cache.dtype,
                device="mps",
            )
            dummy_v = torch.zeros_like(dummy_k)
            dummy_slot = torch.zeros(1, dtype=torch.long, device="mps")

            ops.reshape_and_cache(
                dummy_k,
                dummy_v,
                cache.key_caches[0],
                cache.value_caches[0],
                dummy_slot,
                "auto",
                cache.k_scale_tensor,
                cache.v_scale_tensor,
            )
            logger.info("Paged attention Metal kernel warm-up complete")
        except RuntimeError as e:
            mac_ver = platform.mac_ver()[0]
            if "language version" in str(e):
                raise RuntimeError(
                    f"Metal kernel incompatible with this OS (macOS {mac_ver}). "
                    f"The kernel requires a newer Metal language version than "
                    f"this OS supports. Original error: {e}"
                ) from e
            raise

    def _make_sampling_metadata(
        self,
        sampling_params_list: list[SamplingParams],
        prompt_token_id_lists: list[list[int]],
        output_token_id_lists: list[list[int]],
        generators: dict[int, torch.Generator] | None = None,
    ) -> SamplingMetadata:
        """Create SamplingMetadata from per-request SamplingParams.

        Args:
            sampling_params_list: List of SamplingParams, one per request
            prompt_token_id_lists: Prompt token IDs per request (prefix used for
                repetition penalty).
            output_token_id_lists: Generated token IDs per request (used for
                presence/frequency penalties, and also repetition penalty).
            generators: Optional per-request torch generators keyed by batch index.
                If omitted, sampler falls back to the global RNG for those entries.

        Returns:
            SamplingMetadata for the batch
        """
        batch_size = len(sampling_params_list)
        if len(prompt_token_id_lists) != batch_size:
            raise ValueError(
                "Expected prompt token ids for each request in the batch "
                f"(len(prompt_token_id_lists)={len(prompt_token_id_lists)} "
                f"!= batch_size={batch_size})."
            )
        if len(output_token_id_lists) != batch_size:
            raise ValueError(
                "Expected output token ids for each request in the batch "
                f"(len(output_token_id_lists)={len(output_token_id_lists)} "
                f"!= batch_size={batch_size})."
            )

        # Determine sampling mode
        all_greedy = all(sp.temperature < 1e-5 for sp in sampling_params_list)
        all_random = not all_greedy and all(
            sp.temperature >= 1e-5 for sp in sampling_params_list
        )

        # Check if any penalties are applied
        no_penalties = all(
            sp.frequency_penalty == 0
            and sp.presence_penalty == 0
            and sp.repetition_penalty == 1.0
            for sp in sampling_params_list
        )

        generators = generators or {}

        # top_k: pass None if all values indicate no filtering
        # -1 = vLLM default (no filtering), 0 = OpenAI API convention (no filtering)
        # vLLM's sampler expects None to skip top-k entirely
        top_k_values = [sp.top_k for sp in sampling_params_list]
        top_k = (
            None
            if all(k <= 0 for k in top_k_values)
            else torch.tensor(top_k_values, dtype=torch.int32, device=self.device)
        )

        # top_p: pass None if all values are 1.0 (no filtering)
        # vLLM's sampler expects None to skip top-p entirely
        top_p_values = [sp.top_p for sp in sampling_params_list]
        top_p = (
            None
            if all(p == 1.0 for p in top_p_values)
            else torch.tensor(top_p_values, dtype=torch.float32, device=self.device)
        )

        vocab_size = self.model_args.get("vocab_size", 32000)
        prompt_token_ids_tensor = None
        if not no_penalties:
            prompt_token_ids_tensor = make_tensor_with_pad(
                prompt_token_id_lists,
                pad=vocab_size,
                device=self.device,
                dtype=torch.int64,
                pin_memory=False,
            )

        return SamplingMetadata(
            temperature=None
            if all_greedy
            else torch.tensor(
                [sp.temperature for sp in sampling_params_list],
                dtype=torch.float32,
                device=self.device,
            ),
            all_greedy=all_greedy,
            all_random=all_random,
            top_p=top_p,
            top_k=top_k,
            generators=generators,
            max_num_logprobs=None,
            prompt_token_ids=prompt_token_ids_tensor,
            output_token_ids=output_token_id_lists,
            frequency_penalties=torch.tensor(
                [sp.frequency_penalty for sp in sampling_params_list],
                dtype=torch.float32,
                device=self.device,
            ),
            presence_penalties=torch.tensor(
                [sp.presence_penalty for sp in sampling_params_list],
                dtype=torch.float32,
                device=self.device,
            ),
            repetition_penalties=torch.tensor(
                [sp.repetition_penalty for sp in sampling_params_list],
                dtype=torch.float32,
                device=self.device,
            ),
            no_penalties=no_penalties,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )

    def _prefill_single(
        self,
        req_id: str,
        token_ids: list[int],
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> tuple[int, list[KVCache]]:
        """Process a single prefill request.

        Args:
            req_id: Request ID
            token_ids: Prompt token IDs
            sampling_params: Sampling parameters for this request

        Returns:
            Tuple of (next_token, cache)
        """
        cache: list[KVCache]
        cached_prefix_len = 0

        # Prefix caching: cache KV for tokens[:-1], always process last token
        prefix = token_ids[:-1] if len(token_ids) > 1 else []
        cache_model = (
            self.model.language_model
            if self._is_vlm and hasattr(self.model, "language_model")
            else self.model
        )

        # Create cache to check if model supports prefix caching
        cache = make_prompt_cache(cache_model)
        # Prefix caching only safe for pure KVCache models (not Mamba/hybrid)
        supports_prefix_cache = all(isinstance(c, KVCache) for c in cache)

        # Try to reuse cached prefix
        if supports_prefix_cache and self._prefix_cache is not None and len(prefix) > 0:
            cached = self._prefix_cache.lookup(prefix)
            if cached is not None:
                # Cache hit: restore KV for prefix, process only last token
                cache = self._prefix_cache.restore_cache(
                    cached, self.model, self._is_vlm
                )
                cached_prefix_len = len(cached.token_ids)
            else:
                # Cache miss: process prefix first, cache it, then last token
                prefix_ids = mx.array([prefix], dtype=mx.int32)
                _ = self.model(prefix_ids, cache=cache)
                self._prefix_cache.insert(prefix, cache)
                cached_prefix_len = len(prefix)

        # Prefill: process remaining tokens (always at least the last token)
        tokens_to_process = token_ids[cached_prefix_len:]
        input_ids = mx.array([tokens_to_process], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)

        logits = self._extract_logits(model_output)

        # Extract last token logits
        last_logits = logits[:, -1, :]

        # Use native MLX greedy sampling when possible (avoids PyTorch round-trip)
        is_greedy = sampling_params.temperature < 1e-5
        needs_advanced_sampling = (
            sampling_params.top_k > 0
            or sampling_params.top_p < 1.0
            or sampling_params.frequency_penalty != 0
            or sampling_params.presence_penalty != 0
            or sampling_params.repetition_penalty != 1.0
        )

        if is_greedy and not needs_advanced_sampling:
            # Fast path: native MLX greedy sampling
            next_token_mlx = _mlx_greedy_sample(last_logits)
            # Single eval for logits, token, and cache state together
            mx.eval(next_token_mlx, *[c.state for c in cache])
            next_token = int(next_token_mlx.item())
        else:
            # Slow path: use vLLM sampler for advanced sampling
            # Single eval for logits and cache state together
            mx.eval(last_logits, *[c.state for c in cache])
            # Convert to torch for sampling
            logits_torch = mlx_to_torch(
                last_logits.astype(mx.float32), device=self.device
            )
            generators = {} if generator is None else {0: generator}
            metadata = self._make_sampling_metadata(
                [sampling_params],
                [token_ids],
                [[]],
                generators=generators,
            )
            output = self._sampler.forward(logits_torch, metadata)
            next_token = int(output.sampled_token_ids[0, 0].item())

        return next_token, cache

    def _batched_decode(self, decode_reqs: list[tuple[str, RequestState]]) -> list[int]:
        """Process multiple decode requests in a single batched forward pass.

        Uses BatchKVCache to merge individual caches, run ONE forward pass,
        then extract updated caches back.

        Args:
            decode_reqs: List of (req_id, state) tuples

        Returns:
            List of next tokens for each request
        """
        batch_size = len(decode_reqs)

        # Use Rust extension for efficient batch token retrieval if available
        if self._rust_state_manager is not None:
            last_tokens = self._rust_state_manager.get_last_tokens_batch(
                [req_id for req_id, _ in decode_reqs]
            )
        else:
            last_tokens = [
                state.token_ids[-1] if state.token_ids else 0
                for _, state in decode_reqs
            ]

        # Collect individual caches for merging
        caches_list = [state.cache for _, state in decode_reqs]

        # Merge individual KV caches into batched cache (one per layer)
        batch_cache = _merge_kv_caches(caches_list)

        # Create batched input: shape (batch_size, 1) for single-token decode
        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

        # === SINGLE FORWARD PASS FOR ALL REQUESTS ===
        model_output = self.model(batched_input, cache=batch_cache)
        logits = self._extract_logits(model_output)

        # Extract next token logits
        next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        sampling_params_list = [state.sampling_params for _, state in decode_reqs]

        # Check if all requests can use fast greedy sampling
        all_greedy = all(sp.temperature < 1e-5 for sp in sampling_params_list)
        any_advanced = any(
            sp.top_k > 0
            or sp.top_p < 1.0
            or sp.frequency_penalty != 0
            or sp.presence_penalty != 0
            or sp.repetition_penalty != 1.0
            for sp in sampling_params_list
        )

        if all_greedy and not any_advanced:
            # Fast path: native MLX greedy sampling for entire batch
            next_tokens_mlx = _mlx_greedy_sample(next_token_logits)
            # Single eval - no intermediate sync needed
            mx.eval(next_tokens_mlx)
            next_tokens: list[int] = next_tokens_mlx.tolist()
        else:
            # Slow path: use vLLM sampler for advanced sampling
            mx.eval(next_token_logits)
            prompt_token_ids_list = [
                state.token_ids[: state.prompt_len] for _, state in decode_reqs
            ]
            output_tokens_list = [
                state.token_ids[state.prompt_len :] for _, state in decode_reqs
            ]
            generators = {
                i: state.generator
                for i, (_, state) in enumerate(decode_reqs)
                if state.generator is not None
            }
            logits_torch = mlx_to_torch(
                next_token_logits.astype(mx.float32), device=self.device
            )
            metadata = self._make_sampling_metadata(
                sampling_params_list,
                prompt_token_ids_list,
                output_tokens_list,
                generators=generators,
            )
            output = self._sampler.forward(logits_torch, metadata)
            next_tokens = [
                int(output.sampled_token_ids[i, 0].item()) for i in range(batch_size)
            ]

        # Extract updated caches back to individual requests
        for i, (req_id, state) in enumerate(decode_reqs):
            state.cache = _extract_kv_cache(batch_cache, i)
            state.token_ids.append(next_tokens[i])
            state.generated_tokens += 1

            # Update Rust state manager if available
            if self._rust_state_manager is not None:
                self._rust_state_manager.append_token(req_id, next_tokens[i])

        return next_tokens

    def _sequential_decode(
        self, decode_reqs: list[tuple[str, RequestState]]
    ) -> list[int]:
        """Fallback: process decode requests sequentially.

        Used when batch size is 1 (no benefit from batching).

        Args:
            decode_reqs: List of (req_id, state) tuples

        Returns:
            List of next tokens for each request
        """
        next_tokens = []

        for req_id, state in decode_reqs:
            last_token = state.token_ids[-1] if state.token_ids else 0
            input_ids = mx.array([[last_token]], dtype=mx.int32)

            model_output = self.model(input_ids, cache=state.cache)
            logits = self._extract_logits(model_output)
            last_logits = logits[:, -1, :]

            # Use native MLX greedy sampling when possible
            sp = state.sampling_params
            is_greedy = sp.temperature < 1e-5
            needs_advanced = (
                sp.top_k > 0
                or sp.top_p < 1.0
                or sp.frequency_penalty != 0
                or sp.presence_penalty != 0
                or sp.repetition_penalty != 1.0
            )

            if is_greedy and not needs_advanced:
                # Fast path: native MLX greedy sampling
                next_token_mlx = _mlx_greedy_sample(last_logits)
                mx.eval(next_token_mlx)
                next_token = int(next_token_mlx.item())
            else:
                # Slow path: use vLLM sampler
                mx.eval(last_logits)
                logits_torch = mlx_to_torch(
                    last_logits.astype(mx.float32), device=self.device
                )
                generators = {} if state.generator is None else {0: state.generator}
                metadata = self._make_sampling_metadata(
                    [state.sampling_params],
                    [state.token_ids[: state.prompt_len]],
                    [state.token_ids[state.prompt_len :]],
                    generators=generators,
                )
                output = self._sampler.forward(logits_torch, metadata)
                next_token = int(output.sampled_token_ids[0, 0].item())

            next_tokens.append(next_token)

            # Update state
            state.token_ids.append(next_token)
            state.generated_tokens += 1

            # Update Rust state manager if available
            if self._rust_state_manager is not None:
                self._rust_state_manager.append_token(req_id, next_token)

        return next_tokens

    # ------------------------------------------------------------------
    # Paged attention paths
    # ------------------------------------------------------------------

    def _allocate_blocks_for_seq(self, req_id: str, num_tokens: int) -> list[int]:
        """Allocate blocks for a sequence, using the Rust allocator as source of truth."""
        bs = self._paged_block_size
        num_blocks = (num_tokens + bs - 1) // bs
        cache = self._paged_kv_cache
        existing = cache.get_sequence_blocks(req_id)
        if len(existing) >= num_blocks:
            return existing
        needed = num_blocks - len(existing)
        new_blocks = cache.allocate_blocks(req_id, needed)
        return existing + new_blocks

    def _prefill_single_request_paged(
        self,
        req_id: str,
        token_ids: list[int],
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> int:
        """Paged-attention prefill for a single request.

        Uses MLX for inline SDPA, then writes K/V to MPS paged cache via
        the HF reshape_and_cache kernel. Returns the next token.
        """
        num_tokens = len(token_ids)
        block_ids = self._allocate_blocks_for_seq(req_id, num_tokens)

        # Stash per-request metadata (slot_mapping) in thread-local so the
        # patched attention wrappers can read it during the forward pass.
        prepare_prefill(block_ids, num_tokens, self._paged_block_size)

        # OffsetCache is a fake cache — it stores no KV data.  It only
        # satisfies mlx_lm's RoPE offset and mask protocol.  Real KV is
        # written to the MPS paged cache by the attention wrapper.
        offset_caches = [OffsetCache(0) for _ in range(self.num_layers)]

        # The model forward calls each layer's self_attn, which has been
        # replaced by MetalKernelPagedAttentionWrapper.  The wrapper:
        # - ignores cache= (OffsetCache) for KV storage
        # - reads get_context() for slot_mapping
        # - computes attention via MLX SDPA
        # - writes K/V to MPS paged cache via reshape_and_cache
        input_ids = mx.array([token_ids], dtype=mx.int32)
        try:
            model_output = self.model(input_ids, cache=offset_caches)
            logits = self._extract_logits(model_output)
            last_logits = logits[:, -1, :]
        finally:
            clear_context()

        # Sample
        is_greedy = sampling_params.temperature < 1e-5
        needs_advanced = (
            sampling_params.top_k > 0
            or sampling_params.top_p < 1.0
            or sampling_params.frequency_penalty != 0
            or sampling_params.presence_penalty != 0
            or sampling_params.repetition_penalty != 1.0
        )

        if is_greedy and not needs_advanced:
            next_token_mlx = _mlx_greedy_sample(last_logits)
            mx.eval(next_token_mlx)
            next_token = int(next_token_mlx.item())
        else:
            mx.eval(last_logits)
            logits_torch = mlx_to_torch(
                last_logits.astype(mx.float32), device=self.device
            )
            generators = {} if generator is None else {0: generator}
            metadata = self._make_sampling_metadata(
                [sampling_params], [token_ids], [[]], generators=generators
            )
            output = self._sampler.forward(logits_torch, metadata)
            next_token = int(output.sampled_token_ids[0, 0].item())

        # Track sequence length
        self._paged_request_seq_lens[req_id] = num_tokens

        return next_token

    def _batched_decode_paged(
        self, decode_reqs: list[tuple[str, RequestState]]
    ) -> list[int]:
        """Paged-attention batched decode.

        Uses MLX for projections + per-request RoPE, then the HF kernel for
        reshape_and_cache + paged_attention_v1 (zero-copy from block tables).
        """

        """
        UNTESTED optimization to avoid unnecessary python rust FFI roundtrips.
        # Store block_ids when they're first computed (prefill) or when they grow (decode)
        # In _batched_decode_paged:
        new_seq_len = seq_len + 1
        prev_blocks = (seq_len + bs - 1) // bs
        curr_blocks = (new_seq_len + bs - 1) // bs

        if curr_blocks > prev_blocks:
            # Crossing a block boundary — need one more block
            new_block = cache.allocate_blocks(req_id, 1)
            self._paged_request_block_ids[req_id].extend(new_block)

        block_ids = self._paged_request_block_ids[req_id]
        """
        batch_size = len(decode_reqs)

        # Build request info for prepare_decode
        requests_info: list[tuple[list[int], int]] = []
        for req_id, state in decode_reqs:
            seq_len = self._paged_request_seq_lens.get(req_id, len(state.token_ids) - 1)
            block_ids = self._allocate_blocks_for_seq(req_id, seq_len + 1)
            requests_info.append((block_ids, seq_len))

        # Stash per-request metadata (slot_mapping, block_tables, context_lens,
        # offsets) in thread-local for the attention wrappers.
        prepare_decode(requests_info, self._paged_block_size)

        # OffsetCache is a fake cache — no KV stored.  The offset value
        # only matters for make_mask(); for single-token decode make_mask(1)
        # returns None regardless, so a shared max_offset is fine.  Actual
        # per-request RoPE offsets come from ctx.offsets in the wrapper.
        max_offset = max(info[1] for info in requests_info)
        offset_caches = [OffsetCache(max_offset) for _ in range(self.num_layers)]

        # Build batched input
        if self._rust_state_manager is not None:
            last_tokens = self._rust_state_manager.get_last_tokens_batch(
                [req_id for req_id, _ in decode_reqs]
            )
        else:
            last_tokens = [
                state.token_ids[-1] if state.token_ids else 0
                for _, state in decode_reqs
            ]

        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

        # The model forward calls each layer's self_attn, which has been
        # replaced by MetalKernelPagedAttentionWrapper.  The wrapper:
        # - ignores cache= (OffsetCache) for KV storage
        # - reads get_context() for block_tables, slot_mapping, offsets
        # - applies per-request RoPE using ctx.offsets
        # - writes new K/V to MPS paged cache via reshape_and_cache
        # - reads all cached K/V via paged_attention_v1 (zero-copy)
        try:
            model_output = self.model(batched_input, cache=offset_caches)
            logits = self._extract_logits(model_output)
            next_token_logits = logits[:, -1, :]
        finally:
            clear_context()

        # Sample
        sampling_params_list = [state.sampling_params for _, state in decode_reqs]
        all_greedy = all(sp.temperature < 1e-5 for sp in sampling_params_list)
        any_advanced = any(
            sp.top_k > 0
            or sp.top_p < 1.0
            or sp.frequency_penalty != 0
            or sp.presence_penalty != 0
            or sp.repetition_penalty != 1.0
            for sp in sampling_params_list
        )

        if all_greedy and not any_advanced:
            next_tokens_mlx = _mlx_greedy_sample(next_token_logits)
            mx.eval(next_tokens_mlx)
            next_tokens: list[int] = next_tokens_mlx.tolist()
        else:
            mx.eval(next_token_logits)
            prompt_token_ids_list = [
                state.token_ids[: state.prompt_len] for _, state in decode_reqs
            ]
            output_tokens_list = [
                state.token_ids[state.prompt_len :] for _, state in decode_reqs
            ]
            generators = {
                i: state.generator
                for i, (_, state) in enumerate(decode_reqs)
                if state.generator is not None
            }
            logits_torch = mlx_to_torch(
                next_token_logits.astype(mx.float32), device=self.device
            )
            metadata = self._make_sampling_metadata(
                sampling_params_list,
                prompt_token_ids_list,
                output_tokens_list,
                generators=generators,
            )
            output = self._sampler.forward(logits_torch, metadata)
            next_tokens = [
                int(output.sampled_token_ids[i, 0].item()) for i in range(batch_size)
            ]

        # Update state
        for i, (req_id, state) in enumerate(decode_reqs):
            state.token_ids.append(next_tokens[i])
            state.generated_tokens += 1
            self._paged_request_seq_lens[req_id] = (
                self._paged_request_seq_lens.get(req_id, len(state.token_ids) - 2) + 1
            )
            if self._rust_state_manager is not None:
                self._rust_state_manager.append_token(req_id, next_tokens[i])

        return next_tokens

    def execute_model(
        self, scheduler_output: SchedulerOutput, grammar_output: Any = None
    ) -> ModelRunnerOutput | None:
        """Execute model inference with true batched decode.

        Key optimization: Uses BatchKVCache.merge() to combine individual
        KV caches and run a SINGLE forward pass for all decode requests.

        Args:
            scheduler_output: Scheduler output with batch information

        Returns:
            Model runner output with generated tokens
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Collect all requests to process
        req_ids: list[str] = []
        req_id_to_index: dict[str, int] = {}
        sampled_tokens: list[list[int]] = []

        # === PHASE 1: Process new requests (prefill phase) ===
        new_reqs = scheduler_output.scheduled_new_reqs

        for new_req in new_reqs:
            req_id = new_req.req_id
            token_ids = new_req.prompt_token_ids or []
            sampling_params = new_req.sampling_params or SamplingParams()

            req_ids.append(req_id)
            req_id_to_index[req_id] = len(req_ids) - 1

            if token_ids:
                generator = _create_request_generator(self.device, sampling_params)

                if self._paged_kv_cache is not None:
                    # Paged attention path (Metal kernel)
                    scheduled_tokens = scheduler_output.num_scheduled_tokens.get(
                        req_id, 0
                    )
                    computed_tokens = new_req.num_computed_tokens
                    prompt_len = len(token_ids)
                    if computed_tokens + scheduled_tokens < prompt_len:
                        # Intermediate chunk: sample then drop (async scheduler
                        # allocates no placeholder for intermediate chunks).
                        cur_len = computed_tokens + scheduled_tokens
                        _discarded = self._prefill_single_request_paged(
                            req_id,
                            token_ids[:cur_len],
                            sampling_params,
                            generator=generator,
                        )
                        cache: list = []
                        sampled_tokens.append([])
                        self._request_states[req_id] = RequestState(
                            token_ids=list(token_ids),
                            prompt_len=prompt_len,
                            cache=cache,
                            sampling_params=sampling_params,
                            generator=generator,
                            generated_tokens=0,
                        )
                        if self._rust_state_manager is not None:
                            self._rust_state_manager.add_request(
                                req_id, list(token_ids[:cur_len])
                            )
                        continue
                    # Prompt complete: generate first output token.
                    next_token = self._prefill_single_request_paged(
                        req_id,
                        token_ids,
                        sampling_params,
                        generator=generator,
                    )
                    cache = []  # No per-request KV cache needed
                else:
                    next_token, cache = self._prefill_single(
                        req_id,
                        token_ids,
                        sampling_params,
                        generator=generator,
                    )
                sampled_tokens.append([next_token])

                # Store request state with cache for future decoding
                self._request_states[req_id] = RequestState(
                    token_ids=list(token_ids) + [next_token],
                    prompt_len=len(token_ids),
                    cache=cache,
                    sampling_params=sampling_params,
                    generator=generator,
                    generated_tokens=1,
                )

                # Register with Rust state manager if available
                if self._rust_state_manager is not None:
                    self._rust_state_manager.add_request(
                        req_id, list(token_ids) + [next_token]
                    )
            else:
                sampled_tokens.append([0])  # Fallback

        # === PHASE 2: Process cached requests (TRUE batched decode) ===
        cached_reqs = scheduler_output.scheduled_cached_reqs
        decode_req_ids = list(cached_reqs.req_ids)

        if decode_req_ids:
            if self._paged_kv_cache is not None:
                # Paged attention path: unified flow using model-runner-local
                # state (state.generated_tokens) instead of is_context_phase().
                req_id_to_cached_idx = {
                    rid: i for i, rid in enumerate(cached_reqs.req_ids)
                }
                paged_decode_reqs: list[tuple[str, RequestState]] = []

                for req_id in decode_req_ids:
                    state = self._request_states.get(req_id)
                    if state is None:
                        # Edge case: no state — emit dummy token
                        req_ids.append(req_id)
                        req_id_to_index[req_id] = len(req_ids) - 1
                        sampled_tokens.append([0])
                        continue

                    if state.generated_tokens == 0:
                        # Still prefilling prompt
                        idx = req_id_to_cached_idx.get(req_id)
                        if idx is not None and idx < len(
                            cached_reqs.num_computed_tokens
                        ):
                            computed = cached_reqs.num_computed_tokens[idx]
                        else:
                            computed = self._paged_request_seq_lens.get(req_id, 0)
                        scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                        target_len = computed + scheduled  # FIX: was just `computed`

                        if target_len < state.prompt_len:
                            # Intermediate chunk: sample then drop
                            prev_seq_len = self._paged_request_seq_lens.get(req_id, 0)
                            _discarded = self._prefill_single_request_paged(
                                req_id,
                                state.token_ids[:target_len],
                                state.sampling_params,
                                generator=state.generator,
                            )
                            if self._rust_state_manager is not None:
                                for tid in state.token_ids[prev_seq_len:target_len]:
                                    self._rust_state_manager.append_token(req_id, tid)
                            req_ids.append(req_id)
                            req_id_to_index[req_id] = len(req_ids) - 1
                            sampled_tokens.append([])
                        else:
                            # Last chunk: sample and keep (drains async placeholder)
                            prev_seq_len = self._paged_request_seq_lens.get(req_id, 0)
                            next_token = self._prefill_single_request_paged(
                                req_id,
                                state.token_ids[: state.prompt_len],
                                state.sampling_params,
                                generator=state.generator,
                            )
                            state.token_ids = list(
                                state.token_ids[: state.prompt_len]
                            ) + [next_token]
                            state.generated_tokens = 1
                            if self._rust_state_manager is not None:
                                for tid in state.token_ids[
                                    prev_seq_len : state.prompt_len
                                ]:
                                    self._rust_state_manager.append_token(req_id, tid)
                                self._rust_state_manager.append_token(
                                    req_id, next_token
                                )
                            req_ids.append(req_id)
                            req_id_to_index[req_id] = len(req_ids) - 1
                            sampled_tokens.append([next_token])
                    else:
                        # Decode phase: collect for batched decode
                        paged_decode_reqs.append((req_id, state))

                # Batch decode all generation-phase requests
                if paged_decode_reqs:
                    decode_tokens = self._batched_decode_paged(paged_decode_reqs)
                    for i, (req_id, _) in enumerate(paged_decode_reqs):
                        req_ids.append(req_id)
                        req_id_to_index[req_id] = len(req_ids) - 1
                        sampled_tokens.append([decode_tokens[i]])
            else:
                # Collect all valid decode requests
                valid_decode_reqs = []
                for req_id in decode_req_ids:
                    state = self._request_states.get(req_id)
                    if state is not None:
                        valid_decode_reqs.append((req_id, state))

                if valid_decode_reqs:
                    if len(valid_decode_reqs) >= _MIN_BATCH_SIZE_FOR_BATCHING:
                        decode_tokens = self._batched_decode(valid_decode_reqs)
                    else:
                        decode_tokens = self._sequential_decode(valid_decode_reqs)

                    # Add decode results to output
                    for i, (req_id, _) in enumerate(valid_decode_reqs):
                        req_ids.append(req_id)
                        req_id_to_index[req_id] = len(req_ids) - 1
                        sampled_tokens.append([decode_tokens[i]])

                # Handle requests with no cached state (edge case)
                for req_id in decode_req_ids:
                    if req_id not in req_id_to_index:
                        req_ids.append(req_id)
                        req_id_to_index[req_id] = len(req_ids) - 1
                        sampled_tokens.append([0])

        # Consistency check: every scheduled request must be represented in
        # req_ids, and decode-phase scheduled requests should not emit empty
        # token lists. Missing/empty outputs here can leave placeholders stale.
        if scheduler_output.total_num_scheduled_tokens > 0:
            new_reqs_by_id = {r.req_id: r for r in new_reqs}
            missing_req_ids: list[str] = []
            unexpected_empty_req_ids: list[str] = []
            for req_id in scheduler_output.num_scheduled_tokens:
                idx = req_id_to_index.get(req_id)
                if idx is None:
                    missing_req_ids.append(req_id)
                    continue
                if sampled_tokens[idx]:
                    continue

                # The only valid empty-output case is an intermediate
                # prefill chunk (generated_tokens == 0 means still
                # prefilling).
                state = self._request_states.get(req_id)
                is_intermediate_ctx = state is not None and state.generated_tokens == 0
                # Also check PHASE 1 intermediate chunks
                if not is_intermediate_ctx:
                    new_req = new_reqs_by_id.get(req_id)
                    if new_req is not None:
                        prompt_len = len(new_req.prompt_token_ids or [])
                        computed = new_req.num_computed_tokens
                        scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                        is_intermediate_ctx = computed + scheduled < prompt_len

                if not is_intermediate_ctx:
                    unexpected_empty_req_ids.append(req_id)

            if missing_req_ids or unexpected_empty_req_ids:
                logger.error(
                    "ModelRunner scheduled/output mismatch: scheduled=%d emitted=%d "
                    "missing=%d unexpected_empty=%d",
                    len(scheduler_output.num_scheduled_tokens),
                    len(req_ids),
                    len(missing_req_ids),
                    len(unexpected_empty_req_ids),
                )
                if missing_req_ids:
                    logger.error("Missing scheduled req ids: %s", missing_req_ids[:16])
                if unexpected_empty_req_ids:
                    logger.error(
                        "Unexpected empty outputs for req ids: %s",
                        unexpected_empty_req_ids[:16],
                    )

        # === PHASE 3: Clean up finished requests ===
        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                state = self._request_states.pop(req_id, None)
                if state is not None:
                    if state.cache:
                        del state.cache
                    del state

                # Free paged KV blocks
                if self._paged_kv_cache is not None:
                    paged_cache = self._paged_kv_cache
                    if paged_cache.has_sequence(req_id):
                        paged_cache.free_sequence(req_id)
                    self._paged_request_seq_lens.pop(req_id, None)

                # Remove from Rust state manager if available
                if self._rust_state_manager is not None:
                    self._rust_state_manager.remove_request(req_id)

            # Lazy cache clearing - only clear periodically to avoid sync overhead
            self._finished_request_count += len(scheduler_output.finished_req_ids)
            if self._finished_request_count >= _CACHE_CLEAR_INTERVAL:
                mx.clear_cache()
                self._finished_request_count = 0

                # Log prefix cache stats periodically
                if self._prefix_cache is not None:
                    stats = self._prefix_cache.get_stats()
                    logger.info(
                        "Prefix cache: %.1f%% hit rate "
                        "(hits=%d, misses=%d, cached=%d, "
                        "%.1fMB/%.1fMB)",
                        stats["hit_rate"] * 100,
                        stats["hits"],
                        stats["misses"],
                        stats["cached_entries"],
                        stats["current_bytes"] / (1024 * 1024),
                        stats["max_bytes"] / (1024 * 1024),
                    )

        # Handle empty case — return directly so the batch-queue path in
        # step_with_batch_queue receives a non-None result from the
        # execute_model future (when model_executed=False, sample_tokens is
        # never called, so _pending_output would go unconsumed).
        if not req_ids:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        self._pending_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_tokens,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
        )
        return None

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | None:
        """Return sampled tokens produced by the last execute_model call.

        vLLM's v1 engine calls ``sample_tokens`` after a successful
        ``execute_model`` call that returned ``None``. When async scheduling is
        enabled, vLLM may still call ``sample_tokens`` even if ``execute_model``
        failed; returning ``None`` in that case allows vLLM to surface the
        original exception from ``execute_model``.
        """
        del grammar_output
        if self._pending_output is None:
            model_id = None
            model_config = getattr(self, "model_config", None)
            if model_config is not None:
                model_id = getattr(model_config, "model", None)

            if getattr(self, "use_async_scheduling", False):
                logger.error(
                    "sample_tokens called without pending output from "
                    "execute_model (model=%r). Returning None so vLLM can "
                    "surface the original execute_model error.",
                    model_id,
                )
                return None

            raise RuntimeError(
                "State error: sample_tokens called without pending output from "
                f"execute_model (model={model_id!r})."
            )
        output = self._pending_output
        self._pending_output = None
        return output

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from a prompt.

        This is a simplified interface for direct text generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded")

        segments: list[str] = []

        # Create sampler based on temperature (mlx_lm 0.29+ uses sampler param)
        def sampler(logits: mx.array) -> mx.array:
            if temperature < 1e-5:
                return mx.argmax(logits, axis=-1)
            return mx.random.categorical(logits / temperature)

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            segments.append(response.text)

        return "".join(segments)
