"""Config-only gpt-oss serving operations, independent of the execution host.

Packed QKV and residual-stream structure follow the CUDA serving implementation:
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/gpt_oss.py
Native attention, normalization, routing and cache operations are shape contracts,
not executed CUDA kernels. Cache dimensions describe logical active KV, not an
engine's allocated paged pool. Quantized storage and launch counts are not inferred.
"""

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from compiled_graph import expert_dispatch


def require_shape_tensor(value):
    if value.device.type != "meta":
        raise ValueError("CUDA serving adapter supports shape tensors only")


def norm_shape(hidden_states, weight, residual, eps):
    return torch.empty_like(hidden_states), torch.empty_like(hidden_states)


@torch.library.custom_op("model_graph::fused_add_rms_norm", mutates_args=())
def fused_add_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor,
                       residual: torch.Tensor | None, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    require_shape_tensor(hidden_states)
    return norm_shape(hidden_states, weight, residual, eps)


fused_add_rms_norm.register_fake(norm_shape)


def rope_cache_shape(query, key, value, positions, cos_sin_cache, past_keys, past_values,
                     query_heads, kv_heads, head_dim):
    batch, tokens = query.shape[:2]
    context = tokens + (past_keys.shape[-2] if past_keys is not None else 0)
    return (query.new_empty(batch, query_heads, tokens, head_dim),
            key.new_empty(batch, kv_heads, context, head_dim),
            value.new_empty(batch, kv_heads, context, head_dim))


@torch.library.custom_op("model_graph::rope_kv_cache", mutates_args=())
def rope_kv_cache(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  positions: torch.Tensor, cos_sin_cache: torch.Tensor, past_keys: torch.Tensor | None,
                  past_values: torch.Tensor | None, query_heads: int,
                  kv_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_shape_tensor(query)
    return rope_cache_shape(query, key, value, positions, cos_sin_cache, past_keys, past_values,
                            query_heads, kv_heads, head_dim)


rope_kv_cache.register_fake(rope_cache_shape)


def attention_shape(query, keys, values, sinks, sliding_window):
    batch, heads, tokens, dim = query.shape
    return query.new_empty(batch, tokens, heads * dim)


@torch.library.custom_op("model_graph::native_attention", mutates_args=())
def native_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                      sinks: torch.Tensor, sliding_window: int) -> torch.Tensor:
    require_shape_tensor(query)
    return attention_shape(query, keys, values, sinks, sliding_window)


native_attention.register_fake(attention_shape)


def topk_shape(logits, count):
    shape = (*logits.shape[:-1], count)
    return logits.new_empty(shape), logits.new_empty(shape, dtype=torch.int32)


@torch.library.custom_op("model_graph::topk_softmax", mutates_args=())
def topk_softmax(logits: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    require_shape_tensor(logits)
    return topk_shape(logits, count)


topk_softmax.register_fake(topk_shape)


class ServingLayerCache:
    def __init__(self):
        self.keys = None
        self.values = None


class ServingCache:
    def __init__(self, layers):
        self.layers = [ServingLayerCache() for _ in range(layers)]


class ServingRMSNorm(torch.nn.Module):
    graph_operation = "FusedAddRMSNorm"
    graph_shape_contract = "ResidualAdd (when present) -> RMSNorm; native serving boundary"

    def __init__(self, config, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(config.hidden_size, dtype=dtype))
        self.eps = config.rms_norm_eps

    def forward(self, hidden_states, residual=None):
        return fused_add_rms_norm(hidden_states, self.weight, residual, self.eps)


class ServingRoPEKVCache(torch.nn.Module):
    graph_operation = "RoPEAndKVCache"
    graph_shape_contract = "RoPE(Q,K) -> KVCacheWrite; logical active KV dimensions"

    def __init__(self, config):
        super().__init__()
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.register_buffer("cos_sin_cache", torch.empty(config.max_position_embeddings, config.head_dim, dtype=torch.float32))

    def forward(self, query, key, value, positions, past_keys=None, past_values=None):
        return rope_kv_cache(query, key, value, positions, self.cos_sin_cache, past_keys, past_values,
                             self.query_heads, self.kv_heads, self.head_dim)


class ServingAttentionCore(torch.nn.Module):
    graph_operation = "NativeAttention"
    graph_shape_contract = "Native decode/prefill attention with sinks; sliding-window selection inside backend"

    def __init__(self, config, layer_idx, dtype):
        super().__init__()
        self.sinks = torch.nn.Parameter(torch.empty(config.num_attention_heads, dtype=dtype))
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else -1

    def forward(self, query, keys, values):
        return native_attention(query, keys, values, self.sinks, self.sliding_window)


class ServingAttention(torch.nn.Module):
    def __init__(self, config, layer_idx, dtype):
        super().__init__()
        self.q_size = config.num_attention_heads * config.head_dim
        self.kv_size = config.num_key_value_heads * config.head_dim
        self.qkv_proj = torch.nn.Linear(config.hidden_size, self.q_size + 2 * self.kv_size,
                                        bias=config.attention_bias, dtype=dtype)
        self.rope_cache = ServingRoPEKVCache(config)
        self.attn = ServingAttentionCore(config, layer_idx, dtype)
        self.o_proj = torch.nn.Linear(self.q_size, config.hidden_size,
                                      bias=config.attention_bias, dtype=dtype)

    def forward(self, hidden_states, positions, cache=None):
        packed = self.qkv_proj(hidden_states)
        query, key, value = packed.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query, keys, values = self.rope_cache(query, key, value, positions,
                                             cache.keys if cache is not None else None,
                                             cache.values if cache is not None else None)
        if cache is not None:
            cache.keys, cache.values = keys, values
        return self.o_proj(self.attn(query, keys, values))


class ServingRouter(torch.nn.Module):
    graph_operation = "Router"

    def __init__(self, config, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(config.num_local_experts, config.hidden_size, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.empty(config.num_local_experts, dtype=dtype))
        self.count = config.num_experts_per_tok

    def forward(self, hidden_states):
        return topk_softmax(torch.nn.functional.linear(hidden_states, self.weight, self.bias), self.count)


class ServingExperts(torch.nn.Module):
    graph_operation = "ExpertDispatch"
    graph_shape_contract = "Gather -> ExpertMLP -> RoutingWeightMultiply -> Scatter; tokens_per_expert=data-dependent"

    def __init__(self, config, dtype):
        super().__init__()
        experts, hidden, intermediate = config.num_local_experts, config.hidden_size, config.intermediate_size
        self.gate_up_proj = torch.nn.Parameter(torch.empty(experts, hidden, 2 * intermediate, dtype=dtype))
        self.gate_up_proj_bias = torch.nn.Parameter(torch.empty(experts, 2 * intermediate, dtype=dtype))
        self.down_proj = torch.nn.Parameter(torch.empty(experts, intermediate, hidden, dtype=dtype))
        self.down_proj_bias = torch.nn.Parameter(torch.empty(experts, hidden, dtype=dtype))

    def forward(self, hidden_states, indices, scores):
        return expert_dispatch(hidden_states, [indices, scores],
                                [self.gate_up_proj, self.gate_up_proj_bias, self.down_proj, self.down_proj_bias])


class ServingMLP(torch.nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.router = ServingRouter(config, dtype)
        self.experts = ServingExperts(config, dtype)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, shape[-1])
        scores, indices = self.router(tokens)
        return self.experts(tokens, indices, scores).reshape(shape)


class ServingDecoderLayer(torch.nn.Module):
    def __init__(self, config, layer_idx, dtype):
        super().__init__()
        self.input_layernorm = ServingRMSNorm(config, dtype)
        self.self_attn = ServingAttention(config, layer_idx, dtype)
        self.post_attention_layernorm = ServingRMSNorm(config, dtype)
        self.mlp = ServingMLP(config, dtype)

    def forward(self, hidden_states, positions, residual, cache):
        normalized, residual = self.input_layernorm(hidden_states, residual)
        attention = self.self_attn(normalized, positions, cache)
        normalized, residual = self.post_attention_layernorm(attention, residual)
        return self.mlp(normalized), residual


class ServingModel(torch.nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = torch.nn.ModuleList(ServingDecoderLayer(config, i, dtype) for i in range(config.num_hidden_layers))
        self.norm = ServingRMSNorm(config, dtype)

    def forward(self, input_ids, positions, cache):
        hidden_states, residual = self.embed_tokens(input_ids), None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, positions, residual, cache.layers[i] if cache else None)
        return self.norm(hidden_states, residual)[0]


class CudaServingForCausalLM(torch.nn.Module):
    def __init__(self, config, dtype=None):
        super().__init__()
        self.config = config
        dtype = dtype or torch.bfloat16
        self.model = ServingModel(config, dtype)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, position_ids, past_key_values=None, use_cache=False, return_dict=True):
        require_shape_tensor(input_ids)
        cache = past_key_values
        if use_cache and cache is None:
            cache = ServingCache(len(self.model.layers))
        logits = self.lm_head(self.model(input_ids, position_ids, cache))
        if return_dict:
            return CausalLMOutputWithPast(logits=logits, past_key_values=cache)
        return (logits, cache) if use_cache else (logits,)
