import os
from typing import List, Optional, Tuple, Union

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig as GPTConfig
from transformers.models.bloom.modeling_bloom import (
    BloomAttention as _BloomAttention,
)
from transformers.models.bloom.modeling_bloom import BloomBlock as _BloomBlock
from transformers.models.bloom.modeling_bloom import BloomMLP

LayerNorm = nn.LayerNorm


try:
    import xformers.ops as xops

    xops_installed = True
    print(">>>>> Xformers installed")
except:
    xops_installed = False

assert xops_installed


def build_alibi_tensor(
    attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=attention_mask.device,
        dtype=torch.float32,
    )
    powers = torch.arange(
        1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32
    )
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=attention_mask.device,
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
            device=attention_mask.device,
            dtype=torch.int32,
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    # jue: change from (batch_size * num_heads, 1, seq_length) to (batch_size, num_heads, 1, seq_length)
    return alibi.reshape(batch_size, num_heads, 1, seq_length).to(dtype)


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty(
        (target_length, target_length + past_key_values_length),
        dtype=torch.bool,
        device=device,
    )
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def dropout_add(
    x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool
) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            esidual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class BloomAttention(_BloomAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        attn_bias: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        bsz, q_len, _ = hidden_states.shape
        qkv = self.query_key_value(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (q, k, v) = self._split_heads(qkv)

        attn_output = xops.memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=attn_bias,
        )

        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.dense(attn_output)

        attn_output = dropout_add(
            attn_output, residual, self.hidden_dropout, self.training
        )

        return (attn_output,)


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon
        )

    def forward(self, input_ids, *args, **kargs):
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.word_embeddings(input_ids)
        hidden_states = self.word_embeddings_layernorm(hidden_states)
        return hidden_states


class GPTBlock(_BloomBlock):
    def __init__(self, config, *args, use_checkpoint=True, **kargs):
        super(_BloomBlock, self).__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon
        )
        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        assert not self.apply_residual_connection_post_layernorm

        self.hidden_dropout = config.hidden_dropout

        self.config = config
        self.use_checkpoint = use_checkpoint

        self.cached_attn_bias = None

        def block_forward(
            x: torch.Tensor,
            attn_bias: torch.Tensor,
        ) -> torch.Tensor:
            res = x
            x = self.input_layernorm(x)
            x = self.self_attention(x, res, attn_bias=attn_bias)[0]
            res = x
            x = self.post_attention_layernorm(x)
            x = self.mlp(x, res)
            return x

        self.block_forward = block_forward

    def _prepare_attn_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        past_key_values_length: int,
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def forward(self, x: torch.Tensor, mask=None, **kargs) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape

        if mask is not None:
            # bool -> float
            attention_mask = mask
        else:
            attention_mask = torch.ones([batch_size, seq_length], device=x.device)

        if self.cached_attn_bias is None:
            create_as = x.dtype if x.dtype is not torch.bfloat16 else torch.float32
            tensor = torch.full(  # type: ignore
                (batch_size, self.num_heads, seq_length, seq_length),
                dtype=create_as,
                fill_value=float("-inf"),
                device=x.device,
            )
            self.cached_attn_bias = torch.triu(tensor, diagonal=1)
            # alibi
            self.cached_attn_bias.data = self.cached_attn_bias + build_alibi_tensor(
                attention_mask=attention_mask,
                num_heads=self.num_heads,
                dtype=torch.float32,
            )
            self.cached_attn_bias = self.cached_attn_bias.to(x.dtype)

        if self.training:
            if self.use_checkpoint:
                x.requires_grad_(True)
                x = checkpoint(self.block_forward, x, self.cached_attn_bias)
            else:
                x = self.block_forward(x, self.cached_attn_bias)

            return x

        else:
            x = self.block_forward(x, self.cached_attn_bias)
            return x


class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x, *args, **kargs):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
