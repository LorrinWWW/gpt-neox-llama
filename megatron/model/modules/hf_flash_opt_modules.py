import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.opt.configuration_opt import OPTConfig as GPTConfig
from transformers.models.opt.modeling_opt import (
    ACT2FN,
    OPTDecoderLayer,
    OPTLearnedPositionalEmbedding,
)
from transformers.models.opt.modeling_opt import OPTAttention as _OPTAttention



try:
    from flash_attn.flash_attention import FlashAttention
    flash_attn_installed = True
    print('>>>>> using flash attention')
except ImportError:
    flash_attn_installed = False

try:
    from fav2.fav2_interface import flash_attn_qkvpacked_func as fav2_qkvpacked_func
    flash_attn_v2_installed = True
    print('>>>>> using flash attention v2')

    class FlashAttentionV2(nn.Module):
        """Implement the scaled dot product attention with softmax.
        Arguments
        ---------
            softmax_scale: The temperature to use for the softmax attention.
                          (default: 1/sqrt(d_keys) where d_keys is computed at
                          runtime)
            attention_dropout: The dropout rate to apply to the attention
                               (default: 0.0)
        """
        def __init__(self, softmax_scale=None, attention_dropout=0.0):
            super().__init__()
            self.softmax_scale = softmax_scale
            self.dropout_p = attention_dropout
    
        def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                    max_s=None, need_weights=False):
            """Implements the multihead softmax attention.
            Arguments
            ---------
                qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                    if unpadded: (nnz, 3, h, d)
                key_padding_mask: a bool tensor of shape (B, S)
            """
            assert not need_weights
            assert qkv.dtype in [torch.float16, torch.bfloat16]
            assert qkv.is_cuda
            assert key_padding_mask is None
            assert cu_seqlens is None
            assert max_s is None

            output = fav2_qkvpacked_func(
                qkv, self.dropout_p if self.training else 0.0, 
                softmax_scale=self.softmax_scale, causal=causal
            )
    
            return output, None
except ImportError:
    flash_attn_v2_installed = False

try:
    import apex

    apex_installed = True
    print(">>>>> apex")
except ImportError:
    apex_installed = False

try:
    import xformers.ops as xops

    xops_installed = True
    print(">>>>> Xformers installed")
except:
    xops_installed = False


from einops import rearrange


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(float("-inf")), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def _prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class GPTEmbeddings(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.word_embed_proj_dim,
            self.padding_idx,
            device=device,
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
                device=device,
            )
        else:
            self.project_in = None

    def forward(self, input_ids, past_layer=None, mask=None, **kargs):
        if mask is None:
            if past_layer is not None:
                past_length = past_layer[0].size(2)
            else:
                past_length = 0
        else:
            # masked tokens
            past_length = (mask - 1).sum(-1, keepdims=True)
            if past_layer is not None:
                past_length += past_layer[0].size(2)

        device = input_ids.device
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        input_ids.shape[0]

        inputs_embeds = self.embed_tokens(input_ids)

        # attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        # position_embeds = self.embed_positions(attention_mask, past_length)
        # position ids
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_ids = position_ids + past_length + self.embed_positions.offset
        position_ids[position_ids < 0] = 0

        position_embeds = F.embedding(
            position_ids,
            self.embed_positions.weight,
            self.embed_positions.padding_idx,
            self.embed_positions.max_norm,
            self.embed_positions.norm_type,
            self.embed_positions.scale_grad_by_freq,
            self.embed_positions.sparse,
        )

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + position_embeds

        # hidden_states = self.drop(hidden_states)

        return hidden_states


class OPTAttention(_OPTAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        device="cpu",
    ):
        super(_OPTAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)

        if flash_attn_v2_installed:
            self.flash_attn = FlashAttentionV2(softmax_scale=self.scaling, attention_dropout=0)
        elif flash_attn_installed:
            self.flash_attn = FlashAttention(softmax_scale=self.scaling, attention_dropout=0)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder

        bsz, tgt_len, _ = hidden_states.size()

        assert flash_attn_installed

        # get query proj
        query_states = self.q_proj(hidden_states)  # B S H
        key_states = self.k_proj(hidden_states)  # B S H
        value_states = self.v_proj(hidden_states)  # B S H

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            pass

        qkv = torch.stack(
            [
                query_states.view((bsz, tgt_len, self.num_heads, self.head_dim)),
                key_states.view((bsz, tgt_len, self.num_heads, self.head_dim)),
                value_states.view((bsz, tgt_len, self.num_heads, self.head_dim)),
            ],
            dim=2,
        )

        if flash_attn_v2_installed:
            attn_output, _ = self.flash_attn(qkv, causal=True)
        elif xops_installed:
            q, k, v = qkv.unbind(2)
            attn_output = xops.memory_efficient_attention(
                q, k, v, attn_bias=xops.LowerTriangularMask()
            )
        elif flash_attn_installed:
            attn_output, _ = self.flash_attn(qkv, causal=True)
        else:
            raise Exception('Flash Attention not found.')
            
        attn_output = attn_output.reshape((bsz, tgt_len, self.embed_dim))
        attn_output = self.out_proj(attn_output)

        return attn_output, None, None


class GPTBlock(OPTDecoderLayer):
    def __init__(self, config, *args, use_checkpoint=True, device="cpu", **kargs):
        # super().__init__(config=config, *args, **kargs)
        super(OPTDecoderLayer, self).__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            device=device,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.activation_dropout = config.activation_dropout

        if apex_installed:
            self.self_attn_layer_norm = apex.normalization.FusedLayerNorm(
                self.embed_dim
            )
            self.final_layer_norm = apex.normalization.FusedLayerNorm(self.embed_dim)
        else:
            self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, device=device)
            self.final_layer_norm = nn.LayerNorm(self.embed_dim, device=device)

        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, device=device)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, device=device)

        self.config = config
        self.use_checkpoint = use_checkpoint

        def attn_res(hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
            residual = hidden_states
            if self.do_layer_norm_before:
                hidden_states = self.self_attn_layer_norm(hidden_states)

            # Self Attention
            hidden_states, _, present = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = residual + hidden_states

            # 350m applies layer norm AFTER attention
            if not self.do_layer_norm_before:
                hidden_states = self.self_attn_layer_norm(hidden_states)

            return hidden_states

        self.attn_res = attn_res

        def mlp_res(hidden_states: torch.Tensor) -> torch.Tensor:
            # Fully Connected
            hidden_states_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            residual = hidden_states

            # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
            if self.do_layer_norm_before:
                hidden_states = self.final_layer_norm(hidden_states)

            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)

            hidden_states = self.fc2(hidden_states)

            hidden_states = (residual + hidden_states).view(hidden_states_shape)
            return hidden_states

        self.mlp_res = mlp_res

    def forward(
        self, x: torch.Tensor, layer_past=None, mask=None, *args, **kargs
    ) -> torch.Tensor:
        if layer_past is not None:
            past_length = layer_past[0].size(2)
        else:
            past_length = 0
        if mask is None:
            mask = torch.ones(
                (x.size(0), x.size(1) + past_length), dtype=torch.bool, device=x.device
            )
        attention_mask = _prepare_decoder_attention_mask(
            mask, x.shape[:2], x, past_length
        )

        if self.training:
            if self.use_checkpoint:
                x.requires_grad_(True)
                x = checkpoint(self.attn_res, x, attention_mask)
            else:
                x = self.attn_res(x, attention_mask)

            if self.use_checkpoint:
                x.requires_grad_(True)
                x = checkpoint(self.mlp_res, x)
            else:
                x = self.mlp_res(x)

            return x

        else:
            hidden_states = x  # alias
            residual = hidden_states

            # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
            if self.do_layer_norm_before:
                hidden_states = self.self_attn_layer_norm(hidden_states)

            # Self Attention
            hidden_states, _, present = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
            )
            hidden_states = residual + hidden_states

            # 350m applies layer norm AFTER attention
            if not self.do_layer_norm_before:
                hidden_states = self.self_attn_layer_norm(hidden_states)

            # Fully Connected
            hidden_states_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            residual = hidden_states

            # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
            if self.do_layer_norm_before:
                hidden_states = self.final_layer_norm(hidden_states)

            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)

            hidden_states = self.fc2(hidden_states)

            hidden_states = (residual + hidden_states).view(hidden_states_shape)

            return hidden_states


class GPTLMHead(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size, device=device)
        else:
            self.final_layer_norm = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
                device=device,
            )
        else:
            self.project_out = None

        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False, device=device
        )

    def forward(self, x, input_ids=None, *args, **kargs):
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        if self.project_out is not None:
            x = self.project_out(x)
        x = self.lm_head(x)
        return x
