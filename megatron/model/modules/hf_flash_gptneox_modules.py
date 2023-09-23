import os
from typing import List, Optional, Tuple, Union

import torch

from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.gpt_neox.configuration_gpt_neox import (
    GPTNeoXConfig as GPTConfig,
)
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention as _GPTNeoXAttention,
)
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer as _GPTNeoXBlock
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP


from flash_attn.layers.rotary import RotaryEmbedding as _RotaryEmbedding

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func

    flash_attn_v2_installed = True
    print(">>>>> using flash attention v2")
except ImportError:
    flash_attn_v2_installed = False

try:
    import xformers.ops as xops

    xops_installed = True
    print(">>>>> Xformers installed")
except:
    xops_installed = False


try:
    import apex.contrib.layer_norm

    # LayerNorm = apex.normalization.FusedLayerNorm
    LayerNorm = apex.contrib.layer_norm.FastLayerNorm
    print(">>>>> Apex FastLayerNorm")
except:
    LayerNorm = nn.LayerNorm

try:
    import apex.fused_dense

    def _fused_dense_gelu_dense(input, weight1, bias1, weight2, bias2):
        return apex.fused_dense.FusedDenseGeluDenseFunc.apply(
            input, weight1, bias1, weight2, bias2
        )

    fused_dense_installed = True
    print(">>>>> Apex FusedDenseGeluDense")
except:
    fused_dense_installed = False


GPTNEOX_DEFAULT_SEQ_LENGTH = 2048


class RotaryEmbedding(_RotaryEmbedding):

    # This class inherits from flash_attn.layers.rotary.RotaryEmbedding and add linear position interpolation.

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        device=None,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(
            dim=dim,
            base=base,
            interleaved=interleaved,
            scale_base=scale_base,
            pos_idx_in_fp32=pos_idx_in_fp32,
            device=device,
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # linear interpolation
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                # linear interpolation
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(
                    power, "s -> s 1"
                )
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)


if flash_attn_v2_installed:

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

        def forward(
            self,
            qkv,
            key_padding_mask=None,
            causal=False,
            cu_seqlens=None,
            max_s=None,
            need_weights=False,
        ):
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

            output = flash_attn_qkvpacked_func(
                qkv,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )

            return output, None


from einops import rearrange


class GPTNeoXAttention(_GPTNeoXAttention):
    def __init__(self, config):
        super(_GPTNeoXAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.max_positions = max_positions
        self.config = config

        if self.config.rope_scaling is None:
            # by default do linear scale if not specified.
            scaling_factor = max(self.max_positions / GPTNEOX_DEFAULT_SEQ_LENGTH, 1.0)
            print(f"Linearly scaling {scaling_factor}x.")
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            assert scaling_type == "linear"
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            base=config.rotary_emb_base,
            interleaved=False,
            scaling_factor=scaling_factor,
        )

        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(
                torch.get_default_dtype()
            ),
            persistent=False,
        )
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if flash_attn_v2_installed:
            self.flash_attn = FlashAttentionV2(
                softmax_scale=1.0 / self.norm_factor, attention_dropout=0
            )
        else:
            self.flash_attn = None

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        offset=None,
        output_attentions=False,
    ):

        bsz, tgt_len, _ = hidden_states.shape

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        attn_weights = None
        present = None

        qkv = rearrange(
            qkv, "... (h three d) -> ... h three d", three=3, d=self.head_size
        )
        qkv = qkv.permute(0, 1, 3, 2, 4)
        qkv = self.rotary_emb(qkv)

        if flash_attn_v2_installed:
            attn_output, _ = self.flash_attn(qkv, causal=True)
        elif xops_installed:
            q, k, v = qkv.unbind(2)
            attn_output = xops.memory_efficient_attention(
                q, k, v, attn_bias=xops.LowerTriangularMask()
            )
        else:
            raise Exception("Flash Attention not found.")

        attn_output = attn_output.view(
            bsz, tgt_len, self.num_attention_heads * self.head_size
        )
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.embed_in = nn.Embedding(config.vocab_size, self.embed_dim)

    # @torch.compile
    def forward(self, input_ids, *args, **kargs):
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embed_in(input_ids)
        return hidden_states


class GPTBlock(_GPTNeoXBlock):
    def __init__(self, config, *args, use_checkpoint=True, **kargs):
        super(_GPTNeoXBlock, self).__init__()
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = GPTNeoXAttention(config)
        self.mlp = GPTNeoXMLP(config)
        self.config = config
        self.use_checkpoint = use_checkpoint

        def mha_fw(x: torch.Tensor, res: torch.Tensor, attention_mask: torch.Tensor):
            attention_layer_output = self.attention(
                self.input_layernorm(x), attention_mask=attention_mask
            )
            attn_output = attention_layer_output[0]
            return attn_output + res

        # @torch.compile()
        def mlp_fw(x: torch.Tensor, res: torch.Tensor):
            if fused_dense_installed:
                shape = x.shape
                x = self.post_attention_layernorm(x)
                x = x.view(-1, config.hidden_size)
                mlp_out = _fused_dense_gelu_dense(
                    x,
                    self.mlp.dense_h_to_4h.weight,
                    self.mlp.dense_h_to_4h.bias,
                    self.mlp.dense_4h_to_h.weight,
                    self.mlp.dense_4h_to_h.bias,
                ).view(shape)
            else:
                mlp_out = self.mlp(self.post_attention_layernorm(x))
            return mlp_out + res

        """
        To be compatible with https://github.com/huggingface/transformers/blob/a0ae2310ec46a2c592950babc85cf02e325bf6a7/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L336-L347
        """
        if self.config.use_parallel_residual:
            # @torch.compile()
            def block_forward(
                x: torch.Tensor,
                attention_mask: torch.Tensor,
                prefix_masks: torch.Tensor,
            ) -> torch.Tensor:
                attn_output = mha_fw(x, res=x, attention_mask=attention_mask)

                # x = x + attn(ln1(x)) + mlp(ln2(x))
                # x_a = attn_output,
                mlp_out = mlp_fw(x, res=attn_output)
                return mlp_out

        else:
            # @torch.compile()
            def block_forward(
                x: torch.Tensor,
                attention_mask: torch.Tensor,
                prefix_masks: torch.Tensor,
            ) -> torch.Tensor:
                attn_output = mha_fw(x, res=x, attention_mask=attention_mask)

                # x = x + attn(ln1(x))
                # x = x + mlp(ln2(x))
                mlp_out = mlp_fw(attn_output, res=attn_output)
                return mlp_out

        self.block_forward = block_forward

    def forward(
        self, x: torch.Tensor, layer_past=None, mask=None, **kargs
    ) -> torch.Tensor:
        if mask is not None:
            # bool -> float
            attention_mask = 1e9 * (mask[:, None, None, :] - 1)
        else:
            attention_mask = None

        if self.training:
            if self.use_checkpoint:
                x.requires_grad_(True)
                x = checkpoint(self.block_forward, x, attention_mask, None)
            else:
                x = self.block_forward(x, attention_mask, None)

            return x

        else:
            residual = x
            ln_out = self.input_layernorm(x)
            attention_layer_outputs = self.attention(
                ln_out,
                attention_mask=attention_mask,
            )
            attn_output = attention_layer_outputs[0]  # output_attn: a, present, ...

            mlp_output = self.mlp(self.post_attention_layernorm(x))
            x = mlp_output + attn_output + residual

            return x


class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # @torch.compile
    def forward(self, x, *args, **kargs):
        x = self.final_layer_norm(x)
        x = self.embed_out(x)
        return x
