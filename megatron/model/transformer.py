# Copyright (c) 2021 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer."""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from .norms import get_norm
from megatron import print_rank_0, mpu
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists, get_fusion_type
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_torch,
    AliBi,
)
from megatron.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.model.utils import configure_sparse_attention
from megatron.logging import tb_wandb_log


try:
    from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
    flash_attn_v2_installed = True
    from flash_attn.layers.rotary import RotaryEmbedding as _RotaryEmbedding

    class FlashRotaryEmbedding(_RotaryEmbedding):
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
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
                or (self.training and self._cos_cached.is_inference())
            ):
                self._seq_len_cached = seqlen
                if self.pos_idx_in_fp32:
                    t = torch.arange(seqlen, device=device, dtype=torch.float32)
                    # linear interpolation
                    t /= self.scaling_factor
                    if self.inv_freq.dtype != torch.float32:
                        inv_freq = self._compute_inv_freq(device=device)
                    else:
                        inv_freq = self.inv_freq
                else:
                    t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                    # linear interpolation
                    t /= self.scaling_factor
                    inv_freq = self.inv_freq
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
except ImportError:
    flash_attn_v2_installed = False
    print('Warn: Error when importing flash attention.')


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        args,
        no_reduce=False,
    ):
        from transformers import AutoConfig

        if args.hf_config_name_or_path is not None:
            config = AutoConfig.from_pretrained(args.hf_config_name_or_path)
        else:
            config = AutoConfig.from_pretrained(args.load)
        
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        max_positions = config.max_position_embeddings
        self.max_positions = max_positions
        self.config = config

        self.q_proj = mpu.ColumnParallelLinear(
            neox_args=args,
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.k_proj = mpu.ColumnParallelLinear(
            neox_args=args,
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.v_proj = mpu.ColumnParallelLinear(
            neox_args=args,
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.o_proj = mpu.RowParallelLinear(
            neox_args=args,
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            input_is_parallel=True,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            parallel_output=no_reduce, # True if gpt-j-parallel
            bias=False,
        )

        self.rotary_ndims = self.head_dim
        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(
                torch.get_default_dtype()
            ),
            persistent=False,
        )

        if self.config.rope_scaling is None:
            # by default do linear scale if not specified.
            scaling_factor = max(self.max_positions / 4096, 1.0)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            assert scaling_type == "linear"

        if scaling_factor != 1:
            print(f"Linearly scaling {scaling_factor}x.")
        
        self.rotary_emb = FlashRotaryEmbedding(
            self.rotary_ndims,
            base=config.rope_theta,
            interleaved=False,
            scaling_factor=scaling_factor,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        hidden_states = hidden_states.transpose(0, 1)
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)[0].view(
            bsz, q_len, -1, self.head_dim
        )
        key_states = self.k_proj(hidden_states)[0].view(
            bsz, q_len, -1, self.head_dim
        )
        value_states = self.v_proj(hidden_states)[0].view(
            bsz, q_len, -1, self.head_dim
        )

        q = query_states
        kv = torch.stack([key_states, value_states], dim=2)
        q, kv = self.rotary_emb(q, kv)
        
        if flash_attn_v2_installed:
            attn_output = flash_attn_kvpacked_func(
                q, kv, 0.0,
                causal=True,
            )
        else:
            raise Exception("Flash Attention not found.")
        
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)[0]

        attn_output = attn_output.transpose(0, 1)

        return attn_output, None # bias



def log_norm(norm, key, iteration_no):
    if norm is not None:
        tb_wandb_log(
            key,
            norm,
            iteration_no,
            use_wandb=True,
            tensorboard_writer=None,
            all_ranks=False,
        )


# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmasked-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmasked-attention-scores, attention-mask)
"""


class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self, neox_args, init_method, output_layer_init_method, parallel_output=False
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.bias_gelu_fusion = neox_args.bias_gelu_fusion

        # auto scale so geglu has equal parameters
        ff_mult = 4 * 2 / 3 if self.activation_type == "geglu" else 4
        ff_dim = (
            int(ff_mult * neox_args.hidden_size) * 2
            if self.activation_type == "geglu"
            else ff_mult * neox_args.hidden_size
        )
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim
        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim_in,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if (
            self.activation_type == "gelu" and self.bias_gelu_fusion
        ) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class LLaMAParallelMLP(nn.Module):
    """LLaMA's MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Note: multiple_of is used to compute the hidden dimension of the MLP
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        multiple_of=256,
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        ff_dim = int(2 * neox_args.hidden_size * 4 / 3)
        ff_dim = self.multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
        self.gate_proj = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )
        self.up_proj = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )
        self.down_proj = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=False,
        )

    def forward(self, hidden_states):
        w1_out, _ = self.gate_proj(hidden_states)
        w3_out, _ = self.up_proj(hidden_states)
        
        return self.down_proj(self.activation_func(w1_out) * w3_out)


class ParallelLinear(nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(
        self,
        neox_args,
        parallel_output=True,
        init_method=nn.init.xavier_normal_,
        is_last_layer=False,
    ):
        super().__init__()
        parallelism = neox_args.output_layer_parallelism
        if parallelism == "column":
            self.final_linear = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.padded_vocab_size,
                bias=False,
                init_method=init_method,
                gather_output=not parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # rescale params only called if neox_args.use_mup = True, despite it not being included here
            )
        else:
            self.final_linear = mpu.RowParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.padded_vocab_size,
                bias=False,
                input_is_parallel=False,
                init_method=init_method,
                parallel_output=parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # only called if neox_args.use_mup = True, despite it not being included here
            )

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)


class ParallelTransformerLayer(nn.Module):
    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
    ):
        super().__init__()

        self.layer_number = layer_number

        self.log_attn_norms = neox_args.log_attn_norms
        self.neox_args = neox_args

        self.hidden_dropout = neox_args.hidden_dropout
        self.bias_dropout_fusion = neox_args.bias_dropout_fusion
        self.gpt_j_residual = neox_args.gpt_j_residual
        self.gpt_j_tied = neox_args.gpt_j_tied
        self.mlp_type = neox_args.mlp_type
        self.pre_mlp_norm = neox_args.pre_mlp_norm
        self.outer_mlp_norm = neox_args.outer_mlp_norm
        self.safe = neox_args.safe

        self.attention_type = neox_args.attention_config[layer_number]

        norm, eps = get_norm(neox_args)
        self.prenorm, self.postnorm = neox_args.prenorm, neox_args.postnorm
        if self.prenorm:
            self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
            self.post_attention_layernorm = norm(neox_args.hidden_size, eps=eps)

        if self.outer_mlp_norm:
            self.outer_mlp_layernorm = norm(neox_args.hidden_size, eps=eps)

        self.use_cache = use_cache

        self.reduce = mpu.mappings.reduce_from_model_parallel_region

        # Self attention.
        if self.attention_type == 'flash_v2_llama':
            self.attention = LlamaAttention(
                args=neox_args,
            )
        else:
            self.attention = ParallelSelfAttention(
                neox_args=neox_args,
                attention_mask_func=attention_mask_func,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                layer_number=layer_number,
                rpe=rpe,
                use_cache=self.use_cache,
                rotary=rotary,
                parallel_output=self.gpt_j_residual,
            )

        # MLP
        if neox_args.all_config["identity_mlp"]:
            self.mlp = nn.Identity()
        else:
            if neox_args.mlp_type == "regular":
                self.mlp = ParallelMLP(
                    neox_args=neox_args,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                )
            else:
                self.mlp = LLaMAParallelMLP(
                    neox_args=neox_args,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                )

        self.layer_past = None  # used to cache k/v pairs in inference

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = (
                bias_dropout_add_fused_train
                if self.training
                else bias_dropout_add_fused_inference
            )
        else:
            fn = get_bias_dropout_add(self.training)
        return fn

    def forward(self, x, attention_mask, layer_past=None):
        layer_past = layer_past if layer_past is not None else self.layer_past
        bias_dropout_fn = self._get_bias_dropout()

        # _hidden_states = x
        # print_rank_0('in', _hidden_states.shape, _hidden_states.dtype, torch.isnan(_hidden_states).any())

        # x: [b, s, h]
        if self.gpt_j_residual:
            # pseudocode:
            # x = x + attn(ln(x)) + mlp(ln(x))
            # this means we can avoid doing the allreduce in the attn / mlp outputs
            # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
            # due to a bug, the two layernorms are not tied in GPT-NeoX-20B. This is non-desirable, but
            # we preserve the functionality for backwards compatibility

            residual = x
            # applies the correct normalization depending on if the norms are tied
            if self.gpt_j_tied:
                x = self.input_layernorm(x)
                x1, x2 = x, x
            else:
                x1, x2 = self.input_layernorm(x), self.post_attention_layernorm(x)

            # attention operator
            attention_output, attention_bias = self.attention(
                x1, attention_mask, layer_past=layer_past
            )
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents

            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(attention_output) if attention_output is not None else 0,
                    residual=None,
                    prob=self.hidden_dropout,
                )

            # mlp operator
            # check if identity
            if isinstance(self.mlp, nn.Identity):
                output = attention_output
            else:
                mlp_output, mlp_bias = self.mlp(x2)
                with torch.enable_grad():
                    output = bias_dropout_fn(
                        mlp_output,
                        bias=mlp_bias.expand_as(mlp_output) if mlp_output is not None else 0,
                        residual=attention_output,
                        prob=self.hidden_dropout,
                    )

            # output = (x + attn(ln(x)) + mlp(ln(x))
            output = residual + self.reduce(output)

        # main mixer layer logic
        else:
            residual = x

            if self.prenorm:
                if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                    with torch.no_grad():
                        log_norm(
                            x.norm(2, dim=-1).max(1).values.mean(0),
                            f"prenorm_xnorm_{self.layer_number}",
                            self.neox_args.iteration,
                        )

                x = self.input_layernorm(x)

                if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                    with torch.no_grad():
                        log_norm(
                            x.norm(2, dim=-1).max(1).values.mean(0),
                            f"post_prenorm_xnorm_{self.layer_number}",
                            self.neox_args.iteration,
                        )


            # _hidden_states = x
            # print_rank_0('an', _hidden_states.shape, _hidden_states.dtype, torch.isnan(_hidden_states).any())

            attention_output, attention_bias = self.attention(
                x, attention_mask, layer_past=layer_past
            )

            # _hidden_states = attention_output
            # print_rank_0('ao', _hidden_states.shape, _hidden_states.dtype, torch.isnan(_hidden_states).any())


            if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                with torch.no_grad():
                    log_norm(
                        attention_output.norm(2, dim=-1).max(1).values.mean(0),
                        f"post_attn_norm_{self.layer_number}",
                        self.neox_args.iteration,
                    )
                    if attention_bias is not None:
                        log_norm(
                            attention_bias.expand_as(residual)
                            .norm(2, dim=-1)
                            .max(1)
                            .values.mean(0),
                            f"post_attn_residual_bias_borm_{self.layer_number}",
                            self.neox_args.iteration,
                        )

            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents
            
            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(residual) if attention_bias is not None else 0,
                    residual=residual,
                    prob=self.hidden_dropout,
                )

            if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                with torch.no_grad():
                    log_norm(
                        attention_output.norm(2, dim=-1).max(1).values.mean(0),
                        f"post_attn_residual_norm_{self.layer_number}",
                        self.neox_args.iteration,
                    )

            if isinstance(self.mlp, nn.Identity):
                output = attention_output

            # main mlp logic
            else:
                # output = x + mlp(ln2(x))
                # option for pre-mlp norm
                if self.pre_mlp_norm:
                    output = self.post_attention_layernorm(attention_output)
                else:
                    output = attention_output

                # _hidden_states = output
                # print_rank_0('mn', _hidden_states.shape, _hidden_states.dtype, torch.isnan(_hidden_states).any())


                mlp_output, mlp_bias = self.mlp(output)

                # _hidden_states = mlp_output
                # print_rank_0('mo', _hidden_states.shape, _hidden_states.dtype, torch.isnan(_hidden_states).any())

                if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                    with torch.no_grad():
                        log_norm(
                            mlp_output.norm(2, dim=-1).max(1).values.mean(0),
                            f"post_mlp_norm_{self.layer_number}",
                            self.neox_args.iteration,
                        )

                # assert not (self.pre_mlp_norm and self.postnorm), "can't do both pre mlp and post norm!"

                if (
                    self.postnorm  # this is after attention, but before residual
                ):
                    mlp_output = self.post_attention_layernorm(mlp_output)

                if torch.distributed.get_rank() == 0 and self.log_attn_norms:
                    with torch.no_grad():
                        log_norm(
                            mlp_output.norm(2, dim=-1).max(1).values.mean(0),
                            f"post_postnorm_norm_{self.layer_number}",
                            self.neox_args.iteration,
                        )

                if self.mlp_type == "llama" or self.mlp_type == "doublegate_llama":
                    with torch.enable_grad():
                        output = bias_dropout_fn(
                            mlp_output,
                            bias=mlp_bias.expand_as(mlp_output) if mlp_bias is not None else 0,
                            residual=attention_output,
                            prob=self.hidden_dropout,
                        )

                    # after attn AND residual
                    if self.outer_mlp_norm:
                        assert not (
                            self.postnorm and self.outer_mlp_norm
                        ), "can't do both post norm and pre mlp norm!"
                        output = self.outer_mlp_layernorm(output)

                else:
                    with torch.enable_grad():
                        output = bias_dropout_fn(
                            mlp_output,
                            bias=mlp_bias.expand_as(mlp_output) if mlp_bias is not None else 0,
                            residual=attention_output,
                            prob=self.hidden_dropout,
                        )

        return output



class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelTransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        hidden_states.requires_grad_(True)
        # print_rank_0('Outside', hidden_states.float().abs().sum(-1).mean().item())
        return super().forward(hidden_states, attention_mask), attention_mask


class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(
            args, torch.Tensor
        ), "ParallelLinearPipe expects a single argument - hidden_states"
        hidden_state = args
        # hidden_state.requires_grad_(True)
        logits, bias = super().forward(hidden_state)

        return logits


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.norm = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        assert not isinstance(
            args, tuple
        ), "NormPipe should only receive a single tensor as input"
        return self.norm(args)


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel
    return mpu.gather_from_model_parallel_region(logits_parallel)
