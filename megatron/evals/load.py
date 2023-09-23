"""
Utilities to load a checkpoint from safari-neox and run single node eval or finetuning.
"""

import os
import json

import torch


from megatron.model.gpt2_model import GPT2ModelPipe
from megatron.neox_arguments.neox_args import *
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron.tokenizer import build_tokenizer
import deepspeed
from transformers import AutoTokenizer

import yaml


def load_checkpoint(model_cfg, to_sequential=False):
    args = NeoXArgs(**model_cfg)

    print("Instantiating tokenizer...")
    tokenizer = build_tokenizer(args)

    assert args.pipe_parallel_size > 0, "pipeline parallelism must be set (> 0)"
    world_size = torch.cuda.device_count()
    print("World size: ", world_size)
    os.environ["RANK"] = "0"

    # [MP 170723] Make sure no other MPI processes are running on the same node
    # *** An error occurred in MPI_Init_thread
    # *** on a NULL communicator
    # *** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
    # ***    and potentially your MPI job)
    # Local abort before MPI_INIT completed completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
    os.environ["WORLD_SIZE"] = str(world_size)

    # deepspeed.init_distributed(dist_backend="nccl")
    args.rank = 0
    initialize_megatron(args)

    model = GPT2ModelPipe(args, num_tokentypes=0, parallel_output=True)
    if to_sequential:
        model = model.to_sequential()

    ckpt_path = args.load
    iteration = args.iteration
    print("Loading checkpoint from: ", ckpt_path)
    for layer_idx in range(len(model.sequential)):
        print("Loading layer:", layer_idx)
        try:
            ckpt = torch.load(
                f"{ckpt_path}/global_step{iteration}/layer_{layer_idx:02d}-model_00-model_states.pt"
            )
            model.sequential[layer_idx].load_state_dict(ckpt)
            print("Loaded state_dict for layer:", layer_idx)
        except:
            pass

    return model, tokenizer


if __name__ == "__main__":
    print("Loading checkpoint")

    with open("./configs/hyena/reference/evals/test_load.yml", "r") as file:
        # Load the YAML content
        test_cfg = yaml.load(file, Loader=yaml.FullLoader)

    print(test_cfg)

    model, tokenizer = load_checkpoint(test_cfg, to_sequential=True)
    print(model)
    print(tokenizer)
    print("Done")
