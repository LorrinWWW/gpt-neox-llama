import os
import torch
import transformers
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer


def write_ckpt(outpath: Path, model: torch.nn.Module, model_config: transformers.AutoConfig, mp: int, dtype=torch.bfloat16):
    model = model.to(dtype)
    loaded = model.state_dict()
    n_layers = model_config.num_hidden_layers
    if mp == 1:
        # embedding
        sd = {"word_embeddings.weight": loaded['model.embed_tokens.weight']}
        torch.save(sd, os.path.join(outpath,  "layer_00-model_00-model_states.pt"))
        # norm
        sd = {"norm.weight": loaded['model.norm.weight']}
        torch.save(sd, os.path.join(outpath, f"layer_{n_layers+3:02d}-model_00-model_states.pt"))
        # lm head
        sd = {"final_linear.weight": loaded['lm_head.weight']}
        torch.save(sd, os.path.join(outpath, f"layer_{n_layers+4:02d}-model_00-model_states.pt"))
        # decoder layers
        for layer_i in range(n_layers):
            sd = {
                nm.replace(
                    f"model.layers.{layer_i}.", f""
                ).replace(
                    "self_attn", "attention"
                ): weight for nm, weight in loaded.items() if nm.startswith(f"model.layers.{layer_i}.")}
            torch.save(sd, os.path.join(outpath, f"layer_{layer_i+2:02d}-model_00-model_states.pt"))
    else:
        # embedding
        for i_mp in range(mp):
            vocab_size = loaded['model.embed_tokens.weight'].size(0) // mp
            sd = {"word_embeddings.weight": loaded['model.embed_tokens.weight'][i_mp*vocab_size: (i_mp+1)*vocab_size].clone()}
            torch.save(sd, os.path.join(outpath,  f"layer_00-model_{i_mp:02d}-model_states.pt"))

            sd = {"norm.weight": loaded['model.norm.weight']}
            torch.save(sd, os.path.join(outpath, f"layer_{n_layers+3:02d}-model_{i_mp:02d}-model_states.pt"))

            assert loaded['lm_head.weight'].size(0)  // mp == vocab_size
            sd = {"final_linear.weight": loaded['lm_head.weight'][i_mp*vocab_size: (i_mp+1)*vocab_size].clone()}
            torch.save(sd, os.path.join(outpath,  f"layer_{n_layers+4:02d}-model_{i_mp:02d}-model_states.pt"))

            for layer_i in range(n_layers):

                original_sd = {nm.replace(f"model.layers.{layer_i}.", f""): weight for nm, weight in loaded.items() if nm.startswith(f"model.layers.{layer_i}.")}
                sd = {
                    nm.replace(
                        f"model.layers.{layer_i}.", f""
                    ).replace(
                        "self_attn", "attention"
                    ): weight for nm, weight in loaded.items() if nm.startswith(f"model.layers.{layer_i}.")}
                
                for n, p in sd.items():
                    if 'gate_proj' in n or 'up_proj' in n \
                      or 'q_proj' in n or 'k_proj' in n or 'v_proj' in n:
                        dim = p.size(0) // mp
                        sd[n] = p[i_mp*dim: (i_mp+1)*dim].clone()
                    elif 'down_proj' in n or 'o_proj' in n:
                        dim = p.size(1) // mp
                        sd[n] = p[:, i_mp*dim: (i_mp+1)*dim].clone()
                        
                torch.save(sd, os.path.join(outpath, f"layer_{layer_i+2:02d}-model_{i_mp:02d}-model_states.pt"))
    
    model_state = {
        "dp_world_size": 1,
        "mp_world_size": mp,
        "module": None,
        "optimizer": None,
        "global_steps": 1,
        "skipped_steps": 1,
        "iteration": 1,
    }
    
    for rank in range(mp):
        torch.save(model_state, os.path.join(outpath, f"mp_rank_{rank:02d}_model_states.pt"))


def from_hf(model_name_or_path: str, outdir: str, mp_size:int, dtype=torch.bfloat16):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    model_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    torch.nn.Linear.reset_parameters = lambda x: None
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
    outpath = Path(outdir)
    if outpath.exists():
        print(f"Output directory {outpath} already exists. Exiting.")
        exit(0)
    print(f"Writing to {outpath}")
    outpath.mkdir()
    with open(os.path.join(outpath, "latest"), "w") as fout:
        fout.write("global_step001")
    steppath = os.path.join(outpath, "global_step001")
    os.mkdir(steppath)
    write_ckpt(steppath, model, model_config, mp_size, dtype=dtype)
    tokenizer.save_pretrained(outpath)
    model_config.save_pretrained(outpath)

if __name__ == '__main__':
    from_hf("/home/jue/v3/llama-2-70b", "/home/jue/v3/llama-2-70b-mp4", 4, torch.bfloat16)