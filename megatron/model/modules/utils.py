from torch.nn import functional


# @torch.jit.script
def gpt_loss_func(input, target):
    lm_logits, labels = input, target
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    return loss
