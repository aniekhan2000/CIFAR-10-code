
 
import torch

# -------------------- Flatten & Unflatten --------------------
def flatten_state(sd):
    return torch.cat([p.reshape(-1) for p in sd.values()])

def unflatten_like(vec, ref):
    out = {}
    idx = 0
    for k, p in ref.items():
        n = p.numel()
        out[k] = vec[idx:idx+n].view_as(p)
        idx += n
    return out

# -------------------- State Dict Math --------------------
def state_dict_diff(sd_new, sd_old):
    return {k: sd_new[k] - sd_old[k] for k in sd_new}

def add_state_dicts(sds, weights):
    out = {k: torch.zeros_like(next(iter(sds)).get(k)) for k in sds[0].keys()}
    for sd, w in zip(sds, weights):
        for k in out:
            out[k] += sd[k] * w
    return out

# -------------------- EMA (Exponential Moving Average) --------------------
def ema_update(global_model, new_model, alpha=0.9):
    """
    Smoothly updates the global model weights using EMA:
      global = alpha * global + (1 - alpha) * new
    """
    for old_param, new_param in zip(global_model.parameters(), new_model.parameters()):
        old_param.data = alpha * old_param.data + (1 - alpha) * new_param.data
