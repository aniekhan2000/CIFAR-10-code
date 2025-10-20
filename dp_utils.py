import torch
from .model_utils import flatten_state, unflatten_like

def clip_and_noise(update, clip_norm, noise_mult):
    ref = {k: p.clone().detach() for k, p in update.items()}
    flat = flatten_state(ref)
    norm = flat.norm(2) + 1e-12
    scale = min(1.0, clip_norm / norm.item())
    flat = flat * scale
    if noise_mult > 0.0:
        sigma = noise_mult * clip_norm
        flat += torch.normal(0, sigma, size=flat.shape)
    return unflatten_like(flat, ref)
