import numpy as np
import torch
import copy
from utils.model_utils import add_state_dicts

def aggregate_updates(global_sd, client_updates, client_weights, benign_sets_by_layer, alpha=0.5):
    """
    Blends layer-wise benign aggregation with client-level trust weighting.

    Args:
        global_sd: dict, current global model state_dict
        client_updates: list[dict], updates from all clients
        client_weights: list or np.array, trust weights for each client (0–1)
        benign_sets_by_layer: dict[layer_name] = list[benign_client_ids]
        alpha: float in [0,1], blending factor between layer-wise and trust-based aggregation
    Returns:
        dict: aggregated model update
    """

    # --- Defensive checks ---
    n_clients = len(client_updates)
    if client_weights is None or len(client_weights) != n_clients:
        client_weights = np.ones(n_clients) / n_clients

    # Normalize weights
    client_weights = np.array(client_weights)
    if client_weights.sum() == 0:
        client_weights = np.ones_like(client_weights)
    client_weights = client_weights / client_weights.sum()

    # --- Trust-weighted aggregation ---
    trust_sd = add_state_dicts(client_updates, client_weights)

    # --- Layer-wise benign aggregation ---
    layer_sd = {}
    for k in global_sd.keys():
        benign_ids = benign_sets_by_layer.get(k, [])
        if len(benign_ids) == 0:
            # fallback to all clients
            layer_sd[k] = torch.mean(torch.stack([client_updates[i][k] for i in range(n_clients)]), dim=0)
        else:
            layer_sd[k] = torch.mean(torch.stack([client_updates[i][k] for i in benign_ids]), dim=0)

    # --- Blend the two ---
    blended = {}
    for k in global_sd.keys():
        blended[k] = alpha * layer_sd[k] + (1 - alpha) * trust_sd[k]

    return blended
