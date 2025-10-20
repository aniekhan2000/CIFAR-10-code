import numpy as np
from utils.feature_utils import linear_cka

def cross_layer_cka(client_updates, mean_update, layer_keys, ema):
    cka_dev = np.zeros(len(client_updates))
    for cid in range(len(client_updates)):
        cka_vals_client, cka_vals_ref = [], []
        for i in range(len(layer_keys)-1):
            a, b = client_updates[cid][layer_keys[i]].reshape(-1), client_updates[cid][layer_keys[i+1]].reshape(-1)
            a_ref, b_ref = mean_update[layer_keys[i]].reshape(-1), mean_update[layer_keys[i+1]].reshape(-1)
            cka_vals_client.append(linear_cka(a,b))
            cka_vals_ref.append(linear_cka(a_ref,b_ref))
        dev = float(np.mean(np.abs(np.array(cka_vals_client) - np.array(cka_vals_ref))))
        cka_dev[cid] = ema.update_cross(cid, dev)
    return cka_dev
