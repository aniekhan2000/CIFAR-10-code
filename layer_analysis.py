import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from utils.feature_utils import cosine_sim, l2_dist

def kmeans_2(X, seed):
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=seed)
    return kmeans.fit_predict(X), kmeans.cluster_centers_

def choose_benign_cluster(centers, idx_sim, idx_dist, idx_stab, idx_ckadev=None):
    scores = []
    for c in centers:
        score = c[idx_sim] + c[idx_stab] + (1 - c[idx_dist])
        if idx_ckadev is not None: score += (1 - c[idx_ckadev])
        scores.append(score)
    return int(np.argmax(scores))

def layer_wise_analysis(global_sd, client_updates, ema, seed):
    layer_keys = list(global_sd.keys())
    benign_sets_by_layer = {k: [] for k in layer_keys}
    sim_layer, dist_layer, stab_layer = {}, {}, {}
    num_clients = len(client_updates)

    for cid in range(num_clients):
        sim_layer[cid], dist_layer[cid], stab_layer[cid] = [], [], []

    # compute mean update
    mean_update = {k: sum(u[k] for u in client_updates)/len(client_updates) for k in layer_keys}

    for k in layer_keys:
        feats = []
        for cid in range(num_clients):
            u, ref = client_updates[cid][k].detach(), mean_update[k].detach()
            sim = cosine_sim(u, ref)
            dist = l2_dist(u, ref)
            ema_v = ema.update_layer(cid, k, u)
            stab = 1.0 / (1e-6 + (u.reshape(-1)-ema_v.reshape(-1)).norm(2).item())
            feats.append([sim, dist, stab])
            sim_layer[cid].append(sim); dist_layer[cid].append(dist); stab_layer[cid].append(stab)

        Xn = MinMaxScaler().fit_transform(feats)
        labels, centers = kmeans_2(Xn, seed)
        benign_cluster = choose_benign_cluster(centers, 0, 1, 2)
        for cid in range(num_clients):
            if labels[cid] == benign_cluster:
                benign_sets_by_layer[k].append(cid)

    return benign_sets_by_layer, sim_layer, dist_layer, stab_layer, mean_update
