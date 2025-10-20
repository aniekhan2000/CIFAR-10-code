import numpy as np
from sklearn.cluster import KMeans
from utils.feature_utils import mm
from analysis.layer_analysis import choose_benign_cluster

def client_level_analysis(sim_layer, dist_layer, stab_layer, cka_dev, CFG):
    num_clients = len(sim_layer)
    sim_avg = np.array([np.mean(sim_layer[c]) for c in range(num_clients)])
    dist_avg = np.array([np.mean(dist_layer[c]) for c in range(num_clients)])
    stab_avg = np.array([np.mean(stab_layer[c]) for c in range(num_clients)])

    # 🛠 Handle None or invalid cka_dev safely
    if cka_dev is None:
        cka_dev = np.zeros(num_clients)
    else:
        cka_dev = np.array(cka_dev)
        # handle dict input (layer-wise) by averaging
        if isinstance(cka_dev, dict):
            try:
                cka_dev = np.mean(list(cka_dev.values()), axis=0)
            except Exception:
                cka_dev = np.zeros(num_clients)

    # Normalize all metrics
    sim_n = mm(sim_avg)
    dist_n = mm(dist_avg)
    stab_n = mm(stab_avg)
    cka_n = mm(cka_dev)

    # Feature matrix for clients
    X_client = np.vstack([sim_n, dist_n, stab_n, cka_n]).T

    # KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=CFG.seed)
    labels_c = kmeans.fit_predict(X_client)
    centers_c = kmeans.cluster_centers_

    # Identify benign cluster (based on weights)
    benign_c = choose_benign_cluster(centers_c, 0, 1, 2, 3)

    # Compute trust score
    trust = (
        CFG.w_sim * sim_n
        + CFG.w_stab * stab_n
        - CFG.w_neg_dist * dist_n
        - CFG.w_neg_cka_dev * cka_n
    )
    trust = np.clip(trust, 0.0, 1.0)
    # 🧩 Debugging: Show per-client Trust Breakdown
    print("\n🔍 Per-client Trust Breakdown:")
    for i in range(len(trust)):
     print(f"Client {i}:")
     print(f"  - sim_n      = {sim_n[i]:.4f}")
     print(f"  - stab_n     = {stab_n[i]:.4f}")
     print(f"  - dist_n     = {dist_n[i]:.4f}")
     print(f"  - cka_n      = {cka_n[i]:.4f}")
     print(f"  => Final trust = {trust[i]:.4f}")
     print("--------------------------------------------------\n")



    return labels_c, benign_c, trust
