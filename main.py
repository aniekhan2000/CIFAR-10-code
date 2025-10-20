"""
FLGuardian Modular — Main Script (Trust-Calibrated Detection)
------------------------------------------------------------
Runs the full federated learning pipeline:
- Data loading
- Local training (with optional malicious clients)
- DP noise addition
- Layer-wise & client-level analysis
- Aggregation and evaluation
- Dynamic trust-based detection
"""

import os, sys, time, copy, numpy as np, torch

# Ensure modules are found when running from Colab
sys.path.append('/content')

from config import CFG, device
from model.smallcnn import SmallCNN
from data.loader import make_loaders
from utils.dp_utils import clip_and_noise
from utils.metrics import evaluate
from utils.ema_tracker import EMA
from analysis.layer_analysis import layer_wise_analysis
from analysis.cross_layer import cross_layer_cka
from analysis.client_analysis import client_level_analysis
from aggregation.aggregator import aggregate_updates


# -------------------- Local Training --------------------
def local_train(model, loader, epochs, lr, device):
    import torch.nn as nn
    model = copy.deepcopy(model)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    before = copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    return {k: model.state_dict()[k] - before[k] for k in before}


# -------------------- Simple Poisoning --------------------
def poison(update, scale=3.0):
    return {k: -scale * v for k, v in update.items()}


# -------------------- Main Federated Run --------------------
def run():
    print("📦 Preparing data loaders...")
    t0 = time.time()
    train_loaders, test_loader = make_loaders(
        CFG.num_clients, CFG.batch_size, CFG.iid, CFG.dirichlet_alpha
    )
    print(f"✅ Data ready ({time.time() - t0:.2f}s). Clients: {len(train_loaders)}")

    # Identify malicious clients
    if hasattr(CFG, "malicious_clients") and CFG.malicious_clients > 0:
        n_mal = min(CFG.malicious_clients, CFG.num_clients)
    else:
        n_mal = int(getattr(CFG, "malicious_frac", 0.0) * CFG.num_clients)

    mal_clients = np.random.choice(CFG.num_clients, n_mal, replace=False)
    print(f"⚠️ Malicious clients: {mal_clients.tolist()}")

    # Initialize global model
    model = SmallCNN().to(device)
    global_sd = copy.deepcopy(model.state_dict())

    # Initial evaluation
    init_acc = evaluate(model, test_loader, device)
    print(f"\nInitial Global Test Accuracy: {init_acc:.4f}")
    ema = EMA()

    # Metrics trackers
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    all_tp, all_fp, all_tn, all_fn = [], [], [], []
    global_acc_per_round = []

    # -------------------- Federated Rounds --------------------
    for rnd in range(1, CFG.rounds + 1):
        rnd_start = time.time()
        client_updates = []

        # ----- Per-client local training -----
        for cid in range(CFG.num_clients):
            base_model = copy.deepcopy(model)
            base_model.load_state_dict(global_sd)
            delta = local_train(base_model, train_loaders[cid], CFG.local_epochs, CFG.lr, device)
            dp_delta = clip_and_noise(delta, CFG.dp_clip_norm, CFG.dp_noise_mult)
            if cid in mal_clients:
                dp_delta = poison(dp_delta)
            client_updates.append(dp_delta)

        # ---- Analysis Phase ----
        benign_sets_by_layer, sim_layer, dist_layer, stab_layer, mean_update = layer_wise_analysis(
            global_sd, client_updates, ema, CFG.seed
        )

        cka_dev = np.zeros(CFG.num_clients)
        labels_c, benign_c, trust = client_level_analysis(
            sim_layer, dist_layer, stab_layer, cka_dev, CFG
        )

        # ---- Trust-based Dynamic Detection ----
        threshold = np.mean(trust) - 0.5 * np.std(trust)
        pred_malicious = np.where(trust < threshold)[0]
        benign_clients = np.where(trust >= threshold)[0]

        print(f"\nDynamic threshold = {threshold:.4f}")
        print(f"Predicted malicious clients: {pred_malicious.tolist()}\n")

        # ---- Round Diagnostics ----
        layer_weights = np.linspace(0.05, 0.5, len(benign_sets_by_layer))
        ground_truth = sorted(list(mal_clients))
        predicted_mal = pred_malicious.tolist()

        TP = len(set(ground_truth) & set(pred_malicious))
        FP = len(set(pred_malicious) - set(ground_truth))
        TN = CFG.num_clients - TP - FP - len(set(ground_truth) - set(pred_malicious))
        FN = len(set(ground_truth) - set(pred_malicious))

        det_acc = (TP + TN) / CFG.num_clients
        det_prec = TP / (TP + FP + 1e-6)
        det_rec = TP / (TP + FN + 1e-6)
        det_f1 = 2 * det_prec * det_rec / (det_prec + det_rec + 1e-6)

        print(f"\n================= Round {rnd} =================")
        print(f"Layer weights: {np.round(layer_weights, 6).tolist()}")
        print(f"Trust scores: {np.round(trust, 6).tolist()}")
        print(f"Selected (benign) clients: {benign_clients.tolist()}")
        print(f"Ground-truth malicious: {ground_truth}")
        print(f"Predicted malicious: {predicted_mal}\n")

        if isinstance(benign_sets_by_layer, dict):
            for i, (pname, clients) in enumerate(benign_sets_by_layer.items()):
                print(f"Param {i} ({pname}) benign clients: {list(clients)}")
        else:
            for i, clients in enumerate(benign_sets_by_layer):
                print(f"Param {i} benign clients: {list(clients)}")

        print(f"\nDetection - Acc={det_acc:.3f}, Prec={det_prec:.3f}, Rec/ADR={det_rec:.3f}, "
              f"F1={det_f1:.3f} (TP={TP}, FP={FP}, TN={TN}, FN={FN})")
        print("-" * 60)

        # ---- Aggregation (trust-weighted) ----
        aggregated = aggregate_updates(
            global_sd=global_sd,
            client_updates=client_updates,
            client_weights=trust,
            benign_sets_by_layer=benign_sets_by_layer,
            alpha=CFG.blend_alpha
        )

        # Update global model
        if isinstance(aggregated, dict):
            for k in global_sd.keys():
                global_sd[k] = global_sd[k] + aggregated.get(k, 0)
            model.load_state_dict(global_sd)

        # ---- Evaluate ----
        acc, prec, rec, f1, tp, fp, tn, fn = evaluate(model, test_loader, device, detailed=True)
        global_acc_per_round.append(acc)
        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)
        all_tp.append(TP)
        all_fp.append(FP)
        all_tn.append(TN)
        all_fn.append(FN)

        print(f"📊 Global Test Acc={acc:.4f} (Elapsed {time.time() - rnd_start:.2f}s)")
        print("----------------------------------------------------------", flush=True)

    # -------------------- Overall Averages --------------------
    avg_acc = np.mean(all_acc)
    avg_prec = np.mean(all_prec)
    avg_rec = np.mean(all_rec)
    avg_f1 = np.mean(all_f1)
    avg_TP = np.mean(all_tp)
    avg_FP = np.mean(all_fp)
    avg_TN = np.mean(all_tn)
    avg_FN = np.mean(all_fn)
    avg_global_acc = np.mean(global_acc_per_round)

    print("\n================= Overall (average over rounds) =================")
    print(f"Accuracy: {avg_acc:.3f}")
    print(f"Precision: {avg_prec:.3f}")
    print(f"Recall / Attack Detection Rate: {avg_rec:.3f}")
    print(f"F1 score: {avg_f1:.3f}")
    print(f"Avg TP: {avg_TP:.2f}, FP: {avg_FP:.2f}, TN: {avg_TN:.2f}, FN: {avg_FN:.2f}")
    print(f"Avg Global Test Acc: {avg_global_acc:.4f}")
    print("===================================================")
    print("\n🎯 Training completed successfully!")


# -------------------- Main Entry --------------------
if __name__ == "__main__":
    run()
