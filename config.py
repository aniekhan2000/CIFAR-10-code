from dataclasses import dataclass
import torch, random, numpy as np

@dataclass
class CFG:
    seed: int = 1337
    device: str = "cpu"

    # Federated setup
    num_clients: int = 20
    malicious_clients: int = 5
    rounds: int = 50
    local_epochs: int = 4
    batch_size: int = 64
    lr: float = 0.01

    # DP
    dp_clip_norm: float = 1.0
    dp_noise_mult: float = 0.0

    # Aggregation
    top_k: int = 5
    blend_alpha: float = 0.5
    temperature: float = 1.0

    # Verbosity
    verbose: bool = True   # ✅ lowercase, proper dataclass field

    # Trust weights
    w_sim: float = 0.6 # 0.3
    w_stab: float = 0.1 # 0.4
    w_neg_dist: float = 0.2
    w_neg_cka_dev: float = 0.1
    trust_threshold: float = 0.45

    # Data
    iid: bool = True
    dirichlet_alpha: float = 0.9

CFG = CFG()

# Seeding
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if torch.cuda.is_available() and CFG.device == "cuda":
    torch.cuda.manual_seed_all(CFG.seed)

device = torch.device(CFG.device)
