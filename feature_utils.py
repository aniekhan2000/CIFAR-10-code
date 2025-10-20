import torch, torch.nn.functional as F, numpy as np

def cosine_sim(a,b): return F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
def l2_dist(a,b): return (a.reshape(-1)-b.reshape(-1)).norm(2).item()

def mm(x):
    m, M = x.min(), x.max()
    return (x - m) / (1e-12 + (M - m))

def linear_cka(x,y):
    x = x.reshape(1,-1) if x.ndim==1 else x
    y = y.reshape(1,-1) if y.ndim==1 else y
    x, y = x - x.mean(0, keepdim=True), y - y.mean(0, keepdim=True)
    xx, yy, xy = (x.t()@x), (y.t()@y), (x.t()@y)
    hsic_xy = (xy**2).sum()
    hsic_xx = (xx**2).sum()+1e-12
    hsic_yy = (yy**2).sum()+1e-12
    return (hsic_xy/torch.sqrt(hsic_xx*hsic_yy)).item()
