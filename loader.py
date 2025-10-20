import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def make_loaders(num_clients, batch_size, iid=True, dirichlet_alpha=0.5):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    test = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)

    idxs = np.arange(len(train))
    if iid:
        np.random.shuffle(idxs)
        parts = np.array_split(idxs, num_clients)
    else:
        labels = np.array(train.targets)
        parts = [[] for _ in range(num_clients)]
        for c in range(10):
            c_idx = np.where(labels == c)[0]
            np.random.shuffle(c_idx)
            proportions = np.random.dirichlet([dirichlet_alpha]*num_clients)
            proportions = (np.cumsum(proportions)*len(c_idx)).astype(int)
            splits = np.split(c_idx, proportions[:-1])
            for i,s in enumerate(splits):
                parts[i].extend(s.tolist())
        parts = [np.array(p) for p in parts]

    train_loaders = [DataLoader(Subset(train,p), batch_size=batch_size, shuffle=True) for p in parts]
    test_loader = DataLoader(test, batch_size=256, shuffle=False)
    return train_loaders, test_loader
