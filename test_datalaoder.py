import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from math import floor
from tqdm import tqdm
from dataset import HitsDataset

batch_size = 16
ds = HitsDataset()
train_len = floor(0.8 * len(ds))
test_len = len(ds) - train_len
generator=torch.Generator().manual_seed(42)
train_ds, test_ds = random_split(ds, lengths=[train_len, test_len], generator=generator)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

batch = next(iter(test_loader))
print(batch)