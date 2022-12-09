import torch
import torch.nn.functional as F

from torchmetrics.classification import Accuracy
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from math import floor
from tqdm import tqdm
from loguru import logger
from dataset import NumpyHits
from net import SegmentationNet


#Logging 
logger.add('log/numpy_segment_{time}.log')

# Model params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 100
num_features = 7
batch_size = 32
model = SegmentationNet(num_features, 2, k=20).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

#Dataloader
data_dir = '/eos/user/p/pbarhama/graphnns/saved_pickles/np_arrays/'
train_file = 'train_events_100_1_aronly.npz'
path = data_dir + train_file
train_ds, val_ds = NumpyHits(data_dir=path, data_mode='train'), NumpyHits(data_dir=path, data_mode='val')
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

#Metrics 
train_acc = Accuracy().to(device)
valid_acc = Accuracy().to(device)

def train():
    model.train()

    total_loss = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.x.to(torch.long))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        train_acc.update(out, data.x.to(torch.int))
    total_train_acc = train_acc.compute()
    train_acc.reset()   
    
    return total_loss / len(train_ds), total_train_acc


def validate(loader):
    model.eval()
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
        valid_acc.update(pred, data.x.to(torch.int))
    total_valid_acc = valid_acc.compute()
    valid_acc.reset()
    return total_valid_acc


best_val_acc = 0.0
for epoch in range(1, 20):
    loss, current_train_acc = train()
    current_val_acc = validate(val_loader)
    logger.info(f"\nCurrent val acc:, {current_val_acc:.4f}")  
    logger.info(f"\nCurrent train acc:, {current_train_acc:.4f}")      
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        logger.info(f"\nSaving best model with accurracy  {current_val_acc:.2f} for epoch: {epoch}\n")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'outputs/new_segment_epoch{epoch}_acc_{current_val_acc:.2f}.pth')

    logger.info(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val: {current_val_acc:.4f}')
