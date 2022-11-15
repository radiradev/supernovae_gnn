import torch
import torch.nn.functional as F

from torchmetrics.classification import Accuracy
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from math import floor
from tqdm import tqdm
from loguru import logger
from dataset import HitsDataset
from net import Net


#Logging 
logger.add('log/file_{time}.log')

# Model params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
num_features = 7
batch_size = 32
model = Net(num_features, num_classes, k=20).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

#Dataloader
ds = HitsDataset()
train_len = floor(0.8 * len(ds))
test_len = len(ds) - train_len
generator=torch.Generator().manual_seed(42)
train_ds, test_ds = random_split(ds, lengths=[train_len, test_len], generator=generator)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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
        loss = F.nll_loss(out, data.y.to(torch.long))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        train_acc.update(out, data.y.to(torch.int))
    total_train_acc = train_acc.compute()
    train_acc.reset()   
    
    return total_loss / len(train_ds), total_train_acc


def test(loader):
    model.eval()
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
        valid_acc.update(pred, data.y.to(torch.int))
    total_valid_acc = valid_acc.compute()
    valid_acc.reset()
    return total_valid_acc


best_test_acc = 0.0
for epoch in range(1, 1000):
    loss, current_train_acc = train()
    current_test_acc = test(test_loader)
    logger.info(f"\nCurrent test acc:, {current_test_acc:.4f}")  
    logger.info(f"\nCurrent train acc:, {current_train_acc:.4f}")      
    if current_test_acc > best_test_acc:
        best_test_acc = current_test_acc
        logger.info(f"\nSaving best model with accurracy  {current_test_acc:.2f} for epoch: {epoch}\n")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'outputs/model_epoch{epoch}_acc_{current_test_acc:.2f}.pth')

    logger.info(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {current_test_acc:.4f}')