import torch
import pickle
import numpy as np

from sklearn.preprocessing import scale
from torch_geometric.data import Data

def load_pickle(filename):
    with open(filename, 'rb') as fp:
        file = pickle.load(fp)
    return file

class TorchStandardScaler:
    def fit(self, x):
        self.mean = torch.mean(x, dim=0, keepdim=True)
        self.std = torch.std(x, dim=0, unbiased=False, keepdim=True)
    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x

def shuffle_tensor_rows(x, return_idx=True):
    dim = 0
    idx = torch.randperm(x.shape[dim])
    x_shuffled = x[idx]
    if return_idx:
        return x_shuffled, idx
    return x_shuffled


class HitsDataset(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(
        self, 
        data_dir='/eos/user/p/pbarhama/graphnns/saved_pickles/',
        signal='gnn_sn_events_100_1_aronly',
        background ='gnn_bg_events_100_1_aronly',
        hits_per_event=100, 
        scale_data=True,
        n_features=9,
        per_event_shuffle=False
    ): 
        self.data_dir = data_dir
        self.signal = signal 
        self.background = background 
        self.scale_data = scale_data
        self.hits_per_event = hits_per_event
        self.data, self.labels, self.node_labels = self.preload_data()
        self.n_features = n_features
        self.per_event_shuffle = per_event_shuffle
        
    def preload_data(self):
        bg = load_pickle(self.data_dir + self.background)
        sn = load_pickle(self.data_dir + self.signal)
        bg_sn = np.array(bg + sn)
        bg_length = len(bg)
        sn_length = len(sn)

        #Free memory
        del bg
        del sn

        #Last column is the per event label - we create this manually
        data, node_labels = bg_sn[:, :, :-2], bg_sn[:, :, -2] 
        # Scale the data
        if self.scale_data:
            n_events, n_hits, n_features = data.shape
            scaler = TorchStandardScaler()
            data = torch.from_numpy(data).reshape(self.hits_per_event * data.shape[0], data.shape[2])
            scaler.fit(data)
            
            print('Fit Scaler, now transforming ...')
            data = scaler.transform(torch.tensor(data))
            data = data.reshape(n_events, n_hits, n_features)
        labels = np.append(np.zeros(bg_length), np.ones(sn_length))
        return data, labels, node_labels

        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        event = self.data[index]
        assert event.shape[0] == self.hits_per_event, f"Event should have {self.hits_per_event} but found {event.shape[0]} hits"

        # Get event label
        label = torch.tensor(self.labels[index]).to(torch.float)

        # Split positions and features
        node_label = torch.tensor(self.node_labels[index]).to(torch.float)
        op_channel, peak_time, x, y, z, width, area, amplitude, pe = torch.transpose(event, 0, 1)
        pos = torch.stack([peak_time, x, y, z, width, amplitude, pe], dim=1).to(torch.float)

        if self.per_event_shuffle:
            pos, idx = shuffle_tensor_rows(pos)
            node_label = node_label[idx]

        return Data(
            x=node_label,
            y=label,
            pos=pos.to(torch.float))

class NumpyHits(HitsDataset):
    def __init__(self, data_mode='train', *args, **kwargs):
        self.data_mode = data_mode
        super().__init__(*args, **kwargs)
        self.data, self.labels, self.node_labels = self.preload_data()

    def preload_data(self):
        array = np.load(self.data_dir)[self.data_mode]
        data, node_labels = array[:, :, :-2], array[:, :, -2]
        labels = array[:, 0, -1]
        return torch.tensor(data), labels, node_labels
    