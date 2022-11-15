import torch
import pickle
import numpy as np

from sklearn.preprocessing import scale
from torch_geometric.data import Data

def load_pickle(filename):
    with open(filename, 'rb') as fp:
        file = pickle.load(fp)
    return file


class HitsDataset(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """
    def __init__(
        self, 
        data_dir='/eos/user/p/pbarhama/graphnns/saved_pickles/',
        signal='gnn_bg_events_250_5',
        background ='gnn_sn_events_250_5',
        hits_per_event=250, 
        scale_data=True,
        shuffle_data=True,
    ): 
        self.data_dir = data_dir
        self.signal = signal 
        self.background = background 
        self.scale_data = scale_data
        self.shuffle_data = shuffle_data
        self.hits_per_event = hits_per_event
        self.data, self.labels = self.preload_data()
        
        
    def preload_data(self):
        bg = load_pickle(self.data_dir + self.background)
        sn = load_pickle(self.data_dir + self.signal)
        
        data = np.array(bg + sn)
        if self.scale_data:
            n_events, n_hits, n_features = data.shape
            data = scale(np.array(data).reshape(self.hits_per_event * data.shape[0], data.shape[2]))
            data = data.reshape(n_events, n_hits, n_features)
        labels = np.append(np.zeros(len(bg)), np.ones(len(sn)))
        return data, labels 
   
    def convert_event(self, event):
        assert event.shape[0] == self.hits_per_event, f"Event should have {self.hits_per_event} but found {event.shape[0]} hits"
        return torch.tensor(event)
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        event = self.data[index]
        label = torch.tensor(self.labels[index]).to(torch.float)
        # Convert to torch tensor and remove IDs
        converted = self.convert_event(event)
        
        # Split positions and features
        op_channel, peak_time, x, y, z, width, area, amplitude, pe, event_label, node_label = torch.transpose(converted, 0, 1)
        pos = torch.stack([peak_time, x, y, z, width, amplitude, pe], dim=1).to(torch.float)
        
        return Data(
            y=label,
            pos=pos.to(torch.float))



class AccumulatedHits(HitsDataset):
    def __getitem__(self, index):
        # Select sample
        event = self.data[index]
        label = torch.tensor(self.labels[index]).to(torch.float)
        # Convert to torch tensor and remove IDs
        converted = self.convert_event(event)
        
        
        # Split positions and features
        op_channel, peak_time, x, y, z, width, area, amplitude, pe, node_label = torch.transpose(converted, 0, 1)
        
        features = torch.stack([width, amplitude, pe], dim=1).to(torch.float)
        pos = torch.stack([peak_time, x, y, z], dim=1).to(torch.float)
        
        return Data(
            x=features.to(torch.float),
            y=label,
            pos=pos.to(torch.float))