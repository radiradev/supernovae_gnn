import numpy as np
import pickle


data_dir='/eos/user/p/pbarhama/graphnns/saved_pickles/'
signal='gnn_sn_events_100_1_aronly'
background ='gnn_bg_events_100_1_aronly'

def load_data(data_dir, signal, background):
    with open(data_dir + background, 'rb') as fp:
        bg = pickle.load(fp)
    with open(data_dir + signal, 'rb') as fp:
        sn = pickle.load(fp)
    bg_sn = np.array(bg + sn)
    return bg_sn

def permute_rows(x):
    idx = np.random.permutation(x.shape[0])
    return x[idx]

def split_data(data, train_frac=0.8, val_frac=0.1):
    "Splits data into train, validation, and test sets."
    train_len = int(train_frac * len(data))
    val_len = int(val_frac * len(data))
    train = data[:train_len]
    val = data[train_len:train_len + val_len]
    test = data[train_len + val_len:]
    return train, val, test

def save_data(data_dir, train, val, test):
    np.savez(data_dir + 'train_events_100_1_aronly.npz', train=train, val=val)
    np.savez(data_dir + 'test_events_100_1_aronly.npz', test=test)

bg_sn = load_data(data_dir, signal, background)
bg_sn = permute_rows(bg_sn)
train, val, test = split_data(bg_sn)
save_data(data_dir, train, val, test)
#load the arrays from the saved files
train = np.load(data_dir + 'train_events_100_1_aronly.npz')['train']
val = np.load(data_dir + 'train_events_100_1_aronly.npz')['val']

print(train.shape, val.shape)