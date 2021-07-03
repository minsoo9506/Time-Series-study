import torch
from torch.utils.data import Dataset
import numpy as np

FEATURE_DIM = 6
TIME_SEQ = 24
TARGET_COL_IDX = 0

def split_multivariate_sequence(
    data: np.array,
    time_seq: int=TIME_SEQ,
    target_col_idx: int=TARGET_COL_IDX):
    """
    data : np.array format data
    time_seq : time sequence length of data
    target_col_idx : target's columns index(int) in data 
    """
    x, y = data[0:time_seq, :].reshape(1,-1,time_seq), data[0 + time_seq, target_col_idx]
    for start_idx in range(1, len(data)):
        end_idx = start_idx + time_seq - 1

        if end_idx > len(data)-2:
            break
        seq_x, seq_y = data[start_idx:end_idx+1, :].reshape(1,-1,time_seq), data[end_idx+1, target_col_idx]
        x = np.vstack([x, seq_x])
        y = np.vstack([y, seq_y]) 
    return x, y

class MultivariateCnnDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,idx):
        X = self.data[idx]
        # |X| = (feature_dim, time_seq)
        y = self.target[idx]
        # |y| = (1,)
        return X, y

def split_multivariate_sequence_rnn(
    data: np.array,
    seq_len: int=TIME_SEQ,
    target_col_idx: int=TARGET_COL_IDX,
    stage: str='train',
    train_ratio: float=0.8
    ):
    """
    data : np.array format data
    seq_len : time sequence length of data
    target_col_idx : target's columns index(int) in data 
    is_train : True if it is the training dataset
    """
    # |data for RNN| = (batch_size, seq_len, input_dim)
    # initial
    enc_x = data[0 : seq_len, :].reshape(1,seq_len,-1)
    dec_emb = data[seq_len - 1 : 2 * seq_len - 1, target_col_idx+1:].reshape(1,seq_len,-1)
    dec_teacher = data[seq_len - 1 : 2 * seq_len - 1, target_col_idx].reshape(1,seq_len,1)
    y = data[seq_len : 2 * seq_len, target_col_idx]
    
    for start_idx in range(1, len(data)):
        end_idx = start_idx + seq_len - 1

        if end_idx > len(data) - 1 - seq_len:
            break
        
        seq_enc_x = data[start_idx : end_idx + 1, :].reshape(1,seq_len,-1)
        seq_dec_emb = data[end_idx : end_idx + seq_len, target_col_idx+1:].reshape(1,seq_len,-1)
        seq_y = data[end_idx + 1 : end_idx + 1 + seq_len, target_col_idx]
        
        enc_x = np.vstack([enc_x, seq_enc_x])
        dec_emb = np.vstack([dec_emb, seq_dec_emb])
        y = np.vstack([y, seq_y])

        if stage == 'train' or 'validate':
            seq_dec_teacher = data[end_idx : end_idx + seq_len, target_col_idx].reshape(1,seq_len,1)
            dec_teacher = np.vstack([dec_teacher, seq_dec_teacher])

    # |enc_x| = (total, seq_len, input_dim)
    # |dec_emb| = (total, seq_len, input_dim-1)
    # |dec_teacher| = (total, seq_len, 1)

    last_train_idx = int(data.shape[0] * 0.8)

    if stage == 'train':
        return np.concatenate((enc_x[:last_train_idx], dec_emb[:last_train_idx], dec_teacher[:last_train_idx]), 2), y[:last_train_idx]
    elif stage == 'validate':
        return np.concatenate((enc_x[last_train_idx:], dec_emb[last_train_idx:], dec_teacher[last_train_idx:]), 2), y[last_train_idx:]
    else:
        return np.concatenate((enc_x[last_train_idx:], dec_emb[last_train_idx:]), 2), y[last_train_idx:]

class MultivariateRnnTrainDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        data = torch.tensor(data, dtype=torch.float32)
        self.enc_x = data[:, :, 0:6]
        self.dec_emb = data[:, :, 6:11]
        self.dec_teacher = torch.unsqueeze(data[:, :, 11], 2)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.data_len = int(data.shape[0] / 3)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self,idx):
        X = self.enc_x[idx], self.dec_emb[idx], self.dec_teacher[idx]
        y = self.target[idx]
        # |y| = (time_seq_len,)
        return X, y

class MultivariateRnnInferenceDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        data = torch.tensor(data, dtype=torch.float32)
        self.enc_x = data[:, :, 0:6]
        self.dec_emb = data[:, :, 6:]
        self.target = torch.tensor(target, dtype=torch.float32)
        self.data_len = int(data.shape[0] / 3)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self,idx):
        X = self.enc_x[idx], self.dec_emb[idx]
        y = self.target[idx]
        # |y| = (time_seq_len,)
        return X, y