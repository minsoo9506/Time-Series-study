from torch.utils.data import Dataset
import numpy as np

BATCH_SIZE = 7
FEATURE_DIM = 6
TIME_SEQ = 24
TARGET_COL_IDX = 0

def split_multivariate_sequence(data: np.array, time_seq: int=TIME_SEQ, target_col_idx: int=TARGET_COL_IDX):
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
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data.shape[0])
    
    def __getitem__(self,idx):
        X = self.data[idx]
        # |X| = (feature_dim, time_seq)
        y = self.target[idx]
        # |y| = (1,)
        return X, y