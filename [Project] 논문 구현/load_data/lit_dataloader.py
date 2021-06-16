import pandas as pd
import numpy as np

from load_data.dataset import split_multivariate_sequence, MultivariateCnnDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

BATCH_SIZE = 128
NUM_WORKERS = -1


class BaseDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = vars(config) if config is not None else {}
        self.batch_size = self.config.get("batch_size", BATCH_SIZE)
        self.num_workers = self.config.get("num_workers", NUM_WORKERS)

        self.on_gpu = isinstance(self.config.get("gpus", None), (str, int))

    def prepare_data(self, *args, **kwargs):
        train = pd.read_csv('./data/train.csv', encoding='euc-kr')
        data = train.loc[train['num']==1,:]
        del_cols = ['num','date_time', '비전기냉방설비운영', '태양광보유']
        data.drop(del_cols, axis=1, inplace=True)
        col_mean = np.mean(data, axis=0)
        col_std = np.std(data, axis=0)
        col_mean = np.mean(data, axis=0)
        col_std = np.std(data, axis=0)
        scaled_data = (data - col_mean) / col_std
        X, y = split_multivariate_sequence(data=np.array(scaled_data), time_seq=24, target_col_idx=0)
        nums_train_idx = int(X.shape[0] * 0.8)
        num_val_idx = int(X.shape[0] * 0.9)
        self.train_X, self.train_y = X[:nums_train_idx], y[:nums_train_idx]
        self.valid_X, self.valid_y = X[nums_train_idx:num_val_idx], y[nums_train_idx:num_val_idx]
        self.test_X, self.test_y = X[num_val_idx:], y[num_val_idx:]

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            self.dataset_train = MultivariateCnnDataset(self.train_X, self.train_y)
            self.dataset_val = MultivariateCnnDataset(self.valid_X, self.valid_y)
        if stage == 'test' or stage is None:
            self.dataset_test = MultivariateCnnDataset(self.test_X, self.test_y)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )