import argparse

import torch
import pytorch_lightning as pl
from models.DilatedCNN import CNNForecasting
from load_data.lit_dataloader import CNNDataModule
from lit_models.lit_CNN_based import CNNLitModel

def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--num_epochs', type=int, default=5)

    p.add_argument('--in_channels', type=int, nargs='*', default=[6, 8, 16, 8])
    p.add_argument('--out_channels', type=int, nargs='*', default=[8, 16, 8, 4])
    p.add_argument('--kernel_size', type=int,  nargs='*', default=[3, 3, 3, 4])
    p.add_argument('--num_layers', type=int, default=4)

    p.add_argument("--load_checkpoint", type=str, default=None)

    config = p.parse_args()

    return config

def main(config):
    data = CNNDataModule(config)
    model = CNNForecasting(config)

    if config.load_checkpoint is not None:
        checkpoint = torch.load('./logs/CNN.pt')
        model.load_state_dict(checkpoint['model'])
    
    lit_model = CNNLitModel(config=config, model=model)

    trainer = pl.Trainer(max_epochs=config.num_epochs, gpus=config.gpus)
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


if __name__ == '__main__':
    config = define_argparser()
    main(config)