import argparse
from models.DeepAR import DeepAR
from lit_models.lit_RNN_based import RNNLitModel

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from load_data.lit_dataloader import RNNDataModule

def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--num_epochs', type=int, default=3)

    p.add_argument("--load_checkpoint", type=str, default=None)

    config = p.parse_args()

    return config

def main(config):
    data = RNNDataModule(config)
    model = DeepAR(config)

    if config.load_checkpoint is not None:
        try:
            checkpoint = torch.load('./saved_models/DeepAr.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            pass
    
    lit_model = RNNLitModel(config=config, model=model)

    trainer = pl.Trainer(max_epochs=config.num_epochs,
                         gpus=config.gpus,
                         #callbacks=[EarlyStopping(monitor='val_loss')]
                         )
    trainer.fit(lit_model, datamodule=data)
    
    # model 저장
    from datetime import datetime
    today = datetime.now()  

    PATH = './saved_models/DeepAr.pt'
    torch.save({'date' : today,
                'epoch': config.num_epochs,
                'model_state_dict': model.state_dict(),
                }, PATH)

    # trainer.test(lit_model, datamodule=data)

if __name__ == '__main__':
    config = define_argparser()
    main(config)