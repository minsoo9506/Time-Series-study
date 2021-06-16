import torch
import torch.optim
import pytorch_lightning as pl

OPTIMIZER = 'Adam'
LR = 1e-3
LOSS = 'MSELoss'

class CNNLitModel(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = vars(config) if config is not None else {}

        optimizer = self.config.get('optimizer', OPTIMIZER)
        # if `optimizer` is in the self.config return it
        # else return OPTIMIZER
        self.optimizer = getattr(torch.optim, optimizer)
        # return torch.optim.optimizer

        self.lr = self.config.get("lr", LR)

        loss = self.config.get('loss', LOSS)
        self.loss_fn = getattr(torch.nn, loss)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), self.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        return loss