from typing import Any
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression.mse import MeanSquaredError
from src.models.components.lorenz96 import Lorenz96_torch

class NN4DVarLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch_lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        after_scheduler: torch.optim.lr_scheduler,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.l96 = Lorenz96_torch(8)
        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, xb: torch.Tensor, obs: torch.Tensor):
        return self.net(xb, obs)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_rmse.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        gt, xb, xa, obs, obs_idx, mean_xa, std_xa, mean_xb, std_xb = batch
        pred_xb = torch.ones_like(obs).to(xb.device, dtype=xb.dtype)
        pred_xb[:,0:1,:] = xb[:,0:1,:]*std_xb+mean_xb
        for i in range(obs.shape[1]-1):
            pred_xb[:,i+1:i+2,:] = torch.unsqueeze(self.l96(pred_xb[:,i:i+1,:], 0, 0.01), dim=1)
        # in_obs_idx = np.zeros_like(obs.detach().cpu().numpy())
        # in_obs_idx[:,:,obs_idx.detach().cpu().numpy()] = 1
        # in_obs_idx = torch.from_numpy(in_obs_idx).to(obs.device, dtype=torch.float32)
        # obs = in_obs_idx * obs + (1-in_obs_idx) * pred_xb
        obs = torch.where(obs!=0, obs, (pred_xb-mean_xb)/std_xb)
        pred_xa = self.forward(xb, obs)
        if xa.shape[1] > 1:
            pred_len = xa.shape[1] - 1
            pred = pred_xa
            loss = 0
            for i in range(pred_len):
                for j in range(obs.shape[1]):
                    pred = (torch.unsqueeze(self.l96(pred*std_xa+mean_xa, 0, 0.01), dim=1)-mean_xa)/std_xa
                loss += self.criterion(pred, torch.unsqueeze(xa[:,i+1,:], dim=1))
            loss /= pred_len
        else:
            loss = self.criterion(pred_xa, xa)
        return loss, pred_xa, gt, mean_xa, std_xa

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, mean, std = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_rmse(preds*std+mean, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, mean, std = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_rmse(preds*std+mean, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, mean, std = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_rmse(preds*std+mean, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            if self.hparams.after_scheduler is not None:
                after_scheduler = self.hparams.after_scheduler(optimizer=optimizer)  # ,
                # eta_min=1e-3*optimizer.state_dict()['param_groups'][0]['lr'])
                scheduler = self.hparams.scheduler(optimizer=optimizer, after_scheduler=after_scheduler)
            else:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = NN4DVarLitModule(None, None, None)
