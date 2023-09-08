from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression.mse import MeanSquaredError
from src.models.components.lorenz96 import Lorenz96_torch
import src.models.components.torch_4DVarNN_dinAE as NN_4DVar
import src.models.components.solver as solver
from src.models.components.fdvarnet import Encoder, Decoder, Model_AE, Model_L96 
import numpy as np

class FDVarNNLitModule(LightningModule):
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
        optimizer: torch.optim.Optimizer,
        after_scheduler: torch.optim.lr_scheduler,
        scheduler: torch.optim.lr_scheduler,
        alpha: np.ndarray = np.array([1., 0.1]),
        GradType: int = 1,  # Gradient computation (0: subgradient, 1: true gradient/autograd)
        OptimType: int = 2, # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
        IterUpdate: list = [0, 100, 200, 500, 2000, 1000, 1200],  # [0,2,4,6,9,15]
        NbProjection: list = [0, 0, 0, 0, 0, 0, 0],  # [0,0,0,0,0,0]#[5,5,5,5,5]##
        NbGradIter: list = [10, 10, 20, 20, 20, 20, 20],  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        lrUpdate: list = [1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-4, 1e-5, 1e-6, 1e-7],
        lambda_LRAE: float = 0.5,
        model_scheme: str = 'AE',
        shapeData: int = 40
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

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

        NBGradCurrent = NbGradIter[0]
        NBProjCurrent = NbProjection[0]
        self.lrCurrent = lrUpdate[0]
        UsePriodicBoundary = True
        if isinstance(shapeData, int):
            shapeData = [1, shapeData, 1]
        if model_scheme == 'AE':
            model_AE = Model_AE(shapeData)
        elif model_scheme == 'L96':
            model_AE = Model_L96()
        self.alpha = alpha
        self.lambda_LRAE = lambda_LRAE

        self.model = NN_4DVar.Model_4DVarNN_GradFP(model_AE, shapeData, NBProjCurrent, NBGradCurrent, GradType, OptimType, periodicBnd=UsePriodicBoundary)

    def forward(self, xb: torch.Tensor, obs: torch.Tensor):
        return self.net(xb, obs)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_rmse.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        gt, xb, xa, obs, obs_idx, mean, std = batch
        gt = torch.unsqueeze(gt, -1)
        mean = torch.unsqueeze(mean, -1)
        std = torch.unsqueeze(std, -1)
        with torch.set_grad_enabled(True):
            xb = torch.unsqueeze(xb, -1)
            xb = torch.autograd.Variable(xb, requires_grad=True)
            xa = torch.unsqueeze(xa, -1)
            obs = torch.unsqueeze(obs, 1).permute(0, 1, 3, 2)
            in_obs_idx = np.zeros_like(obs.detach().cpu().numpy())
            in_obs_idx[:,:,obs_idx.detach().cpu().numpy(),:] = 1
            in_obs_idx = torch.from_numpy(in_obs_idx).to(obs.device, dtype=torch.float32)
            if self.model.OptimType == 1:
                outputs, grad_new, normgrad = self.model(torch.unsqueeze(xb[:,0,:,:], dim=1), obs, in_obs_idx, None)
            elif self.model.OptimType == 2:
                outputs, hidden_new, cell_new, normgrad = self.model(torch.unsqueeze(xb[:,0,:,:], dim=1), obs, in_obs_idx, None, None)
            else:
                outputs, normgrad = self.model(torch.unsqueeze(xb[:,0,:,:], dim=1), obs, in_obs_idx)

            # loss_R = torch.sum((outputs - torch.unsqueeze(obs[:,:,:,0], dim=1)) ** 2 * in_obs_idx)
            # loss_R = torch.mul(1.0 / torch.sum(obs_idx), loss_R)
            # loss_I = torch.sum((outputs - torch.unsqueeze(xa[:,0,:,:], dim=1)) ** 2 * (1. - in_obs_idx))
            # loss_I = torch.mul(1.0 / torch.sum(1. - in_obs_idx), loss_I)
            loss_All = self.criterion(outputs, torch.unsqueeze(gt[:,0,:,:], dim=1))
            pred = self.model.model_AE(outputs)
            loss_AE = self.criterion(pred, torch.unsqueeze(gt[:,0,:,:], dim=1))
            pred_gt = self.model.model_AE(torch.unsqueeze(gt[:,0,:,:], dim=1))
            for i in range(obs.shape[-1]-1):
                pred_gt = self.model.model_AE(pred_gt)
            # gt_pred = self.l96(xa[:,0,:,0], 0, 0.01)
            # for i in range(obs.shape[-1] -1):
                # gt_pred = self.l96(gt_pred, 0, 0.01)
            # loss_AE_GT = self.criterion(pred_gt, torch.unsqueeze(torch.unsqueeze(gt_pred, dim=1), dim=-1))
            loss_AE_GT = self.criterion(pred_gt, torch.unsqueeze(gt[:,1,:,:], dim=1))

            loss = self.alpha[0] * loss_All + 0.5 * self.alpha[1] * (loss_AE + loss_AE_GT)

            return loss, outputs, torch.unsqueeze(gt[:,0,:,:], dim=1), mean, std

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
        optimizer = self.hparams.optimizer([{'params': self.model.model_Grad.parameters()},
                                            {'params': self.model.model_AE.encoder.parameters(), 'lr': self.lambda_LRAE * self.lrCurrent}
                                            ], lr=self.lrCurrent)
        if self.hparams.scheduler is not None:
            if self.hparams.after_scheduler is not None:
                after_scheduler = self.hparams.after_scheduler(optimizer=optimizer) 
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
    _ = FDVarNNLitModule(None, None, None)
