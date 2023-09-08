from typing import Any, Dict, Optional, Tuple
import sys
sys.path.append('.')
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.components.L96_4DVarNet_dataset import L96_Dataset


class L96DataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch_lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/train/",
        da_method: str = 'EnKF',
        years: int = 4,
        N_trials: int = 2,
        dim: int = 40,
        da_N: int = 20,
        da_Inf: float = 1.02,
        da_Bx: float = 1.0,
        da_rot: bool = True,
        da_loc_rad: float = 1.0,
        obs_partial: float = 1.0,
        da_Lag: int = 1,
        pred_len: int = 0,
        normalize: bool = False,
        obs_num: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    # @property
    # def num_classes(self):
    #     return 10

    # def prepare_data(self):
    #     """Download data if needed.
    #
    #     Do not use it to assign state (self.x = y).
    #     """
    #     L96_Dataset(self.hparams.data_dir, mode='train')
    #     L96_Dataset(self.hparams.data_dir, mode='val')

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            self.data_train = L96_Dataset(data_dir=self.hparams.data_dir,
                                         da_method=self.hparams.da_method,
                                         years=self.hparams.years,
                                         N_trials=self.hparams.N_trials,
                                         dim=self.hparams.dim,
                                         da_N=self.hparams.da_N,
                                         da_Inf=self.hparams.da_Inf,
                                         da_Bx=self.hparams.da_Bx,
                                         da_rot=self.hparams.da_rot,
                                         da_loc_rad=self.hparams.da_loc_rad,
                                         obs_partial=self.hparams.obs_partial,
                                         da_Lag=self.hparams.da_Lag,
                                         obs_num=self.hparams.obs_num,
                                         pred_len=self.hparams.pred_len,
                                         normalize=self.hparams.normalize,
                                         train=True)

            self.data_val = L96_Dataset(data_dir=self.hparams.data_dir,
                                       da_method=self.hparams.da_method,
                                       years=self.hparams.years,
                                       N_trials=self.hparams.N_trials,
                                       dim=self.hparams.dim,
                                       da_N=self.hparams.da_N,
                                       da_Inf=self.hparams.da_Inf,
                                       da_Bx=self.hparams.da_Bx,
                                       da_rot=self.hparams.da_rot,
                                       da_loc_rad=self.hparams.da_loc_rad,
                                       obs_partial=self.hparams.obs_partial,
                                       da_Lag=self.hparams.da_Lag,
                                       obs_num=self.hparams.obs_num,
                                       pred_len=self.hparams.pred_len,
                                       normalize=self.hparams.normalize,
                                       train=False)
            self.data_test = self.data_val
            # dataset = ConcatDataset(datasets=[trainset, testset])
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=tuple(list(self.hparams.train_val_test_split)*2*trainset.__len__()),
            #     generator=torch.Generator().manual_seed(2023),
            # )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "L96_enkf_N40_partial1.0.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
