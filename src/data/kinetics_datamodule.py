from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import ipdb
# from kinetics_dataset import KineticsDataset
from src.data.kinetics_dataset import KineticsDataset
#from src.data.kinetics_vmae import VideoClsDataset

class KineticsDataModule(LightningDataModule):
    """Lightning data module storing the preprocessing informatino for the kinetics dataset.

    This contanis the cropping / etc. Transforms needed on the set.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        train_metadata_path: str = None,
        val_metadata_path: str = None,
        test_metadata_path: str = None,
    ) -> None:
        """Initialize a `KineticsDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # Data transformations. Defines the crops / segments
        # used at training + eval.

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        # Add this to a config; 
        self.metadata_train = train_metadata_path
        self.metadata_val = val_metadata_path
        self.metadata_test = test_metadata_path

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of Kinetics-400 classes (400).
        """
        return 400

    def prepare_data(self) -> None:
        """Download data if needed.

        Lightning ensures that `self.prepare_data()` is called only within a single process on CPU,
        so you can safely add your downloading logic within. In case of multi-node training, the
        execution of this hook depends upon `self.prepare_data_per_node()`.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if self.metadata_train is not None:
            self.data_train = KineticsDataset(
                metadata=self.metadata_train,
                mode='train'
            )

        if self.metadata_val is not None:
            self.data_val = KineticsDataset(
                metadata=self.metadata_val,
                mode='val'
            )
            self.data_val.num_clips = 1

        if self.metadata_test is not None:
            self.data_test = KineticsDataset(
                metadata=self.metadata_test,
                mode='test'
            )


    def dynamic_pad_collate_fn(self, batch):
        return torch.utils.data.dataloader.default_collate(batch)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        print("Creating val dataloader")
        #ipdb.set_trace()
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        print("Creating test dataloader")
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.dynamic_pad_collate_fn,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = KineticsDataModule()
