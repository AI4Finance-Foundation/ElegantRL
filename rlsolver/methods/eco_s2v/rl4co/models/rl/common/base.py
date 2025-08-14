import abc
import time
from functools import partial
from typing import Any, Iterable

import torch
import torch.nn as nn
from lightning import LightningModule
from torch.utils.data import DataLoader

from rlsolver.methods.eco_s2v.rl4co.data.generate_data import generate_default_datasets
from rlsolver.methods.eco_s2v.rl4co.envs.common.base import RL4COEnvBase
from rlsolver.methods.eco_s2v.rl4co.utils.optim_helpers import create_optimizer, create_scheduler
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class RL4COLitModule(LightningModule, metaclass=abc.ABCMeta):
    """Base class for Lightning modules for RL4CO. This defines the general training loop in terms of
    RL algorithms. Subclasses should implement mainly the `shared_step` to define the specific
    loss functions and optimization routines.

    Args:
        env: RL4CO environment
        policy: policy network (actor)
        batch_size: batch size (general one, default used for training)
        val_batch_size: specific batch size for validation. If None, will use `batch_size`. If list, will use one for each dataset
        test_batch_size: specific batch size for testing. If None, will use `val_batch_size`. If list, will use one for each dataset
        train_data_size: size of training dataset for one epoch
        val_data_size: size of validation dataset for one epoch
        test_data_size: size of testing dataset for one epoch
        optimizer: optimizer or optimizer name
        optimizer_kwargs: optimizer kwargs
        lr_scheduler: learning rate scheduler or learning rate scheduler name
        lr_scheduler_kwargs: learning rate scheduler kwargs
        lr_scheduler_interval: learning rate scheduler interval
        lr_scheduler_monitor: learning rate scheduler monitor
        generate_default_data: whether to generate default datasets, filling up the data directory
        shuffle_train_dataloader: whether to shuffle training dataloader. Default is False since we recreate dataset every epoch
        dataloader_num_workers: number of workers for dataloader
        data_dir: data directory
        metrics: metrics
        litmodule_kwargs: kwargs for `LightningModule`
    """

    def __init__(
            self,
            env: RL4COEnvBase,
            policy: nn.Module,
            batch_size: int = 512,
            val_batch_size: list[int] | int = None,
            test_batch_size: list[int] | int = None,
            train_data_size: int = 100_000,
            val_data_size: int = 10_000,
            test_data_size: int = 10_000,
            optimizer: str | torch.optim.Optimizer | partial = "Adam",
            optimizer_kwargs: dict = {"lr": 1e-4},
            lr_scheduler: str | torch.optim.lr_scheduler.LRScheduler | partial = None,
            lr_scheduler_kwargs: dict = {
                "milestones": [80, 95],
                "gamma": 0.1,
            },
            lr_scheduler_interval: str = "epoch",
            lr_scheduler_monitor: str = "val/reward",
            generate_default_data: bool = False,
            shuffle_train_dataloader: bool = False,
            dataloader_num_workers: int = 0,
            data_dir: str = "data/",
            log_on_step: bool = True,
            metrics: dict = {},
            **litmodule_kwargs,
    ):
        super().__init__(**litmodule_kwargs)

        # This line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        # Note: we will send to logger with `self.logger.save_hyperparams` in `setup`
        self.save_hyperparameters(logger=False)

        self.env = env
        self.policy = policy
        self.instantiate_metrics(metrics)
        self.log_on_step = log_on_step

        self.data_cfg = {
            "batch_size": batch_size,
            "val_batch_size": val_batch_size,
            "test_batch_size": test_batch_size,
            "generate_default_data": generate_default_data,
            "data_dir": data_dir,
            "train_data_size": train_data_size,
            "val_data_size": val_data_size,
            "test_data_size": test_data_size,
        }

        self._optimizer_name_or_cls: str | torch.optim.Optimizer = optimizer
        self.optimizer_kwargs: dict = optimizer_kwargs
        self._lr_scheduler_name_or_cls: str | torch.optim.lr_scheduler.LRScheduler = (
            lr_scheduler
        )
        self.lr_scheduler_kwargs: dict = lr_scheduler_kwargs
        self.lr_scheduler_interval: str = lr_scheduler_interval
        self.lr_scheduler_monitor: str = lr_scheduler_monitor

        self.shuffle_train_dataloader = shuffle_train_dataloader
        self.dataloader_num_workers = dataloader_num_workers

    def on_train_start(self):
        self.train_start_time = time.time()

    def instantiate_metrics(self, metrics: dict):
        """Dictionary of metrics to be logged at each phase"""

        if not metrics:
            log.info("No metrics specified, using default")
        self.train_metrics = metrics.get("train", ["loss", "reward"])
        self.val_metrics = metrics.get("val", ["reward"])
        self.test_metrics = metrics.get("test", ["reward"])
        self.log_on_step = metrics.get("log_on_step", True)

    def setup(self, stage="fit"):
        """Base LightningModule setup method. This will setup the datasets and dataloaders

        Note:
            We also send to the loggers all hyperparams that are not `nn.Module` (i.e. the policy).
            Apparently PyTorch Lightning does not do this by default.
        """

        log.info("Setting up batch sizes for train/val/test")
        train_bs, val_bs, test_bs = (
            self.data_cfg["batch_size"],
            self.data_cfg["val_batch_size"],
            self.data_cfg["test_batch_size"],
        )
        self.train_batch_size = train_bs
        self.val_batch_size = train_bs if val_bs is None else val_bs
        self.test_batch_size = self.val_batch_size if test_bs is None else test_bs

        if self.data_cfg["generate_default_data"]:
            log.info(
                "Generating default datasets. If found, they will not be overwritten"
            )
            generate_default_datasets(data_dir=self.data_cfg["data_dir"])

        log.info("Setting up datasets")
        self.train_dataset = self.wrap_dataset(
            self.env.dataset(self.data_cfg["train_data_size"], phase="train")
        )
        self.val_dataset = self.env.dataset(self.data_cfg["val_data_size"], phase="val")
        self.test_dataset = self.env.dataset(
            self.data_cfg["test_data_size"], phase="test"
        )
        self.dataloader_names = None
        self.setup_loggers()
        self.post_setup_hook()

    def setup_loggers(self):
        """Log all hyperparameters except those in `nn.Module`"""
        if self.loggers is not None:
            hparams_save = {
                k: v for k, v in self.hparams.items() if not isinstance(v, nn.Module)
            }
            for logger in self.loggers:
                logger.log_hyperparams(hparams_save)
                logger.log_graph(self)
                logger.save()

    def post_setup_hook(self):
        """Hook to be called after setup. Can be used to set up subclasses without overriding `setup`"""
        pass

    def configure_optimizers(self, parameters=None):
        """
        Args:
            parameters: parameters to be optimized. If None, will use `self.parameters()`, i.e. all parameters
        """

        if parameters is None:
            parameters = self.parameters()

        log.info(f"Instantiating optimizer <{self._optimizer_name_or_cls}>")
        if isinstance(self._optimizer_name_or_cls, str):
            optimizer = create_optimizer(
                parameters, self._optimizer_name_or_cls, **self.optimizer_kwargs
            )
        elif isinstance(self._optimizer_name_or_cls, partial):
            optimizer = self._optimizer_name_or_cls(parameters, **self.optimizer_kwargs)
        else:  # User-defined optimizer
            opt_cls = self._optimizer_name_or_cls
            optimizer = opt_cls(parameters, **self.optimizer_kwargs)
            assert isinstance(optimizer, torch.optim.Optimizer)

        # instantiate lr scheduler
        if self._lr_scheduler_name_or_cls is None:
            return optimizer
        else:
            log.info(f"Instantiating LR scheduler <{self._lr_scheduler_name_or_cls}>")
            if isinstance(self._lr_scheduler_name_or_cls, str):
                scheduler = create_scheduler(
                    optimizer, self._lr_scheduler_name_or_cls, **self.lr_scheduler_kwargs
                )
            elif isinstance(self._lr_scheduler_name_or_cls, partial):
                scheduler = self._lr_scheduler_name_or_cls(
                    optimizer, **self.lr_scheduler_kwargs
                )
            else:  # User-defined scheduler
                scheduler_cls = self._lr_scheduler_name_or_cls
                scheduler = scheduler_cls(optimizer, **self.lr_scheduler_kwargs)
                assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
            return [optimizer], {
                "scheduler": scheduler,
                "interval": self.lr_scheduler_interval,
                "monitor": self.lr_scheduler_monitor,
            }

    def log_metrics(
            self, metric_dict: dict, phase: str, dataloader_idx: int | None = None
    ):
        """Log metrics to logger and progress bar"""
        metrics = getattr(self, f"{phase}_metrics")
        dataloader_name = ""
        if dataloader_idx is not None and self.dataloader_names is not None:
            dataloader_name = "/" + self.dataloader_names[dataloader_idx]
        metrics = {
            f"{phase}/{k}{dataloader_name}": (
                v.mean() if isinstance(v, torch.Tensor) else v
            )
            for k, v in metric_dict.items()
            if k in metrics
        }
        log_on_step = self.log_on_step if phase == "train" else False
        on_epoch = False if phase == "train" else True
        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,  # we add manually above
        )
        return metrics

    def forward(self, td, **kwargs):
        """Forward pass for the model. Simple wrapper around `policy`. Uses `env` from the module if not provided."""
        if kwargs.get("env", None) is None:
            env = self.env
        else:
            log.info("Using env from kwargs")
            env = kwargs.pop("env")
        return self.policy(td, env, **kwargs)

    def shared_step(self, batch: Any, batch_idx: int, phase: str, **kwargs):
        """Shared step between train/val/test. To be implemented in subclass"""
        raise NotImplementedError("Shared step is required to implemented in subclass")

    def training_step(self, batch: Any, batch_idx: int):
        # To use new data every epoch, we need to call reload_dataloaders_every_epoch=True in Trainer
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self.shared_step(
            batch, batch_idx, phase="val", dataloader_idx=dataloader_idx
        )

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self.shared_step(
            batch, batch_idx, phase="test", dataloader_idx=dataloader_idx
        )

    def train_dataloader(self):
        return self._dataloader(
            self.train_dataset, self.train_batch_size, self.shuffle_train_dataloader
        )

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, self.val_batch_size)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, self.test_batch_size)

    def on_train_epoch_end(self):
        """Called at the end of the training epoch. This can be used for instance to update the train dataset
        with new data (which is the case in RL).
        """
        # Only update if not in the first epoch
        # If last epoch, we don't need to update since we will not use the dataset anymore
        if self.current_epoch < self.trainer.max_epochs - 1:
            log.info("Generating training dataset for next epoch...")
            train_dataset = self.env.dataset(self.data_cfg["train_data_size"], "train")
            self.train_dataset = self.wrap_dataset(train_dataset)

    def wrap_dataset(self, dataset):
        """Wrap dataset with policy-specific wrapper. This is useful i.e. in REINFORCE where we need to
        collect the greedy rollout baseline outputs.
        """
        return dataset

    def _dataloader(self, dataset, batch_size, shuffle=False):
        """Handle both single datasets and list / dict of datasets"""
        if isinstance(dataset, Iterable):
            # load dataloader names if available as dict, else use indices
            if isinstance(dataset, dict):
                self.dataloader_names = list(dataset.keys())
            else:
                self.dataloader_names = [f"{i}" for i in range(len(dataset))]
            # if batch size is int, make it into list
            if isinstance(batch_size, int):
                batch_size = [batch_size] * len(self.dataloader_names)
            assert len(batch_size) == len(
                self.dataloader_names
            ), f"Batch size must match number of datasets. \
                        Found: {len(batch_size)} and {len(self.dataloader_names)}"
            return [
                self._dataloader_single(dset, bsize, shuffle)
                for dset, bsize in zip(dataset.values(), batch_size)
            ]
        else:
            assert isinstance(
                batch_size, int
            ), f"Batch size must be an integer for a single dataset, found {batch_size}"
            return self._dataloader_single(dataset, batch_size, shuffle)

    def _dataloader_single(self, dataset, batch_size, shuffle=False):
        """The dataloader used by the trainer. This is a wrapper around the dataset with a custom collate_fn
        to efficiently handle TensorDicts.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.dataloader_num_workers,
            collate_fn=dataset.collate_fn,
        )
