from typing import Iterable, Optional

import lightning.pytorch as pl
import torch
from lightning import Callback, Trainer
from lightning.fabric.accelerators.cuda import num_cuda_devices
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies import DDPStrategy, Strategy
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from rlsolver.methods.eco_s2v.rl4co import utils

log = utils.get_pylogger(__name__)


class RL4COTrainer(Trainer):
    """Wrapper around Lightning Trainer, with some RL4CO magic for efficient training.

    Note:
        The most important hyperparameter to use is `reload_dataloaders_every_n_epochs`.
        This allows for datasets to be re-created on the run and distributed by Lightning across
        devices on each epoch. Setting to a value different than 1 may lead to overfitting to a
        specific (such as the initial) data distribution.

    Args:
        accelerator: hardware accelerator to use.
        callbacks: list of callbacks.
        logger: logger (or iterable collection of loggers) for experiment tracking.
        min_epochs: minimum number of training epochs.
        max_epochs: maximum number of training epochs.
        strategy: training strategy to use (if any), such as Distributed Data Parallel (DDP).
        devices: number of devices to train on (int) or which GPUs to train on (list or str) applied per node.
        gradient_clip_val: 0 means don't clip. Defaults to 1.0 for stability.
        precision: allows for mixed precision training. Can be specified as a string (e.g., '16').
            This also allows to use `FlashAttention` by default.
        disable_profiling_executor: Disable JIT profiling executor. This reduces memory and increases speed.
        auto_configure_ddp: Automatically configure DDP strategy if multiple GPUs are available.
        reload_dataloaders_every_n_epochs: Set to a value different than 1 to reload dataloaders every n epochs.
        matmul_precision: Set matmul precision for faster inference https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
        **kwargs: Additional keyword arguments passed to the Lightning Trainer. See :class:`lightning.pytorch.trainer.Trainer` for details.
    """

    def __init__(
            self,
            accelerator: str | Accelerator = "auto",
            callbacks: Optional[list[Callback]] = None,
            logger: Optional[Logger | Iterable[Logger]] = None,
            min_epochs: Optional[int] = None,
            max_epochs: Optional[int] = None,
            strategy: str | Strategy = "auto",
            devices: list[int] | str | int = "auto",
            gradient_clip_val: int | float = 1.0,
            precision: str | int = "16-mixed",
            reload_dataloaders_every_n_epochs: int = 1,
            disable_profiling_executor: bool = True,
            auto_configure_ddp: bool = True,
            matmul_precision: str | int = "medium",
            **kwargs,
    ):
        # Disable JIT profiling executor. This reduces memory and increases speed.
        # Reference: https://github.com/HazyResearch/safari/blob/111d2726e7e2b8d57726b7a8b932ad8a4b2ad660/train.py#LL124-L129C17
        if disable_profiling_executor:
            try:
                torch._C._jit_set_profiling_executor(False)
                torch._C._jit_set_profiling_mode(False)
            except AttributeError:
                pass

        # Configure DDP automatically if multiple GPUs are available
        if auto_configure_ddp and strategy == "auto":
            if devices == "auto":
                n_devices = num_cuda_devices()
            elif isinstance(devices, Iterable):
                n_devices = len(devices)
            else:
                n_devices = devices
            if n_devices > 1:
                log.info(
                    "Configuring DDP strategy automatically with {} GPUs".format(
                        n_devices
                    )
                )
                strategy = DDPStrategy(
                    find_unused_parameters=True,  # We set to True due to RL envs
                    gradient_as_bucket_view=True,  # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
                )

        # Set matmul precision for faster inference https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
        if matmul_precision is not None:
            torch.set_float32_matmul_precision(matmul_precision)

        # Check if gradient_clip_val is set to None
        if gradient_clip_val is None:
            log.warning(
                "gradient_clip_val is set to None. This may lead to unstable training."
            )

        # We should reload dataloaders every epoch for RL training
        if reload_dataloaders_every_n_epochs != 1:
            log.warning(
                "We reload dataloaders every epoch for RL training. Setting reload_dataloaders_every_n_epochs to a value different than 1 "
                + "may lead to unexpected behavior since the initial conditions will be the same for `n_epochs` epochs."
            )

        # Main call to `Trainer` superclass
        super().__init__(
            accelerator=accelerator,
            callbacks=callbacks,
            logger=logger,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            strategy=strategy,
            gradient_clip_val=gradient_clip_val,
            devices=devices,
            precision=precision,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            **kwargs,
        )

    def fit(
            self,
            model: "pl.LightningModule",
            train_dataloaders: Optional[TRAIN_DATALOADERS | LightningDataModule] = None,
            val_dataloaders: Optional[EVAL_DATALOADERS] = None,
            datamodule: Optional[LightningDataModule] = None,
            ckpt_path: Optional[str] = None,
    ) -> None:
        """
        We override the `fit` method to automatically apply and handle RL4CO magic
        to 'self.automatic_optimization = False' models, such as PPO

        It behaves exactly like the original `fit` method, but with the following changes:
        - if the given model is 'self.automatic_optimization = False', we override 'gradient_clip_val' as None
        """

        if not model.automatic_optimization:
            if self.gradient_clip_val is not None:
                log.warning(
                    "Overriding gradient_clip_val to None for 'automatic_optimization=False' models"
                )
                self.gradient_clip_val = None

        super().fit(
            model=model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
