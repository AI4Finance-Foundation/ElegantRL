# Adapted from https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html#GPUStatsMonitor
# We only need the speed monitoring, not the GPU monitoring
import time

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.parsing import AttributeDict
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class SpeedMonitor(Callback):
    """Monitor the speed of each step and each epoch."""

    def __init__(
            self,
            intra_step_time: bool = True,
            inter_step_time: bool = True,
            epoch_time: bool = True,
            verbose=False,
    ):
        super().__init__()
        self._log_stats = AttributeDict(
            {
                "intra_step_time": intra_step_time,
                "inter_step_time": inter_step_time,
                "epoch_time": epoch_time,
            }
        )
        self.verbose = verbose

    def on_train_start(self, trainer: "L.Trainer", L_module: "L.LightningModule") -> None:
        self._snap_epoch_time = None

    def on_train_epoch_start(
            self, trainer: "L.Trainer", L_module: "L.LightningModule"
    ) -> None:
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None
        self._snap_epoch_time = time.time()

    def on_validation_epoch_start(
            self, trainer: "L.Trainer", L_module: "L.LightningModule"
    ) -> None:
        self._snap_inter_step_time = None

    def on_test_epoch_start(
            self, trainer: "L.Trainer", L_module: "L.LightningModule"
    ) -> None:
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(
            self,
            trainer: "L.Trainer",
            *unused_args,
            **unused_kwargs,  # easy fix for new pytorch lightning versions
    ) -> None:
        if self._log_stats.intra_step_time:
            self._snap_intra_step_time = time.time()

        if not self._should_log(trainer):
            return

        logs = {}
        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs["time/inter_step (ms)"] = (
                                                   time.time() - self._snap_inter_step_time
                                           ) * 1000

        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(
            self,
            trainer: "L.Trainer",
            L_module: "L.LightningModule",
            *unused_args,
            **unused_kwargs,  # easy fix for new pytorch lightning versions
    ) -> None:
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if (
                self.verbose
                and self._log_stats.intra_step_time
                and self._snap_intra_step_time
        ):
            L_module.print(
                f"time/intra_step (ms): {(time.time() - self._snap_intra_step_time) * 1000}"
            )

        if not self._should_log(trainer):
            return

        logs = {}
        if self._log_stats.intra_step_time and self._snap_intra_step_time:
            logs["time/intra_step (ms)"] = (
                                                   time.time() - self._snap_intra_step_time
                                           ) * 1000

        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(
            self,
            trainer: "L.Trainer",
            L_module: "L.LightningModule",
    ) -> None:
        logs = {}
        if self._log_stats.epoch_time and self._snap_epoch_time:
            logs["time/epoch (s)"] = time.time() - self._snap_epoch_time
        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @staticmethod
    def _should_log(trainer) -> bool:
        return (
                trainer.global_step + 1
        ) % trainer.log_every_n_steps == 0 or trainer.should_stop
