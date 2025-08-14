import time
from typing import IO, Any, Optional, cast

import torch
import torch.nn as nn
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.core.saving import _load_from_checkpoint
from tensordict import TensorDict
from typing_extensions import Self

from rlsolver.methods.eco_s2v.rl4co.envs.common.base import RL4COEnvBase
from rlsolver.methods.eco_s2v.rl4co.models.rl.common.base import RL4COLitModule
from rlsolver.methods.eco_s2v.rl4co.models.rl.common.utils import RewardScaler
from rlsolver.methods.eco_s2v.rl4co.models.rl.reinforce.baselines import REINFORCEBaseline, get_reinforce_baseline
from rlsolver.methods.eco_s2v.rl4co.utils.lightning import get_lightning_device
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class REINFORCE(RL4COLitModule):
    """REINFORCE algorithm, also known as policy gradients.
    See superclass `RL4COLitModule` for more details.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline
        baseline_kwargs: Keyword arguments for baseline. Ignored if baseline is not a string
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
            self,
            env: RL4COEnvBase,
            policy: nn.Module,
            baseline: REINFORCEBaseline | str = "rollout",
            baseline_kwargs: dict = {},
            reward_scale: str = None,
            **kwargs,
    ):
        super().__init__(env, policy, **kwargs)
        self.train_start_time = None

        self.save_hyperparameters(logger=False)

        if baseline == "critic":
            log.warning(
                "Using critic as baseline. If you want more granular support, use the A2C module instead."
            )

        if isinstance(baseline, str):
            baseline = get_reinforce_baseline(baseline, **baseline_kwargs)
        else:
            if baseline_kwargs != {}:
                log.warning("baseline_kwargs is ignored when baseline is not a string")
        self.baseline = baseline
        self.advantage_scaler = RewardScaler(reward_scale)

    def shared_step(
            self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out = self.policy(td, self.env, phase=phase, select_best=phase != "train")
        # Compute loss
        if phase == "train":
            out = self.calculate_loss(td, batch, out)
            if self.train_start_time is not None:
                elapsed_time = time.time() - self.train_start_time
                # self.log("train/reward_vs_time", td['reward'].mean(), on_step=True, prog_bar=True)
                self.log("train/time", elapsed_time, on_step=True, prog_bar=True)

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def calculate_loss(
            self,
            td: TensorDict,
            batch: TensorDict,
            policy_out: dict,
            reward: Optional[torch.Tensor] = None,
            log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss for REINFORCE algorithm.

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # REINFORCE baseline
        bl_val, bl_loss = (
            self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
        )

        # Main loss function
        advantage = reward - bl_val  # advantage = reward - baseline
        advantage = self.advantage_scaler(advantage)
        reinforce_loss = -(advantage * log_likelihood).mean()
        loss = reinforce_loss + bl_loss
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )
        return policy_out

    def post_setup_hook(self, stage="fit"):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(
            self.policy,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            dataset_size=self.data_cfg["val_data_size"],
        )

    def on_train_epoch_end(self):
        """Callback for end of training epoch: we evaluate the baseline"""
        self.baseline.epoch_callback(
            self.policy,
            env=self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            epoch=self.current_epoch,
            dataset_size=self.data_cfg["val_data_size"],
        )
        # Need to call super() for the dataset to be reset
        super().on_train_epoch_end()

    def wrap_dataset(self, dataset):
        """Wrap dataset from baseline evaluation. Used in greedy rollout baseline"""
        return self.baseline.wrap_dataset(
            dataset,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
        )

    def set_decode_type_multistart(self, phase: str):
        """Set decode type to `multistart` for train, val and test in policy.
        For example, if the decode type is `greedy`, it will be set to `multistart_greedy`.

        Args:
            phase: Phase to set decode type for. Must be one of `train`, `val` or `test`.
        """
        attribute = f"{phase}_decode_type"
        attr_get = getattr(self.policy, attribute)
        # If does not exist, log error
        if attr_get is None:
            log.error(f"Decode type for {phase} is None. Cannot prepend `multistart_`.")
            return
        elif "multistart" in attr_get:
            return
        else:
            setattr(self.policy, attribute, f"multistart_{attr_get}")

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: _PATH | IO,
            map_location: _MAP_LOCATION_TYPE = None,
            hparams_file: Optional[_PATH] = None,
            strict: bool = False,
            load_baseline: bool = True,
            **kwargs: Any,
    ) -> Self:
        """Load model from checkpoint/

        Note:
            This is a modified version of `load_from_checkpoint` from `pytorch_lightning.core.saving`.
            It deals with matching keys for the baseline by first running setup
        """

        if strict:
            log.warning("Setting strict=False for loading model from checkpoint.")
            strict = False

        # Do not use strict
        loaded = _load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            **kwargs,
        )

        # Load baseline state dict
        if load_baseline:
            # setup baseline first
            loaded.setup()
            loaded.post_setup_hook()
            # load baseline state dict
            state_dict = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )["state_dict"]
            # get only baseline parameters
            state_dict = {k: v for k, v in state_dict.items() if "baseline" in k}
            state_dict = {k.replace("baseline.", "", 1): v for k, v in state_dict.items()}
            loaded.baseline.load_state_dict(state_dict)

        return cast(Self, loaded)
