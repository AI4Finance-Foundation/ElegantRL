import abc
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from rlsolver.methods.eco_s2v.rl4co.envs.common.base import RL4COEnvBase
from rlsolver.methods.eco_s2v.rl4co.models.rl.common.critic import CriticNetwork, create_critic_from_actor
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class REINFORCEBaseline(nn.Module, metaclass=abc.ABCMeta):
    """Base class for REINFORCE baselines"""

    def __init__(self, *args, **kw):
        super().__init__()
        pass

    def wrap_dataset(self, dataset: Dataset, *args, **kw):
        """Wrap dataset with baseline-specific functionality"""
        return dataset

    @abc.abstractmethod
    def eval(
            self, td: TensorDict, reward: torch.Tensor, env: RL4COEnvBase = None, **kwargs
    ):
        """Evaluate baseline"""
        raise NotImplementedError

    def epoch_callback(self, *args, **kw):
        """Callback at the end of each epoch
        For example, update baseline parameters and obtain baseline values
        """
        pass

    def setup(self, *args, **kw):
        """To be called before training during setup phase
        This follow PyTorch Lightning's setup() convention
        """
        pass


class NoBaseline(REINFORCEBaseline):
    """No baseline: return 0 for baseline and neg_los"""

    def eval(self, td, reward, env=None):
        return 0, 0  # No baseline, no neg_los


class SharedBaseline(REINFORCEBaseline):
    """Shared baseline: return mean of reward as baseline"""

    def eval(self, td, reward, env=None, on_dim=1):  # e.g. [batch, pomo, ...]
        return reward.mean(dim=on_dim, keepdims=True), 0


class ExponentialBaseline(REINFORCEBaseline):
    """Exponential baseline: return exponential moving average of reward as baseline

    Args:
        beta: Beta value for the exponential moving average
    """

    def __init__(self, beta=0.8, **kw):
        super(REINFORCEBaseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, td, reward, env=None):
        if self.v is None:
            v = reward.mean()
        else:
            v = self.beta * self.v + (1.0 - self.beta) * reward.mean()
        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss


class MeanBaseline(REINFORCEBaseline):
    """Mean baseline: return mean of reward as baseline"""

    def __new__(cls, **kw):
        return ExponentialBaseline(beta=0.0, **kw)


class WarmupBaseline(REINFORCEBaseline):
    """Warmup baseline: return convex combination of baseline and exponential baseline

    Args:
        baseline: Baseline to use after warmup
        n_epochs: Number of epochs to warmup
        warmup_exp_beta: Beta value for the exponential baseline during warmup
    """

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8, **kw):
        super(REINFORCEBaseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset, *args, **kw):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset, *args, **kw)
        return self.warmup_baseline.wrap_dataset(dataset, *args, **kw)

    def setup(self, *args, **kw):
        self.baseline.setup(*args, **kw)

    def eval(self, td, reward, env=None):
        if self.alpha == 1:
            return self.baseline.eval(td, reward, env)
        if self.alpha == 0:
            return self.warmup_baseline.eval(td, reward, env)
        v_b, l_b = self.baseline.eval(td, reward, env)
        v_wb, l_wb = self.warmup_baseline.eval(td, reward, env)
        # Return convex combination of baseline and of loss
        return (
            self.alpha * v_b + (1 - self.alpha) * v_wb,
            self.alpha * l_b + (1 - self.alpha) * l_wb,
        )

    def epoch_callback(self, *args, **kw):
        # Need to call epoch callback of inner policy (also after first epoch if we have not used it)
        self.baseline.epoch_callback(*args, **kw)
        if kw["epoch"] < self.n_epochs:
            self.alpha = (kw["epoch"] + 1) / float(self.n_epochs)
            log.info("Set warmup alpha = {}".format(self.alpha))


class CriticBaseline(REINFORCEBaseline):
    """Critic baseline: use critic network as baseline

    Args:
        critic: Critic network to use as baseline. If None, create a new critic network based on the environment
    """

    def __init__(self, critic: CriticNetwork = None, **unused_kw):
        super(CriticBaseline, self).__init__()
        self.critic = critic

    def setup(self, policy, env, **kwargs):
        if self.critic is None:
            log.info("Critic not found. Creating critic network for {}".format(env.name))
            self.critic = create_critic_from_actor(policy)

    def eval(self, x, c, env=None):
        v = self.critic(x).squeeze(-1)
        # detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, c.detach())


class RolloutBaseline(REINFORCEBaseline):
    """Rollout baseline: use greedy rollout as baseline

    Args:
        bl_alpha: Alpha value for the baseline T-test
    """

    def __init__(self, bl_alpha=0.05, **kw):
        super(RolloutBaseline, self).__init__()
        self.bl_alpha = bl_alpha

    def setup(self, *args, **kw):
        self._update_policy(*args, **kw)

    def _update_policy(
            self, policy, env, batch_size=64, device="cpu", dataset_size=None, dataset=None
    ):
        """Update policy (=actor) and rollout baseline values"""
        self.policy = copy.deepcopy(policy).to(device)
        if dataset is None:
            log.info("Creating evaluation dataset for rollout baseline")
            self.dataset = env.dataset(batch_size=[dataset_size])

        log.info("Evaluating baseline policy on evaluation dataset")
        self.bl_vals = (
            self.rollout(self.policy, env, batch_size, device, self.dataset).cpu().numpy()
        )
        self.mean = self.bl_vals.mean()

    def eval(self, td, reward, env):
        """Evaluate rollout baseline

        Warning:
            This is not differentiable and should only be used for evaluation.
            Also, it is recommended to use the `rollout` method directly instead of this method.
        """
        with torch.inference_mode():
            reward = self.policy(td, env)["reward"]
        return reward, 0

    def epoch_callback(
            self, policy, env, batch_size=64, device="cpu", epoch=None, dataset_size=None
    ):
        """Challenges the current baseline with the policy and replaces the baseline policy if it is improved"""
        log.info("Evaluating candidate policy on evaluation dataset")
        candidate_vals = self.rollout(policy, env, batch_size, device).cpu().numpy()
        candidate_mean = candidate_vals.mean()

        log.info(
            "Candidate mean: {:.3f}, Baseline mean: {:.3f}".format(
                candidate_mean, self.mean
            )
        )
        if candidate_mean - self.mean > 0:
            # Calc p value with inverse logic (costs)
            t, p = ttest_rel(-candidate_vals, -self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            log.info("p-value: {:.3f}".format(p_val))
            if p_val < self.bl_alpha:
                log.info("Updating baseline")
                self._update_policy(policy, env, batch_size, device, dataset_size)

    def rollout(self, policy, env, batch_size=64, device="cpu", dataset=None):
        """Rollout the policy on the given dataset"""

        # if dataset is None, use the dataset of the baseline
        dataset = self.dataset if dataset is None else dataset

        policy.eval()
        policy = policy.to(device)

        def eval_policy(batch):
            with torch.inference_mode():
                batch = env.reset(batch.to(device))
                return policy(batch, env, decode_type="greedy")["reward"]

        dl = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

        rewards = torch.cat([eval_policy(batch) for batch in dl], 0)
        # breakpoint()
        return rewards

    def wrap_dataset(self, dataset, env, batch_size=64, device="cpu", **kw):
        """Wrap the dataset in a baseline dataset

        Note:
            This is an alternative to `eval` that does not require the policy to be passed
            at every call but just once. Values are added to the dataset. This also allows for
            larger batch sizes since we evauate the policy without gradients.
        """
        rewards = (
            self.rollout(self.policy, env, batch_size, device, dataset=dataset)
            .detach()
            .cpu()
        )
        return dataset.add_key("extra", rewards)

    def __getstate__(self):
        """Do not include datasets in state to avoid pickling issues"""
        state = self.__dict__.copy()
        try:
            del state["dataset"]
        except KeyError:
            pass
        return state

    def __setstate__(self, state):
        """Restore datasets after unpickling. Will be restored in setup"""
        self.__dict__.update(state)
        self.dataset = None


REINFORCE_BASELINES_REGISTRY = {
    "no": NoBaseline,
    "shared": SharedBaseline,
    "exponential": ExponentialBaseline,
    "critic": CriticBaseline,
    "mean": MeanBaseline,
    "rollout_only": RolloutBaseline,
    "warmup": WarmupBaseline,
}


def get_reinforce_baseline(name, **kw):
    """Get a REINFORCE baseline by name
    The rollout baseline default to warmup baseline with one epoch of
    exponential baseline and the greedy rollout
    """
    if name == "warmup":
        inner_baseline = kw.get("baseline", "rollout")
        if not isinstance(inner_baseline, REINFORCEBaseline):
            inner_baseline = get_reinforce_baseline(inner_baseline, **kw)
        return WarmupBaseline(inner_baseline, **kw)
    elif name == "rollout":
        warmup_epochs = kw.get("n_epochs", 1)
        warmup_exp_beta = kw.get("exp_beta", 0.8)
        bl_alpha = kw.get("bl_alpha", 0.05)
        return WarmupBaseline(
            RolloutBaseline(bl_alpha=bl_alpha), warmup_epochs, warmup_exp_beta
        )

    if name is None:
        name = "no"  # default to no baseline
    baseline_cls = REINFORCE_BASELINES_REGISTRY.get(name, None)
    if baseline_cls is None:
        raise ValueError(
            f"Unknown baseline {baseline_cls}. Available baselines: {REINFORCE_BASELINES_REGISTRY.keys()}"
        )
    return baseline_cls(**kw)
