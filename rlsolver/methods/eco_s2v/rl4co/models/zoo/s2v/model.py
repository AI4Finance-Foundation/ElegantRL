from rlsolver.methods.eco_s2v.rl4co.envs.common.base import RL4COEnvBase
from rlsolver.methods.eco_s2v.rl4co.models.rl import REINFORCE
from rlsolver.methods.eco_s2v.rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rlsolver.methods.eco_s2v.rl4co.models.zoo.s2v.policy import S2VModelPolicy


class S2VModel(REINFORCE):
    """Attention Model based on REINFORCE: https://arxiv.org/abs/1803.08475.
    Check :class:`REINFORCE` and :class:`rl4co.models.RL4COLitModule` for more details such as additional parameters  including batch size.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to rollout (1 epoch of exponential, then greedy rollout baseline)
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
            self,
            env: RL4COEnvBase,
            policy: S2VModelPolicy = None,
            baseline: REINFORCEBaseline | str = "rollout",
            policy_kwargs={},
            baseline_kwargs={},
            **kwargs,
    ):
        if policy is None:
            policy = S2VModelPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
