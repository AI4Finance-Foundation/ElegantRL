import abc
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict

from rlsolver.methods.eco_s2v.rl4co.envs import RL4COEnvBase
from rlsolver.methods.eco_s2v.rl4co.utils.ops import batchify, gather_by_index, unbatchify, unbatchify_and_gather
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def get_decoding_strategy(decoding_strategy, **config):
    strategy_registry = {
        "greedy": Greedy,
        "sampling": Sampling,
        "multistart_greedy": Greedy,
        "multistart_sampling": Sampling,
        "beam_search": BeamSearch,
        "evaluate": Evaluate,
    }

    if decoding_strategy not in strategy_registry:
        log.warning(
            f"Unknown decode type '{decoding_strategy}'. Available decode types: {strategy_registry.keys()}. Defaulting to Sampling."
        )

    if "multistart" in decoding_strategy:
        config["multistart"] = True

    return strategy_registry.get(decoding_strategy, Sampling)(**config)


def get_log_likelihood(logprobs, actions=None, mask=None, return_sum: bool = True):
    """Get log likelihood of selected actions.
    Note that mask is a boolean tensor where True means the value should be kept.

    Args:
        logprobs: Log probabilities of actions from the model (batch_size, seq_len, action_dim).
        actions: Selected actions (batch_size, seq_len).
        mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
        return_sum: Whether to return the sum of log probabilities or not. Defaults to True.
    """
    # Optional: select logp when logp.shape = (bs, dec_steps, N)
    if actions is not None and logprobs.dim() == 3:
        logprobs = logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        logprobs[~mask] = 0

    assert (
            logprobs > -1000
    ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    if return_sum:
        return logprobs.sum(1)  # [batch]
    else:
        return logprobs  # [batch, decode_len]


def decode_logprobs(logprobs, mask, decode_type="sampling"):
    """Decode log probabilities to select actions with mask.
    Note that mask is a boolean tensor where True means the value should be kept.
    """
    if "greedy" in decode_type:
        selected = DecodingStrategy.greedy(logprobs, mask)
    elif "sampling" in decode_type:
        selected = DecodingStrategy.sampling(logprobs, mask)
    else:
        assert False, "Unknown decode type: {}".format(decode_type)
    return selected


def random_policy(td):
    """Helper function to select a random action from available actions"""
    action = torch.multinomial(td["action_mask"].float(), 1).squeeze(-1)
    td.set("action", action)
    return td


def rollout(env, td, policy, max_steps: int = None):
    """Helper function to rollout a policy. Currently, TorchRL does not allow to step
    over envs when done with `env.rollout()`. We need this because for environments that complete at different steps.
    """

    max_steps = float("inf") if max_steps is None else max_steps
    actions = []
    steps = 0

    while not td["done"].all():
        td = policy(td)
        actions.append(td["action"])
        td = env.step(td)["next"]
        steps += 1
        if steps > max_steps:
            log.info("Max steps reached")
            break
    return (
        env.get_reward(td, torch.stack(actions, dim=1)),
        td,
        torch.stack(actions, dim=1),
    )


def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done out-of-place.
    Ref: https://github.com/togethercomputer/stripedhyena/blob/7e13f618027fea9625be1f2d2d94f9a361f6bd02/stripedhyena/sample.py#L6
    """
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, float("-inf"))


def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done out-of-place.
    Ref: https://github.com/togethercomputer/stripedhyena/blob/7e13f618027fea9625be1f2d2d94f9a361f6bd02/stripedhyena/sample.py#L14
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def process_logits(
        logits: torch.Tensor,
        mask: torch.Tensor = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
):
    """Convert logits to log probabilities with additional features like temperature scaling, top-k and top-p sampling.

    Note:
        We convert to log probabilities instead of probabilities to avoid numerical instability.
        This is because, roughly, softmax = exp(logits) / sum(exp(logits)) and log(softmax) = logits - log(sum(exp(logits))),
        and avoiding the division by the sum of exponentials can help with numerical stability.
        You may check the [official PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html).

    Args:
        logits: Logits from the model (batch_size, num_actions).
        mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
        temperature: Temperature scaling. Higher values make the distribution more uniform (exploration),
            lower values make it more peaky (exploitation).
        top_p: Top-p sampling, a.k.a. Nucleus Sampling (https://arxiv.org/abs/1904.09751). Remove tokens that have a cumulative probability
            less than the threshold 1 - top_p (lower tail of the distribution). If 0, do not perform.
        top_k: Top-k sampling, i.e. restrict sampling to the top k logits. If 0, do not perform. Note that we only do filtering and
            do not return all the top-k logits here.
        tanh_clipping: Tanh clipping (https://arxiv.org/abs/1611.09940).
        mask_logits: Whether to mask logits of infeasible actions.
    """

    # Tanh clipping from Bello et al. 2016
    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping

    # In RL, we want to mask the logits to prevent the agent from selecting infeasible actions
    if mask_logits:
        assert mask is not None, "mask must be provided if mask_logits is True"
        logits[~mask] = float("-inf")

    logits = logits / temperature  # temperature scaling

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # safety check
        logits = modify_logits_for_top_k_filtering(logits, top_k)

    if top_p > 0:
        assert top_p <= 1.0, "top-p should be in (0, 1]."
        logits = modify_logits_for_top_p_filtering(logits, top_p)

    # Compute log probabilities
    return F.log_softmax(logits, dim=-1)


class DecodingStrategy(metaclass=abc.ABCMeta):
    """Base class for decoding strategies. Subclasses should implement the :meth:`_step` method.
    Includes hooks for pre and post main decoding operations.

    Args:
        temperature: Temperature scaling. Higher values make the distribution more uniform (exploration),
            lower values make it more peaky (exploitation). Defaults to 1.0.
        top_p: Top-p sampling, a.k.a. Nucleus Sampling (https://arxiv.org/abs/1904.09751). Defaults to 0.0.
        top_k: Top-k sampling, i.e. restrict sampling to the top k logits. If 0, do not perform. Defaults to 0.
        mask_logits: Whether to mask logits of infeasible actions. Defaults to True.
        tanh_clipping: Tanh clipping (https://arxiv.org/abs/1611.09940). Defaults to 0.
        multisample: Whether to use sampling decoding. Defaults to False.
        num_samples: Number of samples to evaluate during decoding. Defaults to None.
        num_starts: Number of starts for multistart decoding. Defaults to None.
        multistart: Whether to use multistart decoding. Defaults to False.
        select_start_nodes_fn: Function to select start nodes for multistart decoding. Defaults to None.
        improvement_method_mode: Whether to use improvement method mode. Defaults to False.
        select_best: Whether to select the best action or return all. Defaults to False.
        store_all_logp: Whether to store all log probabilities. Defaults to False. If True, logprobs will be stored for all actions.
            Note that this will increase memory usage.
    """

    name = "base"

    def __init__(
            self,
            temperature: float = 1.0,
            top_p: float = 0.0,
            top_k: int = 0,
            mask_logits: bool = True,
            tanh_clipping: float = 0,
            num_samples: Optional[int] = None,
            multisample: bool = False,
            num_starts: Optional[int] = None,
            multistart: bool = False,
            select_start_nodes_fn: Optional[callable] = None,
            improvement_method_mode: bool = False,
            select_best: bool = False,
            store_all_logp: bool = False,
            **kwargs,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.mask_logits = mask_logits
        self.tanh_clipping = tanh_clipping
        # check if multistart (POMO) and multisample flags
        assert not (
                multistart and multisample
        ), "Using both multistart and multisample is not supported"
        if num_samples and num_starts:
            assert not (
                    num_samples > 1 and num_starts > 1
            ), f"num_samples={num_samples} and num_starts={num_starts} are both > 1"
        if num_samples is not None:
            multisample = True if num_samples > 1 else False
        if num_starts is not None:
            multistart = True if num_starts > 1 else False
        self.multistart = multistart
        self.multisample = multisample
        # num_starts is used for both multistart and multisample
        # the function is to use start multiple rollouts for the same instance in parallel
        self.num_starts = num_starts if multistart else num_samples

        self.select_start_nodes_fn = select_start_nodes_fn
        self.improvement_method_mode = improvement_method_mode
        self.select_best = select_best
        self.store_all_logp = store_all_logp
        # initialize buffers
        self.actions = []
        self.logprobs = []

    @abc.abstractmethod
    def _step(
            self,
            logprobs: torch.Tensor,
            mask: torch.Tensor,
            td: Optional[TensorDict] = None,
            action: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Main decoding operation. This method should be called in a loop until all sequences are done.

        Args:
            logprobs: Log probabilities processed from logits of the model.
            mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
            td: TensorDict containing the current state of the environment.
            action: Optional action to use, e.g. for evaluating log probabilities.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def pre_decoder_hook(
            self, td: TensorDict, env: RL4COEnvBase, action: Optional[torch.Tensor] = None
    ):
        """Pre decoding hook. This method is called before the main decoding operation."""

        # Multi-start decoding. If num_starts is None, we use the number of actions in the action mask
        if self.multistart or self.multisample:
            if self.num_starts is None:
                self.num_starts = env.get_num_starts(td)
                if self.multisample:
                    log.warning(
                        f"num_starts is not provided for sampling, using num_starts={self.num_starts}"
                    )
        else:
            if self.num_starts is not None:
                if self.num_starts >= 1:
                    log.warning(
                        f"num_starts={self.num_starts} is ignored for decode_type={self.name}"
                    )

            self.num_starts = 0

        # Multi-start decoding: first action is chosen by ad-hoc node selection
        if self.num_starts >= 1:
            if self.multistart:
                if action is None:  # if action is provided, we use it as the first action
                    if self.select_start_nodes_fn is not None:
                        action = self.select_start_nodes_fn(td, env, self.num_starts)
                    else:
                        action = env.select_start_nodes(td, num_starts=self.num_starts)

                # Expand td to batch_size * num_starts
                td = batchify(td, self.num_starts)

                td.set("action", action)
                td = env.step(td)["next"]
                # first logprobs is 0, so p = logprobs.exp() = 1
                if self.store_all_logp:
                    logprobs = torch.zeros_like(td["action_mask"])  # [B, N]
                else:
                    logprobs = torch.zeros_like(action, device=td.device)  # [B]

                self.logprobs.append(logprobs)
                self.actions.append(action)
            else:
                # Expand td to batch_size * num_samplestarts
                td = batchify(td, self.num_starts)

        return td, env, self.num_starts

    def post_decoder_hook(
            self, td: TensorDict, env: RL4COEnvBase
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict, RL4COEnvBase]:
        assert (
                len(self.logprobs) > 0
        ), "No logprobs were collected because all environments were done. Check your initial state"
        logprobs = torch.stack(self.logprobs, 1)
        actions = torch.stack(self.actions, 1)
        if self.num_starts > 0 and self.select_best:
            logprobs, actions, td, env = self._select_best(logprobs, actions, td, env)
        return logprobs, actions, td, env

    def step(
            self,
            logits: torch.Tensor,
            mask: torch.Tensor,
            td: Optional[TensorDict] = None,
            action: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> TensorDict:
        """Main decoding operation. This method should be called in a loop until all sequences are done.

        Args:
            logits: Logits from the model.
            mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
            td: TensorDict containing the current state of the environment.
            action: Optional action to use, e.g. for evaluating log probabilities.
        """
        if not self.mask_logits:  # set mask_logit to None if mask_logits is False
            mask = None

        logprobs = process_logits(
            logits,
            mask,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            tanh_clipping=self.tanh_clipping,
            mask_logits=self.mask_logits,
        )
        logprobs, selected_action, td = self._step(
            logprobs, mask, td, action=action, **kwargs
        )

        # directly return for improvement methods, since the action for improvement methods is finalized in its own policy
        if self.improvement_method_mode:
            return logprobs, selected_action

        # for others
        assert td is not None, "td must be provided"
        if not self.store_all_logp:
            logprobs = gather_by_index(logprobs, selected_action, dim=1)
        td.set("action", selected_action)
        self.actions.append(selected_action)
        self.logprobs.append(logprobs)
        return td

    @staticmethod
    def greedy(logprobs, mask=None):
        """Select the action with the highest probability."""
        # [BS], [BS]
        selected = logprobs.argmax(dim=-1)
        if mask is not None:
            assert (
                not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"

        return selected

    @staticmethod
    def sampling(logprobs, mask=None):
        """Sample an action with a multinomial distribution given by the log probabilities."""
        probs = logprobs.exp()
        selected = torch.multinomial(probs, 1).squeeze(1)

        if mask is not None:
            while (~mask).gather(1, selected.unsqueeze(-1)).data.any():
                log.info("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
            assert (
                not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"

        return selected

    def _select_best(self, logprobs, actions, td: TensorDict, env: RL4COEnvBase):
        rewards = env.get_reward(td, actions)
        _, max_idxs = unbatchify(rewards, self.num_starts).max(dim=-1)

        actions = unbatchify_and_gather(actions, max_idxs, self.num_starts)
        logprobs = unbatchify_and_gather(logprobs, max_idxs, self.num_starts)
        td = unbatchify_and_gather(td, max_idxs, self.num_starts)

        return logprobs, actions, td, env


class Greedy(DecodingStrategy):
    name = "greedy"

    def _step(
            self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Select the action with the highest log probability"""
        selected = self.greedy(logprobs, mask)
        return logprobs, selected, td


class Sampling(DecodingStrategy):
    name = "sampling"

    def _step(
            self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Sample an action with a multinomial distribution given by the log probabilities."""
        selected = self.sampling(logprobs, mask)
        return logprobs, selected, td


class Evaluate(DecodingStrategy):
    name = "evaluate"

    def _step(
            self,
            logprobs: torch.Tensor,
            mask: torch.Tensor,
            td: TensorDict,
            action: torch.Tensor,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """The action is provided externally, so we just return the action"""
        selected = action
        return logprobs, selected, td


class BeamSearch(DecodingStrategy):
    name = "beam_search"

    def __init__(self, beam_width=None, select_best=True, **kwargs) -> None:
        # TODO do we really need all logp in beam search?
        kwargs["store_all_logp"] = True
        super().__init__(**kwargs)
        self.beam_width = beam_width
        self.select_best = select_best
        self.parent_beam_logprobs = None
        self.beam_path = []

    def _step(
            self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        selected, batch_beam_idx = self._make_beam_step(logprobs)
        # select the correct state representation, logprobs and mask according to beam parent
        td = td[batch_beam_idx]
        logprobs = logprobs[batch_beam_idx]
        mask = mask[batch_beam_idx]

        assert (
            not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
        ), "infeasible action selected"

        return logprobs, selected, td

    def pre_decoder_hook(self, td: TensorDict, env: RL4COEnvBase, **kwargs):
        if self.beam_width is None:
            self.beam_width = env.get_num_starts(td)
        assert self.beam_width > 1, "beam width must be larger than 1"

        # select start nodes. TODO: include first step in beam search as well
        if self.select_start_nodes_fn is not None:
            action = self.select_start_nodes_fn(td, env, self.beam_width)
        else:
            action = env.select_start_nodes(td, num_starts=self.beam_width)

        # Expand td to batch_size * beam_width
        td = batchify(td, self.beam_width)

        td.set("action", action)
        td = env.step(td)["next"]

        logprobs = torch.zeros_like(td["action_mask"], device=td.device)
        beam_parent = torch.zeros(logprobs.size(0), device=td.device, dtype=torch.int32)

        self.logprobs.append(logprobs)
        self.actions.append(action)
        self.parent_beam_logprobs = logprobs.gather(1, action[..., None])
        self.beam_path.append(beam_parent)

        return td, env, self.beam_width

    def post_decoder_hook(self, td, env):
        # [BS*BW, seq_len]
        aligned_sequences, aligned_logprobs = self._backtrack()

        if self.select_best:
            return self._select_best_beam(aligned_logprobs, aligned_sequences, td, env)
        else:
            return aligned_logprobs, aligned_sequences, td, env

    def _backtrack(self):
        # [BS*BW, seq_len]
        actions = torch.stack(self.actions, 1)
        # [BS*BW, seq_len]
        logprobs = torch.stack(self.logprobs, 1)
        assert actions.size(1) == len(
            self.beam_path
        ), "action idx shape and beam path shape dont match"

        # [BS*BW]
        cur_parent = self.beam_path[-1]
        # [BS*BW]
        reversed_aligned_sequences = [actions[:, -1]]
        reversed_aligned_logprobs = [logprobs[:, -1]]

        aug_batch_size = actions.size(0)
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = (
            torch.arange(0, batch_size).repeat(self.beam_width).to(actions.device)
        )

        for k in reversed(range(len(self.beam_path) - 1)):
            batch_beam_idx = batch_beam_sequence + cur_parent * batch_size

            reversed_aligned_sequences.append(actions[batch_beam_idx, k])
            reversed_aligned_logprobs.append(logprobs[batch_beam_idx, k])
            cur_parent = self.beam_path[k][batch_beam_idx]

        # [BS*BW, seq_len*num_targets]
        actions = torch.stack(list(reversed(reversed_aligned_sequences)), dim=1)
        logprobs = torch.stack(list(reversed(reversed_aligned_logprobs)), dim=1)

        return actions, logprobs

    def _select_best_beam(self, logprobs, actions, td: TensorDict, env: RL4COEnvBase):
        aug_batch_size = logprobs.size(0)  # num nodes
        batch_size = aug_batch_size // self.beam_width
        rewards = env.get_reward(td, actions)
        _, idx = torch.cat(rewards.unsqueeze(1).split(batch_size), 1).max(1)
        flat_idx = torch.arange(batch_size, device=rewards.device) + idx * batch_size
        return logprobs[flat_idx], actions[flat_idx], td[flat_idx], env

    def _make_beam_step(self, logprobs: torch.Tensor):
        aug_batch_size, num_nodes = logprobs.shape  # num nodes
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = (
            torch.arange(0, batch_size).repeat(self.beam_width).to(logprobs.device)
        )

        # [BS*BW, num_nodes] + [BS*BW, 1] -> [BS*BW, num_nodes]
        log_beam_prob = logprobs + self.parent_beam_logprobs  #

        # [BS, num_nodes * BW]
        log_beam_prob_hstacked = torch.cat(log_beam_prob.split(batch_size), dim=1)
        # [BS, BW]
        topk_logprobs, topk_ind = torch.topk(
            log_beam_prob_hstacked, self.beam_width, dim=1
        )

        # [BS*BW, 1]
        logprobs_selected = torch.hstack(torch.unbind(topk_logprobs, 1)).unsqueeze(1)

        # [BS*BW, 1]
        topk_ind = torch.hstack(torch.unbind(topk_ind, 1))

        # since we stack the logprobs from the distinct branches, the indices in
        # topk dont correspond to node indices directly and need to be translated
        selected = topk_ind % num_nodes  # determine node index

        # calc parent this branch comes from
        beam_parent = (topk_ind // num_nodes).int()

        batch_beam_idx = batch_beam_sequence + beam_parent * batch_size

        self.parent_beam_logprobs = logprobs_selected
        self.beam_path.append(beam_parent)

        return selected, batch_beam_idx
