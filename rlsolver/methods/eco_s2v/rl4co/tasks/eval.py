import time

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rlsolver.methods.eco_s2v.rl4co.data.transforms import StateAugmentation
from rlsolver.methods.eco_s2v.rl4co.utils.ops import batchify, gather_by_index, sample_n_random_actions, unbatchify


def check_unused_kwargs(class_, kwargs):
    if len(kwargs) > 0 and not (len(kwargs) == 1 and "progress" in kwargs):
        print(f"Warning: {class_.__class__.__name__} does not use kwargs {kwargs}")


class EvalBase:
    """Base class for evaluation

    Args:
        env: Environment
        progress: Whether to show progress bar
        **kwargs: Additional arguments (to be implemented in subclasses)
    """

    name = "base"

    def __init__(self, env, progress=True, **kwargs):
        check_unused_kwargs(self, kwargs)
        self.env = env
        self.progress = progress

    def __call__(self, policy, dataloader, **kwargs):
        """Evaluate the policy on the given dataloader with **kwargs parameter
        self._inner is implemented in subclasses and returns actions and rewards
        """
        start = time.time()

        with torch.inference_mode():
            rewards_list = []
            actions_list = []

            for batch in tqdm(
                    dataloader, disable=not self.progress, desc=f"Running {self.name}"
            ):
                td = batch.to(next(policy.parameters()).device)
                td = self.env.reset(td)
                actions, rewards = self._inner(policy, td, **kwargs)
                rewards_list.append(rewards)
                actions_list.append(actions)

            rewards = torch.cat(rewards_list)

            # Padding: pad actions to the same length with zeros
            max_length = max(action.size(-1) for action in actions_list)
            actions = torch.cat(
                [
                    torch.nn.functional.pad(action, (0, max_length - action.size(-1)))
                    for action in actions_list
                ],
                0,
            )

        inference_time = time.time() - start

        tqdm.write(f"Mean reward for {self.name}: {rewards.mean():.4f}")
        tqdm.write(f"Time: {inference_time:.4f}s")

        # Empty cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "actions": actions.cpu(),
            "rewards": rewards.cpu(),
            "inference_time": inference_time,
            "avg_reward": rewards.cpu().mean(),
        }

    def _inner(self, policy, td):
        """Inner function to be implemented in subclasses.
        This function returns actions and rewards for the given policy
        """
        raise NotImplementedError("Implement in subclass")


class GreedyEval(EvalBase):
    """Evaluates the policy using greedy decoding and single trajectory"""

    name = "greedy"

    def __init__(self, env, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True))

    def _inner(self, policy, td):
        out = policy(
            td.clone(),
            decode_type="greedy",
            num_starts=0,
        )
        rewards = self.env.get_reward(td, out["actions"])
        return out["actions"], rewards


class AugmentationEval(EvalBase):
    """Evaluates the policy via N state augmentations
    `force_dihedral_8` forces the use of 8 augmentations (rotations and flips) as in POMO
    https://en.wikipedia.org/wiki/Examples_of_groups#dihedral_group_of_order_8

    Args:
        num_augment (int): Number of state augmentations
        force_dihedral_8 (bool): Whether to force the use of 8 augmentations
    """

    name = "augmentation"

    def __init__(self, env, num_augment=8, force_dihedral_8=False, feats=None, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True))
        self.augmentation = StateAugmentation(
            num_augment=num_augment,
            augment_fn="dihedral8" if force_dihedral_8 else "symmetric",
            feats=feats,
        )

    def _inner(self, policy, td, num_augment=None):
        if num_augment is None:
            num_augment = self.augmentation.num_augment
        td_init = td.clone()
        td = self.augmentation(td)
        out = policy(td.clone(), decode_type="greedy", num_starts=0)

        # Move into batches and compute rewards
        rewards = self.env.get_reward(batchify(td_init, num_augment), out["actions"])
        rewards = unbatchify(rewards, num_augment)
        actions = unbatchify(out["actions"], num_augment)

        # Get best reward and corresponding action
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards

    @property
    def num_augment(self):
        return self.augmentation.num_augment


class SamplingEval(EvalBase):
    """Evaluates the policy via N samples from the policy

    Args:
        samples (int): Number of samples to take
        softmax_temp (float): Temperature for softmax sampling. The higher the temperature, the more random the sampling
    """

    name = "sampling"

    def __init__(
            self,
            env,
            samples,
            softmax_temp=None,
            select_best=True,
            temperature=1.0,
            top_p=0.0,
            top_k=0,
            **kwargs,
    ):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True))

        self.samples = samples
        self.softmax_temp = softmax_temp
        self.temperature = temperature
        self.select_best = select_best
        self.top_p = top_p
        self.top_k = top_k

    def _inner(self, policy, td):
        out = policy(
            td.clone(),
            decode_type="sampling",
            num_starts=self.samples,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            multisample=True,
            softmax_temp=self.softmax_temp,
            select_best=self.select_best,
            select_start_nodes_fn=lambda td, _, n: sample_n_random_actions(td, n),
        )

        # Move into batches and compute rewards
        rewards = out["reward"]
        actions = out["actions"]

        return actions, rewards


class GreedyMultiStartEval(EvalBase):
    """Evaluates the policy via `num_starts` greedy multistarts samples from the policy

    Args:
        num_starts (int): Number of greedy multistarts to use
    """

    name = "multistart_greedy"

    def __init__(self, env, num_starts=None, **kwargs):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True))

        assert num_starts is not None, "Must specify num_starts"
        self.num_starts = num_starts

    def _inner(self, policy, td):
        td_init = td.clone()
        out = policy(
            td.clone(),
            decode_type="multistart_greedy",
            num_starts=self.num_starts,
        )

        # Move into batches and compute rewards
        td = batchify(td_init, self.num_starts)
        rewards = self.env.get_reward(td, out["actions"])
        rewards = unbatchify(rewards, self.num_starts)
        actions = unbatchify(out["actions"], self.num_starts)

        # Get the best trajectories
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards


class GreedyMultiStartAugmentEval(EvalBase):
    """Evaluates the policy via `num_starts` samples from the policy
    and `num_augment` augmentations of each sample.`
    `force_dihedral_8` forces the use of 8 augmentations (rotations and flips) as in POMO
    https://en.wikipedia.org/wiki/Examples_of_groups#dihedral_group_of_order_8

    Args:
        num_starts: Number of greedy multistart samples
        num_augment: Number of augmentations per sample
        force_dihedral_8: If True, force the use of 8 augmentations (rotations and flips) as in POMO
    """

    name = "multistart_greedy_augment"

    def __init__(
            self,
            env,
            num_starts=None,
            num_augment=8,
            force_dihedral_8=False,
            feats=None,
            **kwargs,
    ):
        check_unused_kwargs(self, kwargs)
        super().__init__(env, kwargs.get("progress", True))

        assert num_starts is not None, "Must specify num_starts"
        self.num_starts = num_starts
        assert not (
                num_augment != 8 and force_dihedral_8
        ), "Cannot force dihedral 8 when num_augment != 8"
        self.augmentation = StateAugmentation(
            num_augment=num_augment,
            augment_fn="dihedral8" if force_dihedral_8 else "symmetric",
            feats=feats,
        )

    def _inner(self, policy, td, num_augment=None):
        if num_augment is None:
            num_augment = self.augmentation.num_augment

        td_init = td.clone()

        td = self.augmentation(td)
        out = policy(
            td.clone(),
            decode_type="multistart_greedy",
            num_starts=self.num_starts,
        )

        # Move into batches and compute rewards
        td = batchify(td_init, (num_augment, self.num_starts))
        rewards = self.env.get_reward(td, out["actions"])
        rewards = unbatchify(rewards, self.num_starts * num_augment)
        actions = unbatchify(out["actions"], self.num_starts * num_augment)

        # Get the best trajectories
        rewards, max_idxs = rewards.max(dim=1)
        actions = gather_by_index(actions, max_idxs, dim=1)
        return actions, rewards

    @property
    def num_augment(self):
        return self.augmentation.num_augment


def get_automatic_batch_size(eval_fn, start_batch_size=8192, max_batch_size=4096):
    """Automatically reduces the batch size based on the eval function

    Args:
        eval_fn: The eval function
        start_batch_size: The starting batch size. This should be the theoretical maximum batch size
        max_batch_size: The maximum batch size. This is the practical maximum batch size
    """
    batch_size = start_batch_size

    effective_ratio = 1

    if hasattr(eval_fn, "num_starts"):
        batch_size = batch_size // (eval_fn.num_starts // 10)
        effective_ratio *= eval_fn.num_starts // 10
    if hasattr(eval_fn, "num_augment"):
        batch_size = batch_size // eval_fn.num_augment
        effective_ratio *= eval_fn.num_augment
    if hasattr(eval_fn, "samples"):
        batch_size = batch_size // eval_fn.samples
        effective_ratio *= eval_fn.samples

    batch_size = min(batch_size, max_batch_size)
    # get closest integer power of 2
    batch_size = 2 ** int(np.log2(batch_size))

    print(f"Effective batch size: {batch_size} (ratio: {effective_ratio})")

    return batch_size


def evaluate_policy(
        env,
        policy,
        dataset,
        method="greedy",
        batch_size=None,
        max_batch_size=4096,
        start_batch_size=8192,
        auto_batch_size=True,
        samples=1280,
        softmax_temp=1.0,
        num_augment=8,
        force_dihedral_8=True,
        **kwargs,
):
    num_loc = getattr(env.generator, "num_loc", None)

    methods_mapping = {
        "greedy": {"func": GreedyEval, "kwargs": {}},
        "sampling": {
            "func": SamplingEval,
            "kwargs": {"samples": samples, "softmax_temp": softmax_temp},
        },
        "multistart_greedy": {
            "func": GreedyMultiStartEval,
            "kwargs": {"num_starts": num_loc},
        },
        "augment_dihedral_8": {
            "func": AugmentationEval,
            "kwargs": {"num_augment": num_augment, "force_dihedral_8": force_dihedral_8},
        },
        "augment": {"func": AugmentationEval, "kwargs": {"num_augment": num_augment}},
        "multistart_greedy_augment_dihedral_8": {
            "func": GreedyMultiStartAugmentEval,
            "kwargs": {
                "num_augment": num_augment,
                "force_dihedral_8": force_dihedral_8,
                "num_starts": num_loc,
            },
        },
        "multistart_greedy_augment": {
            "func": GreedyMultiStartAugmentEval,
            "kwargs": {"num_augment": num_augment, "num_starts": num_loc},
        },
    }

    assert method in methods_mapping, "Method {} not found".format(method)

    # Set up the evaluation function
    eval_settings = methods_mapping[method]
    func, kwargs_ = eval_settings["func"], eval_settings["kwargs"]
    # subsitute kwargs with the ones passed in
    kwargs_.update(kwargs)
    kwargs = kwargs_
    eval_fn = func(env, **kwargs)

    if auto_batch_size:
        assert (
                batch_size is None
        ), "Cannot specify batch_size when auto_batch_size is True"
        batch_size = get_automatic_batch_size(
            eval_fn, max_batch_size=max_batch_size, start_batch_size=start_batch_size
        )
        print("Using automatic batch size: {}".format(batch_size))

    # Set up the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    # Run evaluation
    retvals = eval_fn(policy, dataloader)

    return retvals


if __name__ == "__main__":
    import argparse
    import importlib
    import os
    import pickle

    import torch

    from rlsolver.methods.eco_s2v.rl4co.envs import get_env

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--problem", type=str, default="tsp", help="Problem to solve")
    parser.add_argument(
        "--generator-params",
        type=dict,
        default={"num_loc": 50},
        help="Generator parameters for the environment",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/tsp/tsp50_test_seed1234.npz",
        help="Path of the test data npz file",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="AttentionModel",
        help="The class name of the valid model",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="checkpoints/am-tsp50.ckpt",
        help="The path of the checkpoint file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1", help="Device to run the evaluation"
    )

    # Evaluation
    parser.add_argument(
        "--method",
        type=str,
        default="greedy",
        help="Evaluation method, support 'greedy', 'sampling',\
                        'multistart_greedy', 'augment_dihedral_8', 'augment', 'multistart_greedy_augment_dihedral_8',\
                        'multistart_greedy_augment'",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top-p for sampling, from 0.0 to 1.0, 0.0 means not activated",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Top-k for sampling")
    parser.add_argument(
        "--select-best",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="During sampling, whether to select the best action, use --no-select_best to disable",
    )
    parser.add_argument(
        "--save-results",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to save the evaluation results",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="results",
        help="The root path to save the results",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1000,
        help="Number of instances to test, maximum 10000",
    )

    parser.add_argument(
        "--samples", type=int, default=1280, help="Number of samples for sampling method"
    )
    parser.add_argument(
        "--softmax-temp",
        type=float,
        default=1.0,
        help="Temperature for softmax in the sampling method",
    )
    parser.add_argument(
        "--num-augment",
        type=int,
        default=8,
        help="Number of augmentations for augmentation method",
    )
    parser.add_argument(
        "--force-dihedral-8",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Force the use of 8 augmentations for augmentation method",
    )

    opts = parser.parse_args()

    # Log the evaluation setting information
    print(f"Problem: {opts.problem}-{opts.generator_params['num_loc']}")
    print(f"Model: {opts.model}")
    print(f"Loading test instances from: {opts.data_path}")
    print(f"Loading model checkpoint from: {opts.ckpt_path}")
    print(f"Using the device: {opts.device}")
    print(f"Evaluation method: {opts.method}")
    print(f"Number of instances to test: {opts.num_instances}")

    if opts.method == "sampling":
        print(f"[Sampling] Number of samples: {opts.samples}")
        print(f"[Sampling] Temperature: {opts.temperature}")
        print(f"[Sampling] Top-p: {opts.top_p}")
        print(f"[Sampling] Top-k: {opts.top_k}")
        print(f"[Sampling] Softmax temperature: {opts.softmax_temp}")
        print(f"[Sampling] Select best: {opts.select_best}")

    if opts.method == "augment" or opts.method == "augment_dihedral_8":
        print(f"[Augmentation] Number of augmentations: {opts.num_augment}")
        print(f"[Augmentation] Force dihedral 8: {opts.force_dihedral_8}")

    if opts.save_results:
        print(f"Saving the results to: {opts.save_path}")
    else:
        print("[Warning] The result will not be saved!")

    # Init the environment
    env = get_env(opts.problem, generator_params=opts.generator_params)

    # Load the test data
    dataset = env.dataset(filename=opts.data_path)

    # Restrict the instances of testing
    dataset.data_len = min(opts.num_instances, len(dataset))

    # Load the model from checkpoint
    model_root = importlib.import_module("rl4co.models.zoo")
    model_cls = getattr(model_root, opts.model)
    model = model_cls.load_from_checkpoint(opts.ckpt_path, load_baseline=False)
    model = model.to(opts.device)

    # Evaluate
    result = evaluate_policy(
        env=env,
        policy=model.policy,
        dataset=dataset,
        method=opts.method,
        temperature=opts.temperature,
        top_p=opts.top_p,
        top_k=opts.top_k,
        samples=opts.samples,
        softmax_temp=opts.softmax_temp,
        num_augment=opts.num_augment,
        select_best=True,
        force_dihedral_8=True,
    )

    # Save the results
    if opts.save_results:
        if not os.path.exists(opts.save_path):
            os.makedirs(opts.save_path)
        save_fname = f"{env.name}{env.generator.num_loc}-{opts.model}-{opts.method}-temp-{opts.temperature}-top_p-{opts.top_p}-top_k-{opts.top_k}.pkl"
        save_path = os.path.join(opts.save_path, save_fname)
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
