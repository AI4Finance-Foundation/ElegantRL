import abc
from os.path import join as pjoin
from typing import Iterable, Optional

import torch
from tensordict.tensordict import TensorDict
from torchrl.envs import EnvBase

from rlsolver.methods.eco_s2v.rl4co.data.dataset import TensorDictDataset
from rlsolver.methods.eco_s2v.rl4co.data.utils import load_npz_to_tensordict
from rlsolver.methods.eco_s2v.rl4co.utils.ops import get_num_starts, select_start_nodes
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class RL4COEnvBase(EnvBase, metaclass=abc.ABCMeta):
    """Base class for RL4CO environments based on TorchRL EnvBase.
    The environment has the usual methods for stepping, resetting, and getting the specifications of the environment
    that shoud be implemented by the subclasses of this class.
    It also has methods for getting the reward, action mask, and checking the validity of the solution, and
    for generating and loading the datasets (supporting multiple dataloaders as well for validation and testing).

    Args:
        data_dir: Root directory for the dataset
        train_file: Name of the training file
        val_file: Name of the validation file
        test_file: Name of the test file
        val_dataloader_names: Names of the dataloaders to use for validation
        test_dataloader_names: Names of the dataloaders to use for testing
        check_solution: Whether to check the validity of the solution at the end of the episode
        dataset_cls: Dataset class to use for the environment (which can influence performance)
        seed: Seed for the environment
        device: Device to use. Generally, no need to set as tensors are updated on the fly
        batch_size: Batch size to use for the environment. Generally, no need to set as tensors are updated on the fly
        run_type_checks: If True, run type checks on the TensorDicts at each step
        allow_done_after_reset: If True, an environment can be done after a reset
        _torchrl_mode: Whether to use the TorchRL mode (see :meth:`step` for more details)
    """

    batch_locked = False

    def __init__(
            self,
            *,
            data_dir: str = "data/",
            train_file: str = None,
            val_file: str = None,
            test_file: str = None,
            val_dataloader_names: list = None,
            test_dataloader_names: list = None,
            check_solution: bool = True,
            dataset_cls: callable = TensorDictDataset,
            seed: int = None,
            device: str = "cpu",
            batch_size: torch.Size = None,
            run_type_checks: bool = False,
            allow_done_after_reset: bool = False,
            _torchrl_mode: bool = False,
            **kwargs,
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            run_type_checks=run_type_checks,
            allow_done_after_reset=allow_done_after_reset,
        )
        # if any kwargs are left, we want to warn the user
        kwargs.pop("name", None)  # we remove the name for checking
        if kwargs:
            log.error(
                f"Unused keyword arguments: {', '.join(kwargs.keys())}. "
                "Please check the base class documentation at https://rl4co.readthedocs.io/en/latest/_content/api/envs/base.html. "
                "In case you would like to pass data generation arguments, please pass a `generator` method instead "
                "or for example: `generator_kwargs=dict(num_loc=50)` to the constructor."
            )
        self.data_dir = data_dir
        self.train_file = pjoin(data_dir, train_file) if train_file is not None else None
        self._torchrl_mode = _torchrl_mode
        self.dataset_cls = dataset_cls

        def get_files(f):
            if f is not None:
                if isinstance(f, Iterable) and not isinstance(f, str):
                    return [pjoin(data_dir, _f) for _f in f]
                else:
                    return pjoin(data_dir, f)
            return None

        def get_multiple_dataloader_names(f, names):
            if f is not None:
                if isinstance(f, Iterable) and not isinstance(f, str):
                    if names is None:
                        names = [f"{i}" for i in range(len(f))]
                    else:
                        assert len(names) == len(
                            f
                        ), "Number of dataloader names must match number of files"
                else:
                    if names is not None:
                        log.warning(
                            "Ignoring dataloader names since only one dataloader is provided"
                        )
            return names

        self.val_file = get_files(val_file)
        self.test_file = get_files(test_file)
        self.val_dataloader_names = get_multiple_dataloader_names(
            self.val_file, val_dataloader_names
        )
        self.test_dataloader_names = get_multiple_dataloader_names(
            self.test_file, test_dataloader_names
        )
        self.check_solution = check_solution
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def step(self, td: TensorDict) -> TensorDict:
        """Step function to call at each step of the episode containing an action.
        If `_torchrl_mode` is True, we call `_torchrl_step` instead which set the
        `next` key of the TensorDict to the next state - this is the usual way to do it in TorchRL,
        but inefficient in our case
        """
        if not self._torchrl_mode:
            # Default: just return the TensorDict without farther checks etc is faster
            td = self._step(td)
            return {"next": td}
        else:
            # Since we simplify the syntax
            return self._torchrl_step(td)

    def reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        """Reset function to call at the beginning of each episode"""
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size
        if td is None or td.is_empty():
            td = self.generator(batch_size=batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.to(td.device)
        return super().reset(td, batch_size=batch_size)

    def _torchrl_step(self, td: TensorDict) -> TensorDict:
        """See :meth:`super().step` for more details.
        This is the usual way to do it in TorchRL, but inefficient in our case

        Note:
            Here we clone the TensorDict to avoid recursion error, since we allow
            for directly updating the TensorDict in the step function
        """
        # sanity check
        self._assert_tensordict_shape(td)
        next_preset = td.get("next", None)

        next_tensordict = self._step(
            td.clone()
        )  # NOTE: we clone to avoid recursion error
        next_tensordict = self._step_proc_data(next_tensordict)
        if next_preset is not None:
            next_tensordict.update(next_preset.exclude(*next_tensordict.keys(True, True)))
        td.set("next", next_tensordict)
        return td

    @abc.abstractmethod
    def _step(self, td: TensorDict) -> TensorDict:
        """Step function to call at each step of the episode containing an action.
        Gives the next observation, reward, done
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        """Reset function to call at the beginning of each episode"""
        raise NotImplementedError

    def _make_spec(self, td_params: TensorDict = None):
        """Make the specifications of the environment (observation, action, reward, done)"""
        raise NotImplementedError

    def get_reward(
            self, td: TensorDict, actions: torch.Tensor, check_solution: Optional[bool] = None
    ) -> torch.Tensor:
        """Function to compute the reward. Can be called by the agent to compute the reward of the current state
        This is faster than calling step() and getting the reward from the returned TensorDict at each time for CO tasks
        """
        # Fallback to env setting if not assigned
        check_solution = self.check_solution if check_solution is None else check_solution
        if check_solution:
            self.check_solution_validity(td, actions)
        return self._get_reward(td, actions)

    @abc.abstractmethod
    def _get_reward(self, td, actions) -> TensorDict:
        """Function to compute the reward. Can be called by the agent to compute the reward of the current state
        This is faster than calling step() and getting the reward from the returned TensorDict at each time for CO tasks
        """
        raise NotImplementedError

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Function to compute the action mask (feasible actions) for the current state
        Action mask is 1 if the action is feasible, 0 otherwise
        """
        raise NotImplementedError

    def get_num_starts(self, td):
        return get_num_starts(td, self.name)

    def select_start_nodes(self, td, num_starts):
        return select_start_nodes(td, self, num_starts)

    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        """Function to check whether the solution is valid. Can be called by the agent to check the validity of the current state
        This is called with the full solution (i.e. all actions) at the end of the episode
        """
        raise NotImplementedError

    def replace_selected_actions(
            self,
            cur_actions: torch.Tensor,
            new_actions: torch.Tensor,
            selection_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace selected current actions with updated actions based on `selection_mask`.
        """
        raise NotImplementedError

    def local_search(
            self, td: TensorDict, actions: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Function to improve the solution. Can be called by the agent to improve the current state
        This is called with the full solution (i.e. all actions) at the end of the episode
        """
        raise NotImplementedError(
            f"Local is not implemented yet for {self.name} environment"
        )

    def dataset(self, batch_size=[], phase="train", filename=None):
        """Return a dataset of observations
        Generates the dataset if it does not exist, otherwise loads it from file
        """
        if filename is not None:
            log.info(f"Overriding dataset filename from {filename}")
        f = getattr(self, f"{phase}_file") if filename is None else filename
        if f is None:
            if phase != "train":
                log.warning(f"{phase}_file not set. Generating dataset instead")
            td = self.generator(batch_size)
        else:
            log.info(f"Loading {phase} dataset from {f}")
            if phase == "train":
                log.warning(
                    "Loading training dataset from file. This may not be desired in RL since "
                    "the dataset is fixed and the agent will not be able to explore new states"
                )
            try:
                if isinstance(f, Iterable) and not isinstance(f, str):
                    names = getattr(self, f"{phase}_dataloader_names")
                    return {
                        name: self.dataset_cls(self.load_data(_f, batch_size))
                        for name, _f in zip(names, f)
                    }
                else:
                    td = self.load_data(f, batch_size)
            except FileNotFoundError:
                log.error(
                    f"Provided file name {f} not found. Make sure to provide a file in the right path first or "
                    f"unset {phase}_file to generate data automatically instead"
                )
                td = self.generator(batch_size)

        return self.dataset_cls(td)

    def transform(self):
        """Used for converting TensorDict variables (such as with torch.cat) efficiently
        https://pytorch.org/rl/reference/generated/torchrl.envs.transforms.Transform.html
        By default, we do not need to transform the environment since we use specific embeddings
        """
        return self

    def render(self, *args, **kwargs):
        """Render the environment"""
        raise NotImplementedError

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file"""
        return load_npz_to_tensordict(fpath)

    def _set_seed(self, seed: Optional[int]):
        """Set the seed for the environment"""
        rng = torch.manual_seed(seed)
        self.rng = rng

    def to(self, device):
        """Override `to` device method for safety against `None` device (may be found in `TensorDict`)"""
        if device is None:
            return self
        else:
            return super().to(device)

    @staticmethod
    def solve(
            instances: TensorDict,
            max_runtime: float,
            num_procs: int = 1,
            **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Classical solver for the environment. This is a wrapper for the baselines solver.

        Args:
            instances: The instances to solve
            max_runtime: The maximum runtime for the solver
            num_procs: The number of processes to use

        Returns:
            A tuple containing the action and the cost, respectively
        """
        raise NotImplementedError

    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        state = self.__dict__.copy()
        state["rng"] = state["rng"].get_state()
        return state

    def __setstate__(self, state):
        """Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        self.__dict__.update(state)
        self.rng = torch.manual_seed(0)
        self.rng.set_state(state["rng"].to('cpu'))
        # self.rng.set_state(state["rng"])


class ImprovementEnvBase(RL4COEnvBase, metaclass=abc.ABCMeta):
    """Base class for Improvement environments based on RL4CO EnvBase.
    Note that this class assumes that the solution is stored in a linked list format.
    Here, if `rec[i] = j`, it means the node `i` is connected to node `j`, i.e., edge `i-j` is in the solution.
    For example, if edge `0-1`, edge `1-5`, edge `2-10` are in the solution, so we have `rec[0]=1`, `rec[1]=5` and `rec[2]=10`.
    Kindly see https://github.com/yining043/VRP-DACT/blob/new_version/Play_with_DACT.ipynb for an example at the end for TSP.
    """

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def _step(self, td: TensorDict, solution_to=None) -> TensorDict:
        raise NotImplementedError

    def step_to_solution(self, td, solution) -> TensorDict:
        return self._step(td, solution_to=solution)

    @staticmethod
    def _get_reward(td, actions) -> TensorDict:
        raise NotImplementedError(
            "This function is not used for improvement tasks since the reward is computed per step"
        )

    @staticmethod
    def get_costs(coordinates, rec):
        batch_size, size = rec.size()

        # calculate the route length value
        d1 = coordinates.gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
        d2 = coordinates
        length = (d1 - d2).norm(p=2, dim=2).sum(1)

        return length

    @staticmethod
    def _get_real_solution(rec):
        batch_size, seq_length = rec.size()
        visited_time = torch.zeros((batch_size, seq_length)).to(rec.device)
        pre = torch.zeros((batch_size), device=rec.device).long()
        for i in range(seq_length):
            visited_time[torch.arange(batch_size), rec[torch.arange(batch_size), pre]] = (
                    i + 1
            )
            pre = rec[torch.arange(batch_size), pre]

        visited_time = visited_time % seq_length
        return visited_time.argsort()

    @staticmethod
    def _get_linked_list_solution(solution):
        solution_pre = solution
        solution_post = torch.cat((solution[:, 1:], solution[:, :1]), 1)

        rec = solution.clone()
        rec.scatter_(1, solution_pre, solution_post)
        return rec

    @classmethod
    def get_best_solution(cls, td):
        return cls._get_real_solution(td["rec_best"])

    @classmethod
    def get_current_solution(cls, td):
        return cls._get_real_solution(td["rec_current"])
