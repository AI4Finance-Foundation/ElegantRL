import copy
import math
import random

import lightning.pytorch as pl
import torch
from lightning import Callback
from rlsolver.methods.rl4co_maxcut import utils
from torch.optim import Adam

log = utils.get_pylogger(__name__)


class ReptileCallback(Callback):
    """ Meta training framework for addressing the generalization issue (implement the Reptile algorithm only)
    Based on Manchanda et al. 2022 (https://arxiv.org/abs/2206.00787) and Zhou et al. 2023 (https://arxiv.org/abs/2305.19587)

    Args:
        - num_tasks: the number of tasks in a mini-batch, i.e. `B` in the original paper
        - alpha: initial weight of the task model for the outer-loop optimization of reptile
        - alpha_decay: weight decay of the task model for the outer-loop optimization of reptile
        - min_size: minimum problem size of the task (only supported in cross-size generalization)
        - max_size: maximum problem size of the task (only supported in cross-size generalization)
        - sch_bar: for the task scheduler of size setting, where lr_decay_epoch = sch_bar * epochs, i.e. after this epoch, learning rate will decay with a weight 0.1
        - data_type: type of the tasks, chosen from ["size", "distribution", "size_distribution"]
        - print_log: whether to print the specific task sampled in each inner-loop optimization
    """

    def __init__(self,
                 num_tasks: int,
                 alpha: float,
                 alpha_decay: float,
                 min_size: int,
                 max_size: int,
                 sch_bar: float = 0.9,
                 data_type: str = "size",
                 print_log: bool = True):

        super().__init__()

        self.num_tasks = num_tasks
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.sch_bar = sch_bar
        self.print_log = print_log
        self.data_type = data_type
        self.task_set = self._generate_task_set(data_type, min_size, max_size)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        # Sample a batch of tasks
        self._sample_task()

        # Pre-set the distribution
        if self.data_type == "size_distribution":
            pl_module.env.generator.loc_distribution = "gaussian_mixture"
            self.selected_tasks[0] = (pl_module.env.generator.num_loc, 0, 0)
        elif self.data_type == "size":
            pl_module.env.generator.loc_distribution = "uniform"
            self.selected_tasks[0] = (pl_module.env.generator.num_loc,)
        elif self.data_type == "distribution":
            pl_module.env.generator.loc_distribution = "gaussian_mixture"
            self.selected_tasks[0] = (0, 0)
        self.task_params = self.selected_tasks[0]

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        # Alpha scheduler (decay for the update of meta model)
        self._alpha_scheduler()

        # Reinitialize the task model with the parameters of the meta model
        if trainer.current_epoch % self.num_tasks == 0:  # Save the meta model
            self.meta_model_state_dict = copy.deepcopy(pl_module.state_dict())
            self.task_models = []
            # Print sampled tasks
            if self.print_log:
                print('\n>> Meta epoch: {} (Exact epoch: {}), Training task: {}'.format(trainer.current_epoch // self.num_tasks, trainer.current_epoch, self.selected_tasks))
        else:
            pl_module.load_state_dict(self.meta_model_state_dict)

        # Reinitialize the optimizer every epoch
        lr_decay = 0.1 if trainer.current_epoch + 1 == int(self.sch_bar * trainer.max_epochs) else 1
        old_lr = trainer.optimizers[0].param_groups[0]['lr']
        new_optimizer = Adam(pl_module.parameters(), lr=old_lr * lr_decay)
        trainer.optimizers = [new_optimizer]

        # Print
        if self.print_log:
            if hasattr(pl_module.env.generator, 'capacity'):
                print('>> Training task: {}, capacity: {}'.format(self.task_params, pl_module.env.generator.capacity))
            else:
                print('>> Training task: {}'.format(self.task_params))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        # Save the task model
        self.task_models.append(copy.deepcopy(pl_module.state_dict()))
        if (trainer.current_epoch + 1) % self.num_tasks == 0:
            # Outer-loop optimization (update the meta model with the parameters of the task model)
            with torch.no_grad():
                state_dict = {params_key: (self.meta_model_state_dict[params_key] +
                                           self.alpha * torch.mean(torch.stack([fast_weight[params_key] - self.meta_model_state_dict[params_key]
                                                                                for fast_weight in self.task_models], dim=0).float(), dim=0))
                              for params_key in self.meta_model_state_dict}
                pl_module.load_state_dict(state_dict)

        # Get ready for the next meta-training iteration
        if (trainer.current_epoch + 1) % self.num_tasks == 0:
            # Sample a batch of tasks
            self._sample_task()

        # Load new training task (Update the environment) for the next meta-training iteration
        self._load_task(pl_module, task_idx=(trainer.current_epoch + 1) % self.num_tasks)

    def _sample_task(self):

        # Sample a batch of tasks
        self.selected_tasks = []
        for b in range(self.num_tasks):
            task_params = random.sample(self.task_set, 1)[0]
            self.selected_tasks.append(task_params)

    def _load_task(self, pl_module: pl.LightningModule, task_idx=0):

        # Load new training task (Update the environment)
        self.task_params = self.selected_tasks[task_idx]

        if self.data_type == "size_distribution":
            assert len(self.task_params) == 3
            pl_module.env.generator.num_loc = self.task_params[0]
            pl_module.env.generator.num_modes = self.task_params[1]
            pl_module.env.generator.cdist = self.task_params[2]
        elif self.data_type == "distribution":  # fixed size
            assert len(self.task_params) == 2
            pl_module.env.generator.num_modes = self.task_params[0]
            pl_module.env.generator.cdist = self.task_params[1]
        elif self.data_type == "size":  # fixed distribution
            assert len(self.task_params) == 1
            pl_module.env.generator.num_loc = self.task_params[0]

        if hasattr(pl_module.env.generator, 'capacity') and self.data_type in ["size_distribution", "size"]:
            task_capacity = math.ceil(30 + self.task_params[0] / 5) if self.task_params[0] >= 20 else 20
            pl_module.env.generator.capacity = task_capacity

    def _alpha_scheduler(self):
        self.alpha = max(self.alpha * self.alpha_decay, 0.0001)

    def _generate_task_set(self, data_type, min_size, max_size):
        """
        Following the setting in Zhou et al. 2023 (https://arxiv.org/abs/2305.19587)
        Current setting:
            size: (n,) \in [20, 150]
            distribution: (m, c) \in {(0, 0) + [1-9] * [1, 10, 20, 30, 40, 50]}
            size_distribution: (n, m, c) \in [50, 200, 5] * {(0, 0) + (1, 1) + [3, 5, 7] * [10, 30, 50]}
        """

        if data_type == "distribution":  # focus on TSP100 with gaussian mixture distributions
            task_set = [(0, 0)] + [(m, c) for m in range(1, 10) for c in [1, 10, 20, 30, 40, 50]]
        elif data_type == "size":  # focus on uniform distribution with different sizes
            task_set = [(n,) for n in range(min_size, max_size + 1)]
        elif data_type == "size_distribution":
            dist_set = [(0, 0), (1, 1)] + [(m, c) for m in [3, 5, 7] for c in [10, 30, 50]]
            task_set = [(n, m, c) for n in range(50, 201, 5) for (m, c) in dist_set]
        else:
            raise NotImplementedError

        print(">> Generating training task set: {} tasks with type {}".format(len(task_set), data_type))
        print(">> Training task set: {}".format(task_set))

        return task_set
