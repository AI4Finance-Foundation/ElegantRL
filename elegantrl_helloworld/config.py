import os
import torch
import numpy as np


class Arguments:
    def __init__(self, agent_class, env_func=None, env_args=None):
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)

        self.env_num = self.env_args['env_num']  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.env_args['max_step']  # the max step of an episode
        self.env_name = self.env_args['env_name']  # the env name. Be used to set 'cwd'.
        self.state_dim = self.env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = self.env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = self.env_args['if_discrete']  # discrete or continuous action space

        self.agent_class = agent_class  # DRL algorithm
        self.net_dim = 2 ** 7  # the middle layer dimension of Fully Connected Network
        self.num_layer = 3  # the layer number of MultiLayer Perceptron, `assert num_layer >= 2`
        self.batch_size = 2 ** 5  # num of transitions sampled from replay buffer.
        self.if_off_policy = self.get_if_off_policy()  # agent is on-policy or off-policy
        if self.if_off_policy:  # off-policy
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.max_capacity = 2 ** 21  # if reach the capacity of ReplayBuffer, first in first out for off-policy.
            self.repeat_times = 2 ** 0  # repeatedly update network using ReplayBuffer to keep critic's loss small
        else:  # on-policy
            self.target_step = 2 ** 12  # collect target_step, then update network
            self.max_capacity = self.target_step  # capacity of ReplayBuffer. Empty the ReplayBuffer for on-policy.
            self.repeat_times = 2 ** 3  # repeatedly update network using ReplayBuffer to keep critic's loss small

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        '''Arguments for device'''
        self.thread_num = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 42  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times = 2 ** 4  # number of times that get episode return

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        if self.cwd is None:  # set cwd (current working directory)
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}_{self.learner_gpus}'

        if self.if_remove is None:  # remove history
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self):
        name = self.agent_class.__name__
        return all((name.find('PPO') == -1, name.find('A2C') == -1))  # if_off_policy
