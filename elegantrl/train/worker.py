import multiprocessing as mp
import sys

import torch

from elegantrl.train.config import build_env
from elegantrl.train.utils import act_dict_to_device, trajectory_to_device
from elegantrl.train.utils import init_agent


class PipeWorker:
    def __init__(self, env_num, worker_num):
        self.env_num = env_num
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent):
        act_dict = agent.act.state_dict()
        act_dict_to_device(
            act_dict, torch.device("cpu")
        ) if sys.platform == "win32" else None
        # Avoid CUDA runtime error (801). WinOS Python<3.9, pipe can't send torch.tensor_gpu, but tensor_cpu can.

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        return [pipe1.recv() for pipe1 in self.pipe1s]  # traj_lists

    def run(self, args, worker_id, learner_id):
        gpu_id = args.learner_gpus[learner_id]

        """init Agent"""
        env = build_env(
            env=args.env, env_func=args.env_func, env_args=args.env_args, gpu_id=gpu_id
        )
        agent = init_agent(args, gpu_id=gpu_id, env=env)

        """loop"""
        target_step = args.target_step
        del args

        torch.set_grad_enabled(False)
        while True:
            act_dict = self.pipes[worker_id][0].recv()
            act_dict_to_device(
                act_dict, agent.device
            ) if sys.platform == "win32" else None
            # Avoid CUDA runtime error (801). WinOS Python<3.9, pipe can't send torch.tensor_gpu, but tensor_cpu can.

            agent.act.load_state_dict(act_dict)

            trajectory = agent.explore_env(env, target_step)
            trajectory_to_device(
                trajectory, torch.device("cpu")
            ) if sys.platform == "win32" else None
            # Avoid CUDA runtime error (801). WinOS Python<3.9, pipe can't send torch.tensor_gpu, but tensor_cpu can.

            self.pipes[worker_id][0].send(trajectory)
