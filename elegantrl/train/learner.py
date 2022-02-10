import multiprocessing as mp

import numpy as np
import torch

from elegantrl.train.utils import init_agent, init_replay_buffer, trajectory_to_device


class PipeLearner:
    def __init__(self, learner_gpus):

        self.learner_num = len(learner_gpus)
        self.round_num = int(np.log2(self.learner_num))

        self.pipes = [mp.Pipe() for _ in range(self.learner_num)]
        pipes = [mp.Pipe() for _ in range(self.learner_num)]
        self.pipe0s = [pipe[0] for pipe in pipes]
        self.pipe1s = [pipe[1] for pipe in pipes]
        self.device_list = [
            torch.device(f"cuda:{i}" if i >= 0 else "cpu") for i in learner_gpus
        ]

        if self.learner_num == 1:
            self.idx_l = None
        elif self.learner_num == 2:
            self.idx_l = [
                (1,),
                (0,),
            ]
        elif self.learner_num == 4:
            self.idx_l = [
                (1, 2),
                (0, 3),
                (3, 0),
                (2, 1),
            ]
        elif self.learner_num == 8:
            self.idx_l = [
                (1, 2, 4),
                (0, 3, 5),
                (3, 0, 6),
                (2, 1, 7),
                (5, 6, 0),
                (4, 7, 1),
                (7, 4, 2),
                (6, 5, 3),
            ]
        else:
            print(
                f"| LearnerPipe, ERROR: learner_num  {self.learner_num} should in (1, 2, 4, 8)"
                f"| LearnerPipe, ERROR: learner_gpus {repr(learner_gpus)}"
            )
            exit()

    def comm_data(self, data, learner_id, round_id):
        learner_jd = self.idx_l[learner_id][round_id]
        if round_id == -1:
            data = [trajectory_to_device(item, torch.device("cpu")) for item in data]
            self.pipes[learner_jd][0].send(data)
            return self.pipes[learner_id][1].recv()
        else:
            self.pipe0s[learner_jd].send(data)
            return self.pipe1s[learner_id].recv()

    def comm_network_optim(self, agent, learner_id):
        device = self.device_list[learner_id]

        for round_id in range(self.round_num):
            data = self.get_comm_data(agent)
            data = self.comm_data(data, learner_id, round_id)

            if data:
                self.average_param(agent.act.parameters(), data[0], device)
                self.average_param(
                    agent.act_optim.parameters(), data[1], device
                ) if data[1] else None

                self.average_param(agent.cri.parameters(), data[2], device) if data[
                    2
                ] else None
                self.average_param(agent.cri_optim.parameters(), data[3], device)

                self.average_param(
                    agent.act_target.parameters(), data[4], device
                ) if agent.if_use_act_target else None
                self.average_param(
                    agent.cri_target.parameters(), data[5], device
                ) if agent.if_use_cri_target else None

    def run(self, args, comm_eva, comm_exp, learner_id):
        gpu_id = args.learner_gpus[learner_id]

        agent = init_agent(args, gpu_id=gpu_id, env=None)
        buffer, update_buffer = init_replay_buffer(args, gpu_id, agent=None, env=None)

        """start training"""
        cwd = args.cwd
        batch_size = args.batch_size
        repeat_times = args.repeat_times
        soft_update_tau = args.soft_update_tau
        del args

        torch.set_grad_enabled(False)
        if_train = True
        while if_train:
            traj_lists = comm_exp.explore(agent)
            # if self.learner_num > 1:
            #     data = self.comm_data(traj_lists, learner_id, round_id=-1)
            #     traj_lists.extend(data)
            traj_list = sum(traj_lists, [])

            steps, r_exp = update_buffer(traj_list)

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(
                buffer, batch_size, repeat_times, soft_update_tau
            )
            torch.set_grad_enabled(False)
            if self.learner_num > 1:
                self.comm_network_optim(agent, learner_id)

            if comm_eva:
                if_train, if_save = comm_eva.evaluate_and_save_mp(
                    agent, steps, r_exp, logging_tuple, cwd
                )

        agent.save_or_load_agent(cwd, if_save=True)

        if hasattr(buffer, "save_or_load_history"):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)

    @staticmethod
    def get_comm_data(agent):
        act = list(agent.act.parameters())
        cri_optim_param = agent.cri_optim.parameters()

        if agent.cri is agent.act:
            cri = None
            act_optim_param = None
        else:
            cri = list(agent.cri.parameters())
            act_optim_param = agent.act_optim.parameters()

        act_target = (
            list(agent.act_target.parameters()) if agent.if_use_act_target else None
        )
        cri_target = (
            list(agent.cri_target.parameters()) if agent.if_use_cri_target else None
        )
        return (
            act,
            act_optim_param,
            cri,
            cri_optim_param,
            act_target,
            cri_target,
        )  # data

    @staticmethod
    def average_param(dst_optim_param, src_optim_param, device):
        for dst, src in zip(dst_optim_param, src_optim_param):
            dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
            # dst.data.copy_(src.data * tau + dst.data * (1 - tau))
