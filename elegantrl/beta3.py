import os
import time
import torch
import numpy as np
import numpy.random as rd

import multiprocessing as mp

from elegantrl.train.replay_buffer import ReplayBufferMP
from elegantrl.train.run_parallel import PipeWorker
from elegantrl.train.run_parallel import process_safely_terminate, get_step_r_exp
from elegantrl.train.evaluator import get_episode_return_and_step
from elegantrl.train.config import Arguments

from elegantrl.envs.Gym import build_env

from elegantrl.agents.AgentPPO import AgentPPO, AgentModPPO
from elegantrl.agents.AgentA2C import AgentA2C

'''run.py'''


def train_and_evaluate_em(args):  # keep
    args.init_before_training()  # necessary!

    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    # todo ensemble
    assert hasattr(args.learner_gpus, '__iter__')
    assert hasattr(args.learner_gpus[0], '__iter__')
    ensemble_num = len(args.learner_gpus)

    for agent_id in range(ensemble_num):
        # todo ensemble
        args.eval_gpu_id = args.learner_gpus[agent_id][0]
        args.random_seed += agent_id * len(args.learner_gpus[agent_id]) * args.worker_num
        os.makedirs(args.cwd, exist_ok=True)

        '''learner'''
        learner_num = len(args.learner_gpus[agent_id])
        learner_pipe = PipeLearner(learner_num)

        learner_id = args.learner_gpus[agent_id][0]
        '''worker'''
        worker_pipe = PipeWorker(args.env_num, args.worker_num)
        for worker_id in range(args.worker_num):
            proc = mp.Process(target=worker_pipe.run, args=(args, worker_id, learner_id))
            proc.start()
            process.append(proc)

        proc = mp.Process(target=learner_pipe.run, args=(args, worker_pipe, learner_id, agent_id))
        proc.start()
        process.append(proc)

    '''evaluator'''
    evaluator_en = PipeEvaluator(args)
    proc = mp.Process(target=evaluator_en.run, args=(args,))
    proc.start()
    process.append(proc)

    # [(p.start(), time.sleep(0.1)) for p in process]
    process[-1].join()
    process_safely_terminate(process)


class Ensemble:
    def __init__(self, ensemble_gap, ensemble_num, agent_id):
        self.ensemble_gap = ensemble_gap
        self.ensemble_num = ensemble_num
        self.ensemble_timer = time.time() + ensemble_gap * agent_id / ensemble_num
        self.agent_id = agent_id

    def run(self, cwd, agent):
        if self.ensemble_timer + self.ensemble_gap > time.time():
            return
        self.ensemble_timer = time.time()

        save_path = f"{cwd}/pod_temp_{self.agent_id:04}"
        os.makedirs(save_path, exist_ok=True)
        agent.save_or_load_agent(save_path, if_save=True)

        load_path = self.get_load_path_randomly(cwd)
        if load_path:
            agent.traj_list[:] = [list() for _ in agent.traj_list]
            agent.save_or_load_agent(load_path, if_save=False)

    def get_load_path_randomly(self, cwd):
        names = [name for name in os.listdir(cwd) if name.find('pod_eval_') == 0]
        if len(names):
            rewards = np.array([float(name[9:]) for name in names])
            rewards_soft_max = self.np_soft_max(rewards)
            name_id = rd.choice(len(rewards_soft_max), p=rewards_soft_max)

            load_path = f"{cwd}/pod_eval_{names[name_id]:8.3f}"
        else:
            load_path = None
        return load_path

    @staticmethod
    def np_soft_max(raw_x):
        norm_x = (raw_x - raw_x.mean()) / (raw_x.std() + 1e-6)
        exp_x = np.exp(norm_x) + 1e-6
        return exp_x / exp_x.sum()


class PipeLearner:
    def __init__(self, learner_num):
        self.learner_num = learner_num

    def run(self, args, comm_exp, learner_id=0, agent_id=0):
        # print(f'| os.getpid()={os.getpid()} PipeLearn.run, {learner_id}')
        pass

        '''init Agent'''
        agent = args.agent
        agent.init(net_dim=args.net_dim, state_dim=args.state_dim, action_dim=args.action_dim,
                   gamma=args.gamma, reward_scale=args.reward_scale,
                   learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae,
                   env_num=args.env_num, gpu_id=args.learner_gpus[learner_id], )
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init ReplayBuffer'''
        if agent.if_off_policy:
            buffer_num = args.worker_num * args.env_num
            if self.learner_num > 1:
                buffer_num *= 2

            buffer = ReplayBufferMP(max_len=args.max_memo, state_dim=args.state_dim,
                                    action_dim=1 if args.if_discrete else args.action_dim,
                                    if_use_per=args.if_per_or_gae,
                                    buffer_num=buffer_num, gpu_id=args.learner_gpus[learner_id])
            buffer.save_or_load_history(args.cwd, if_save=False)

            def update_buffer(_traj_list):
                step_sum = 0
                r_exp_sum = 0
                for buffer_i, (ten_state, ten_other) in enumerate(_traj_list):
                    buffer.buffers[buffer_i].extend_buffer(ten_state, ten_other)

                    step_r_exp = get_step_r_exp(ten_reward=ten_other[:, 0])  # other = (reward, mask, action)
                    step_sum += step_r_exp[0]
                    r_exp_sum += step_r_exp[1]
                return step_sum, r_exp_sum / len(_traj_list)
        else:
            buffer = list()

            def update_buffer(_traj_list):
                _traj_list = list(map(list, zip(*_traj_list)))
                _traj_list = [torch.cat(t, dim=0) for t in _traj_list]
                (ten_state, ten_reward, ten_mask, ten_action, ten_noise) = _traj_list
                buffer[:] = (ten_state.squeeze(1),
                             ten_reward,
                             ten_mask,
                             ten_action.squeeze(1),
                             ten_noise.squeeze(1))

                _step, _r_exp = get_step_r_exp(ten_reward=buffer[1])
                return _step, _r_exp

        '''start training'''
        ensemble = Ensemble(ensemble_gap=args.ensemble_gap,
                            ensemble_num=len(args.ensemble_gpus),
                            agent_id=agent_id)  # todo ensemble

        cwd = args.cwd
        batch_size = args.batch_size
        repeat_times = args.repeat_times
        soft_update_tau = args.soft_update_tau
        eval_gap = args.eval_gap
        del args

        total_step = 0
        r_max = -np.inf
        eval_timer = time.time()

        if_train = True
        while if_train:
            ensemble.run(cwd, agent)  # todo ensemble

            traj_lists = comm_exp.explore(agent)
            traj_list = sum(traj_lists, list())
            del traj_lists

            steps, r_exp = update_buffer(traj_list)
            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

            total_step += steps

            if eval_timer + eval_gap < time.time():
                eval_timer = time.time()

                s_avg = np.mean(steps)
                s_std = np.std(steps)
                r_avg = 0
                r_std = 0
                r_max = max(r_max, r_exp)
                print(f"{agent_id:<3}{total_step:8.2e}{r_max:8.2f} |"
                      f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                      f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in logging_tuple)}")

        agent.save_or_load_agent(cwd, if_save=True)
        if agent.if_off_policy:
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)


class PipeEvaluator:  # [ElegantRL.2021.12.02]
    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self, args):
        torch.set_grad_enabled(False)  # with torch.no_grad():
        # print(f'| os.getpid()={os.getpid()} PipeEvaluate.run {agent_id}')
        pass
        dir(self.args)

        '''init: Agent'''
        agent = args.agent
        agent.init(net_dim=args.net_dim, state_dim=args.state_dim, action_dim=args.action_dim,
                   gamma=args.gamma, reward_scale=args.reward_scale,
                   learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae,
                   env_num=args.env_num, gpu_id=args.eval_gpu_id, )
        agent.save_or_load_agent(args.cwd, if_save=False)
        act = agent.act  # [setattr(param, 'requires_grad', False) for param in agent.act.parameters()]
        del agent

        '''init Evaluator'''
        eval_env = build_env(env=args.eval_env, device_id=args.eval_gpu_id,
                             env_class=args.eval_env_class, env_args=args.eval_env_args, env_info=args.eval_env_info)

        '''loop'''
        cwd = args.cwd
        ensemble_num = len(args.learner_gpus)
        eval_times2 = args.eval_times2
        del args

        from itertools import cycle
        for agent_id in cycle(range(ensemble_num)):
            load_path = f"{cwd}/pod_temp_{agent_id:04}/actor.pth"
            if not os.path.isfile(load_path):
                time.sleep(1)
                continue

            act.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))

            rewards = [get_episode_return_and_step(eval_env, act)
                       for _ in range(eval_times2)]
            r_avg = np.mean(rewards)

            os.rename(src=f"{cwd}/pod_temp_{agent_id:04}",
                      dst=f"{cwd}/pod_eval_{r_avg:09.3f}")


'''run'''


def demo_continuous_action_on_policy():  # [ElegantRL.2021.11.11]
    env_name = ['Pendulum-v1', 'LunarLanderContinuous-v2',
                'BipedalWalker-v3', 'BipedalWalkerHardcore-v3'][ENV_ID]
    agent_class = [AgentA2C, AgentPPO, AgentModPPO][DRL_ID]
    from elegantrl.envs.Gym import GymEnv
    args = Arguments(env=GymEnv(env_name), agent=agent_class())
    # args.if_per_or_gae = True

    if env_name in {'Pendulum-v1', 'Pendulum-v0'}:
        """
        Step 45e4,  Reward -138,  UsedTime 373s PPO
        Step 40e4,  Reward -200,  UsedTime 400s PPO
        Step 46e4,  Reward -213,  UsedTime 300s PPO
        """
        # args = Arguments(env=build_env(env_name), agent=agent_class())  # One way to build env
        # args = Arguments(env=env_name, agent=agent_class())  # Another way to build env
        # args.env_num = 1
        # args.max_step = 200
        # args.state_dim = 3
        # args.action_dim = 1
        # args.if_discrete = False

        args.gamma = 0.97
        args.net_dim = 2 ** 8
        args.worker_num = 2
        args.reward_scale = 2 ** -2
        args.target_step = 200 * 16  # max_step = 200

        args.eval_gap = 2 ** 5
    if env_name in {'LunarLanderContinuous-v2', 'LunarLanderContinuous-v1'}:
        """
        ################################################################################
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        0  1.58e+04 -125.13 | -125.13   45.9     68    12 |   -1.64   1.43  -0.01  -0.50
        0  2.79e+05   13.42 |   13.42  162.1    295   112 |    0.05   0.36   0.04  -0.51
        0  7.27e+05  203.74 |  203.74  100.6    342   113 |    0.16   0.13  -0.01  -0.53
        | UsedTime:     823 |

        0  3.35e+05  -62.39 |  -62.39  144.1    411   151 |    0.00   0.32  -0.02  -0.51
        0  5.43e+05  164.83 |  164.83  145.1    371    97 |    0.13   0.16  -0.06  -0.52
        0  7.82e+05  204.31 |  204.31  126.2    347   108 |    0.18   0.17  -0.00  -0.52
        | UsedTime:     862 |
        """
        args.eval_times1 = 2 ** 4
        args.eval_times2 = 2 ** 6

        args.target_step = args.env.max_step * 4
    if env_name in {'BipedalWalker-v3', 'BipedalWalker-v2'}:
        """
        Step 51e5,  Reward 300,  UsedTime 2827s PPO
        Step 78e5,  Reward 304,  UsedTime 4747s PPO
        Step 61e5,  Reward 300,  UsedTime 3977s PPO GAE
        Step 95e5,  Reward 291,  UsedTime 6193s PPO GAE
        """
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

        args.gamma = 0.98
        args.target_step = args.env.max_step * 16
    if env_name in {'BipedalWalkerHardcore-v3', 'BipedalWalkerHardcore-v2'}:
        """
        Step 57e5,  Reward 295,  UsedTime 17ks PPO
        Step 70e5,  Reward 300,  UsedTime 21ks PPO
        """
        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.max_memo = 2 ** 22
        args.batch_size = args.net_dim * 4
        args.repeat_times = 2 ** 4
        args.learning_rate = 2 ** -16

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 5
        # args.break_step = int(80e5)

        args.worker_num = 4
        args.target_step = args.env.max_step * 16

    # todo ensemble
    args.if_overwrite = False
    args.ensemble_gap = 2 ** 8
    args.cwd = './temp'
    args.target_return = 320
    args.if_allow_break = True

    # args.learner_gpus = (0,)  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    args.learner_gpus = np.array(((0,), (1,), (2,), (3,))) + GPU_ID  # ensemble GPUs
    args.eval_gpu_id = GPU_ID
    train_and_evaluate_em(args)  # multiple process


if __name__ == '__main__':
    import sys

    sys.argv.extend('0 '.split(' '))
    GPU_ID = 0  # eval(sys.argv[1])
    ENV_ID = 2  # eval(sys.argv[2])
    DRL_ID = 2  # eval(sys.argv[3])
    demo_continuous_action_on_policy()
