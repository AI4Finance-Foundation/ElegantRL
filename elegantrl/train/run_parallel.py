import os
import sys
import time
import torch
import numpy as np
import multiprocessing as mp
from elegantrl.envs.gym import build_env, build_eval_env
from elegantrl.train.replay_buffer import ReplayBufferMP
from elegantrl.train.evaluator import Evaluator

"""[ElegantRL.2021.10.21](https://github.com/AI4Finance-Foundation/ElegantRL)"""


def get_step_r_exp(ten_reward):
    return len(ten_reward), ten_reward.mean().item()


'''multiple processing training'''


def train_and_evaluate_mp(args, agent_id=0):
    args.init_before_training()  # necessary!

    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    '''learner'''
    learner_num = len(args.learner_gpus)
    learner_pipe = PipeLearner(learner_num)
    for learner_id in range(learner_num):
        '''evaluator'''
        if learner_id == learner_num - 1:
            evaluator_pipe = PipeEvaluator()
            process.append(mp.Process(target=evaluator_pipe.run, args=(args, agent_id)))
        else:
            evaluator_pipe = None

        '''explorer'''
        worker_pipe = PipeWorker(args.env_num, args.worker_num)
        for worker_id in range(args.worker_num):
            # if args.env_num == 1:
            #     env_pipe = None
            # else:
            #     env_pipe = PipeVectorEnv(args)
            #     process.extend(env_pipe.process)
            env_pipe = None
            process.append(mp.Process(target=worker_pipe.run, args=(args, env_pipe, worker_id, learner_id)))

        process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe, learner_id)))

    [(p.start(), time.sleep(0.1)) for p in process]
    process[-1].join()
    process_safely_terminate(process)


class PipeWorker:
    def __init__(self, env_num, worker_num):
        self.env_num = env_num
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore0(self, agent):
        act_dict = agent.act.state_dict()
        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        return traj_lists

    def explore(self, agent):
        act_dict = agent.act.state_dict()

        if sys.platform == 'win32':  # todo: not elegant. YonV1943. Avoid CUDA runtime error (801)
            # Python3.9< multiprocessing can't send torch.tensor_gpu in WinOS. So I send torch.tensor_cpu
            for key, value in act_dict.items():
                act_dict[key] = value.to(torch.device('cpu'))

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        return traj_lists

    def run(self, args, _comm_env, worker_id, learner_id):  # not elegant: comm_env
        # print(f'| os.getpid()={os.getpid()} PipeExplore.run {learner_id}')
        env = build_env(env=args.env, if_print=False, device_id=args.workers_gpus[learner_id], env_num=args.env_num)

        '''init Agent'''
        agent = args.agent
        agent.init(net_dim=args.net_dim, gpu_id=args.learner_gpus[learner_id],
                   state_dim=args.state_dim, action_dim=args.action_dim, env_num=args.env_num,
                   learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae)
        if args.env_num == 1:
            agent.states = [env.reset(), ]
        else:
            agent.states = env.reset()  # VecEnv

        '''loop'''
        gamma = args.gamma
        target_step = args.target_step
        reward_scale = args.reward_scale
        del args

        with torch.no_grad():
            while True:
                act_dict = self.pipes[worker_id][0].recv()

                if sys.platform == 'win32':  # todo: not elegant. YonV1943. Avoid CUDA runtime error (801)
                    # Python3.9< multiprocessing can't send torch.tensor_gpu in WinOS. So I send torch.tensor_cpu
                    for key, value in act_dict.items():
                        act_dict[key] = value.to(agent.device)

                agent.act.load_state_dict(act_dict)

                trajectory = agent.explore_env(env, target_step, reward_scale, gamma)
                if sys.platform == 'win32':  # todo: not elegant. YonV1943. Avoid CUDA runtime error (801)
                    # Python3.9< multiprocessing can't send torch.tensor_gpu in WinOS. So I send torch.tensor_cpu
                    trajectory = [[item.to(torch.device('cpu'))
                                   for item in item_list]
                                  for item_list in trajectory]

                self.pipes[worker_id][0].send(trajectory)


class PipeLearner:
    def __init__(self, learner_num):
        self.learner_num = learner_num
        self.round_num = int(np.log2(learner_num))

        self.pipes = [mp.Pipe() for _ in range(learner_num)]
        pipes = [mp.Pipe() for _ in range(learner_num)]
        self.pipe0s = [pipe[0] for pipe in pipes]
        self.pipe1s = [pipe[1] for pipe in pipes]
        self.device_list = [torch.device(f'cuda:{i}') for i in range(learner_num)]

        if learner_num == 1:
            self.idx_l = None
        elif learner_num == 2:
            self.idx_l = [(1,), (0,), ]
        elif learner_num == 4:
            self.idx_l = [(1, 2), (0, 3),
                          (3, 0), (2, 1), ]
        elif learner_num == 8:
            self.idx_l = [(1, 2, 4), (0, 3, 5),
                          (3, 0, 6), (2, 1, 7),
                          (5, 6, 0), (4, 7, 1),
                          (7, 4, 2), (6, 5, 3), ]
        else:
            print(f"| LearnerPipe, ERROR: learner_num {learner_num} should in (1, 2, 4, 8)")
            exit()

    def comm_data(self, data, learner_id, round_id):
        if round_id == -1:
            learner_jd = self.idx_l[learner_id][round_id]
            self.pipes[learner_jd][0].send(data)
            return self.pipes[learner_id][1].recv()
        else:
            learner_jd = self.idx_l[learner_id][round_id]
            self.pipe0s[learner_jd].send(data)
            return self.pipe1s[learner_id].recv()

    def comm_network_optim(self, agent, learner_id):
        device = self.device_list[learner_id]

        for round_id in range(self.round_num):
            data = get_comm_data(agent)
            data = self.comm_data(data, learner_id, round_id)

            if data:
                avg_update_net(agent.act, data[0], device)
                avg_update_optim(agent.act_optim, data[1], device) if data[1] else None

                avg_update_net(agent.cri, data[2], device) if data[2] else None
                avg_update_optim(agent.cri_optim, data[3], device)

                avg_update_net(agent.act_target, data[4], device) if agent.if_use_act_target else None
                avg_update_net(agent.cri_target, data[5], device) if agent.if_use_cri_target else None

    def run0(self, args, comm_eva, comm_exp, learner_id=0):
        # print(f'| os.getpid()={os.getpid()} PipeLearn.run, {learner_id}')
        pass

        '''init Agent'''
        agent = args.agent
        agent.init(net_dim=args.net_dim, gpu_id=args.learner_gpus[learner_id],
                   state_dim=args.state_dim, action_dim=args.action_dim, env_num=args.env_num,
                   learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae)
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
        cwd = args.cwd
        batch_size = args.batch_size
        repeat_times = args.repeat_times
        soft_update_tau = args.soft_update_tau
        del args

        if_train = True
        while if_train:
            traj_lists = comm_exp.explore(agent)
            if self.learner_num > 1:
                data = self.comm_data(traj_lists, learner_id, round_id=-1)
                traj_lists.extend(data)
            traj_list = sum(traj_lists, list())

            steps, r_exp = update_buffer(traj_list)
            del traj_lists
            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
            if self.learner_num > 1:
                self.comm_network_optim(agent, learner_id)

            if comm_eva:
                if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)

        agent.save_or_load_agent(cwd, if_save=True)
        if agent.if_off_policy:
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)

    def run(self, args, comm_eva, comm_exp, learner_id=0):
        # print(f'| os.getpid()={os.getpid()} PipeLearn.run, {learner_id}')
        pass

        '''init Agent'''
        agent = args.agent
        agent.init(net_dim=args.net_dim, gpu_id=args.learner_gpus[learner_id],
                   state_dim=args.state_dim, action_dim=args.action_dim, env_num=args.env_num,
                   learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae)
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
        cwd = args.cwd
        batch_size = args.batch_size
        repeat_times = args.repeat_times
        soft_update_tau = args.soft_update_tau
        del args

        if_train = True
        while if_train:
            traj_lists = comm_exp.explore(agent)
            if self.learner_num > 1:
                data = self.comm_data(traj_lists, learner_id, round_id=-1)
                traj_lists.extend(data)
            traj_list = sum(traj_lists, list())

            if sys.platform == 'win32':  # todo: not elegant. YonV1943. Avoid CUDA runtime error (801)
                # Python3.9< multiprocessing can't send torch.tensor_gpu in WinOS. So I send torch.tensor_cpu
                traj_list = [[item.to(torch.device('cpu'))
                              for item in item_list]
                             for item_list in traj_list]

            steps, r_exp = update_buffer(traj_list)
            del traj_lists

            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
            if self.learner_num > 1:
                self.comm_network_optim(agent, learner_id)

            if comm_eva:
                if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)

        agent.save_or_load_agent(cwd, if_save=True)
        if agent.if_off_policy:
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)


class PipeEvaluator:  # [ElegantRL.10.21]
    def __init__(self):
        super().__init__()
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save_mp(self, agent_act, steps, r_exp, logging_tuple):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train, if_save = self.pipe1.recv()
            act_cpu_dict = {k: v.cpu() for k, v in agent_act.state_dict().items()}
        else:
            if_train, if_save = True, False
            act_cpu_dict = None

        self.pipe1.send((act_cpu_dict, steps, r_exp, logging_tuple))
        return if_train, if_save

    def run(self, args, _learner_id):
        # print(f'| os.getpid()={os.getpid()} PipeEvaluate.run {agent_id}')
        pass

        '''init: Agent'''
        agent = args.agent
        agent.init(net_dim=args.net_dim, gpu_id=args.eval_gpu_id,
                   state_dim=args.state_dim, action_dim=args.action_dim, env_num=args.env_num,
                   learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae)
        agent.save_or_load_agent(args.cwd, if_save=False)

        act = agent.act
        [setattr(param, 'requires_grad', False) for param in agent.act.parameters()]
        del agent

        '''init Evaluator'''
        eval_env = build_eval_env(args.eval_env, args.env, args.eval_gpu_id, args.env_num)
        evaluator = Evaluator(cwd=args.cwd, agent_id=0,
                              eval_env=eval_env, eval_gap=args.eval_gap,
                              eval_times1=args.eval_times1, eval_times2=args.eval_times2,
                              target_return=args.target_return, if_overwrite=args.if_overwrite)
        evaluator.save_or_load_recoder(if_save=False)

        '''loop'''
        cwd = args.cwd
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        del args

        if_save = False
        if_train = True
        if_reach_goal = False
        with torch.no_grad():
            while if_train:
                act_dict, steps, r_exp, logging_tuple = self.pipe0.recv()

                if act_dict:
                    act.load_state_dict(act_dict)
                    if_reach_goal, if_save = evaluator.evaluate_and_save(act, steps, r_exp, logging_tuple)
                else:
                    evaluator.total_step += steps

                if_train = not ((if_allow_break and if_reach_goal)
                                or evaluator.total_step > break_step
                                or os.path.exists(f'{cwd}/stop'))
                self.pipe0.send((if_train, if_save))

        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')
        evaluator.save_or_load_recoder(if_save=True)


# class PipeVectorEnv:
#     def __init__(self, args):
#         self.env_num = args.env_num
#         self.pipes = [mp.Pipe() for _ in range(self.env_num)]
#         self.pipe0s = [pipe[0] for pipe in self.pipes]
#
#         env = build_env(args.eval_env)
#         self.max_step = env.max_step
#         self.env_name = env.env_name
#         self.state_dim = env.state_dim
#         self.action_dim = env.action_dim
#         self.action_max = env.action_max
#         self.if_discrete = env.if_discrete
#         self.target_return = env.target_return
#         del env
#
#         self.process = list()
#         for env_id in range(args.env_num):
#             self.process.append(mp.Process(target=self.run, args=(args, env_id)))
#             args.random_seed += 1  # set different for each env
#         # [p.start() for p in self.process]
#
#     def reset(self):
#         vec_state = [pipe0.recv() for pipe0 in self.pipe0s]
#         return vec_state
#
#     def step(self, vec_action):  # pipe0_step
#         for i in range(self.env_num):
#             self.pipe0s[i].send(vec_action[i])
#         return [pipe0.recv() for pipe0 in self.pipe0s]  # list of (state, reward, done)
#
#     def run(self, args, env_id):
#         np.random.seed(args.random_seed)
#
#         env = build_env(args.eval_env, if_print=False)
#         pipe1 = self.pipes[env_id][1]
#         del args
#
#         state = env.reset()
#         pipe1.send(state)
#
#         while True:
#             action = pipe1.recv()
#             state, reward, done, _ = env.step(action)
#             pipe1.send((env.reset() if done else state, reward, done))
#
#     # def check(self):
#     #     vec_state = self.reset()
#     #     ten_state = np.array(vec_state)
#     #     print(ten_state.shape)
#     #
#     #     vec_action = np.array(((0.0, 1.0, 0.0),
#     #                            (0.0, 0.5, 0.0),
#     #                            (0.0, 0.1, 0.0),))[:self.env_num]
#     #     assert self.env_num <= 3
#     #
#     #     trajectory_list = list()
#     #     for _ in range(8):
#     #         s_r_d_list = self.step(vec_action)
#     #         ten_state = np.array([s_r_d[0] for s_r_d in s_r_d_list])
#     #         print(ten_state.shape)
#     #         trajectory_list.append(s_r_d_list)
#     #
#     #     trajectory_list = list(map(list, zip(*trajectory_list)))  # 2D-list transpose
#     #     print('| shape of trajectory_list:', len(trajectory_list), len(trajectory_list[0]))

def get_comm_data(agent):
    act = list(agent.act.parameters())
    cri_optim = get_optim_parameters(agent.cri_optim)

    if agent.cri is agent.act:
        cri = None
        act_optim = None
    else:
        cri = list(agent.cri.parameters())
        act_optim = get_optim_parameters(agent.act_optim)

    act_target = list(agent.act_target.parameters()) if agent.if_use_act_target else None
    cri_target = list(agent.cri_target.parameters()) if agent.if_use_cri_target else None
    return act, act_optim, cri, cri_optim, act_target, cri_target  # data


"""Utils"""


def get_num_learner(visible_gpu):
    assert isinstance(visible_gpu, str)  # visible_gpu may in {'0', '1', '1,', '1,2', '1,2,'}
    visible_gpu = eval(visible_gpu)
    num_learner = 1 if isinstance(visible_gpu, int) else len(visible_gpu)
    return num_learner


def process_safely_terminate(process):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
            pass


def get_optim_parameters(optim):  # for avg_update_optim()
    params_list = list()
    for params_dict in optim.state_dict()['state'].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


def avg_update_optim(dst_optim, src_optim_param, device):
    for dst, src in zip(get_optim_parameters(dst_optim), src_optim_param):
        dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
        # dst.data.copy_(src.data * tau + dst.data * (1 - tau))


def avg_update_net(dst_net, src_net_param, device):
    for dst, src in zip(dst_net.parameters(), src_net_param):
        dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
