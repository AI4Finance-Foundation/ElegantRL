import os
import shutil
import time

import numpy as np
import numpy.random as rd

from elegantrl.envs.Chasing import ChasingEnv, ChasingVecEnv
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train.config import Arguments
from elegantrl.train.run_parallel import *
from elegantrl.train.evaluator import *


class Ensemble:
    def __init__(self, ensemble_gap, ensemble_num, ensemble_id, cwd):
        self.ensemble_gap = ensemble_gap
        self.ensemble_num = ensemble_num
        self.ensemble_timer = time.time() + ensemble_gap * ensemble_id / ensemble_num
        self.agent_id = ensemble_id

        self.save_path = f"{cwd}/pod_temp_{ensemble_id:04}"

    def run(self, cwd, agent):

        if self.ensemble_timer + self.ensemble_gap > time.time():
            return
        self.ensemble_timer = time.time()

        '''save'''
        with DirLock(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            agent.save_or_load_agent(self.save_path, if_save=True)

        '''load'''
        load_path = self.get_load_path_randomly(cwd)

        if load_path:
            agent.traj_list[:] = [list() for _ in agent.traj_list]
            with DirLock(load_path):
                agent.save_or_load_agent(load_path, if_save=False)
                print(f"|{'':20}Load: ID {self.agent_id}    r_avg {load_path[-9:]}")  # todo remove

    def get_load_path_randomly(self, cwd):
        names = [name for name in os.listdir(cwd) if name.find('pod_eval_') == 0]
        if len(names):
            returns = np.array([float(name[9:]) for name in names])
            returns_soft_max = self.np_soft_max(returns)
            name_id = rd.choice(len(returns_soft_max), p=returns_soft_max)

            load_path = f"{cwd}/pod_eval_{names[name_id]}"
        else:
            load_path = None
        return load_path

    @staticmethod
    def np_soft_max(raw_x):
        norm_x = (raw_x - raw_x.mean()) / (raw_x.std() + 1e-6)
        exp_x = np.exp(norm_x) + 1e-6
        return exp_x / exp_x.sum()


class DirLock:
    def __init__(self, dir_path):
        self.lock_path = f'{dir_path}_lock'

    def __enter__(self):
        while os.path.exists(self.lock_path):
            time.sleep(0.25)
        os.mkdir(self.lock_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.lock_path):
            os.rmdir(self.lock_path)


def get_last_list_of_a_file(file_path) -> list:
    """How to read the last line of a file in Python?
    https://stackoverflow.com/a/54278929/9293137
    """
    with open(file_path, 'rb') as f:
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()

    return last_line[:-1].split('\t')  # last_list


def get_r_avg_std_s_avg_std(reward_step_list):
    rewards_steps_ary = np.array(reward_step_list, dtype=np.float32)
    r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
    r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
    return r_avg, r_std, s_avg, s_std


def recorder_get_total_step(recoder_path):
    if os.path.isfile(recoder_path):
        last_list = get_last_list_of_a_file(recoder_path)
        total_step = int(last_list[2])
    else:
        total_step = 0
    return total_step


def recorder_load_last_line(src_path):
    last_list = get_last_list_of_a_file(f'{src_path}/recorder.txt')
    r_exp, step, total_step, logging_str = last_list
    r_exp = float(r_exp)
    step = int(step)
    total_step = int(step)
    return r_exp, step, total_step, logging_str


def recorder_save_last_line(cwd, ensemble_id, steps, r_exp, logging_str):
    recoder_path = f"{cwd}/pod_temp_{ensemble_id:04}/recorder.txt"
    total_step = recorder_get_total_step(recoder_path)

    with open(recoder_path, 'a+') as f:
        total_step += steps
        f.write(f"{r_exp:8.2f}\t{steps:08.0f}\t{total_step:016.0f}\t{logging_str}\n")

    print(f"{ensemble_id:<3}{total_step:8.2e}{'':8} |"
          f"{'':8}{'':7}{'':7}{'':6} |"
          f"{r_exp:8.2f}{logging_str}")


def server_evaluator(args):  # [ElegantRL.2021.12.12]
    torch.set_grad_enabled(False)  # with torch.no_grad():

    '''init: Agent'''
    agent = args.agent
    agent.init(net_dim=args.net_dim,
               state_dim=args.state_dim,
               action_dim=args.action_dim,

               gamma=args.gamma,
               reward_scale=args.reward_scale,
               learning_rate=args.learning_rate,
               if_per_or_gae=args.if_per_or_gae,

               env_num=args.env_num,
               gpu_id=args.eval_gpu_id, )
    act = agent.act  # [setattr(param, 'requires_grad', False) for param in agent.act.parameters()]
    del agent

    '''init Evaluator'''
    eval_env = build_env(env=args.eval_env,
                         env_func=args.eval_env_class,
                         env_args=args.eval_env_args,
                         env_info=args.eval_env_info,

                         device_id=args.eval_gpu_id, )

    '''loop'''
    cwd = args.cwd
    ensemble_num = len(args.ensemble_gpus)
    r_max = -np.inf
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    target_return = args.target_return
    if_allow_break = args.if_allow_break
    del args

    print(f"{'#' * 80}\n"
          f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
          f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
          f"{'expR':>8}{'objC':>7}{'etc.':>7}")
    from itertools import cycle
    for agent_id in cycle(range(ensemble_num)):
        src_path = f"{cwd}/pod_temp_{agent_id:04}"

        load_path = f"{src_path}/actor.pth"
        if not os.path.isfile(load_path):
            time.sleep(1)
            continue

        '''load and evaluate'''
        act.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))

        reward_step_list = [get_episode_return_and_step(eval_env, act)
                            for _ in range(eval_times1)]
        r_avg, r_std, s_avg, s_std = get_r_avg_std_s_avg_std(reward_step_list)
        if r_avg > r_max:
            reward_step_list += [get_episode_return_and_step(eval_env, act)
                                 for _ in range(eval_times2 - eval_times1)]
            r_avg, r_std, s_avg, s_std = get_r_avg_std_s_avg_std(reward_step_list)
        if r_avg > r_max:
            r_max = r_avg

        '''print'''
        r_exp, step, total_step, logging_str = recorder_load_last_line(src_path)

        print(f"{agent_id:<3}{total_step:8.2e}{r_max:8.2f} |"
              f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
              f"{r_exp:8.2f}{logging_str}")

        '''move'''
        with DirLock(src_path):
            os.rename(src=f"{cwd}/pod_temp_{agent_id:04}",
                      dst=f"{cwd}/pod_eval_{r_avg:09.3f}")
            os.mkdir(src_path)
            print(f"|{'':20}Save: ID {agent_id}    r_avg {r_avg:09.3f}")

        '''remove pod'''
        max_pod_num = ensemble_num * 2
        dir_names = [dir_name for dir_name in os.listdir(cwd)
                     if dir_name.find('pod_eval') >= 0]
        if len(dir_names) > max_pod_num:
            dir_names.sort()
            for dir_name in dir_names[:-max_pod_num]:
                remove_dir_path = f"{cwd}/{dir_name}"
                with DirLock(remove_dir_path):
                    shutil.rmtree(remove_dir_path)

        if if_allow_break and r_max > target_return:
            break
    print(f"Evaluator: r_max {r_max:8.3f} > target_return {target_return:8.3f}")


class PipeWorker:
    def __init__(self, env_num, worker_num):
        self.env_num = env_num
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent):
        act_dict = agent.act.state_dict()

        if sys.platform == 'win32':  # Avoid CUDA runtime error (801)
            # Python3.9< multiprocessing can't send torch.tensor_gpu in WinOS. So I send torch.tensor_cpu
            for key, value in act_dict.items():
                act_dict[key] = value.to(torch.device('cpu'))

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        return traj_lists

    def run(self, args, ensemble_id, learner_id, worker_id):  # not elegant: comm_env
        device_id = args.ensemble_gpus[ensemble_id][learner_id]

        env = build_env(env=args.env, device_id=device_id,
                        env_class=args.env_func, env_args=args.env_args, env_info=args.env_info)

        '''init Agent'''
        agent = args.agent
        agent.init(net_dim=args.net_dim, state_dim=args.state_dim, action_dim=args.action_dim,
                   gamma=args.gamma, reward_scale=args.reward_scale,
                   learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae,
                   env_num=args.env_num, gpu_id=device_id, )
        if args.env_num == 1:
            agent.states = [env.reset(), ]
        else:
            agent.states = env.reset()  # VecEnv

        '''loop'''
        target_step = args.target_step
        del args

        with torch.no_grad():
            while True:
                act_dict = self.pipes[worker_id][0].recv()

                if sys.platform == 'win32':
                    # Python3.9< multiprocessing can't send torch.tensor_gpu in WinOS. So I send torch.tensor_cpu
                    for key, value in act_dict.items():
                        act_dict[key] = value.to(agent.device)

                agent.act.load_state_dict(act_dict)

                trajectory = agent.explore_env(env, target_step)
                if sys.platform == 'win32':
                    # Python3.9< multiprocessing can't send torch.tensor_gpu in WinOS. So I send torch.tensor_cpu
                    trajectory = [[item.to(torch.device('cpu'))
                                   for item in item_list]
                                  for item_list in trajectory]

                self.pipes[worker_id][0].send(trajectory)


class PipeLearner:
    def __init__(self, learner_num):
        self.learner_num = learner_num

    def run(self, args, comm_exp, ensemble_id=0, learner_id=0, ):
        device_id = args.ensemble_gpus[ensemble_id][learner_id]

        '''init Agent'''
        agent = args.agent
        agent.init(net_dim=args.net_dim, state_dim=args.state_dim, action_dim=args.action_dim,
                   gamma=args.gamma, reward_scale=args.reward_scale,
                   learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae,
                   env_num=args.env_num, gpu_id=device_id, )
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init ReplayBuffer'''
        if agent.if_off_policy:
            buffer_num = args.worker_num * args.env_num
            if self.learner_num > 1:
                buffer_num *= 2

            buffer = ReplayBufferMP(max_len=args.max_memo,
                                    state_dim=args.state_dim,
                                    action_dim=1 if args.if_discrete else args.action_dim,
                                    if_use_per=args.if_per_or_gae,
                                    buffer_num=buffer_num,
                                    gpu_id=device_id)
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
                            ensemble_id=ensemble_id,
                            cwd=args.cwd)  # todo ensemble

        cwd = args.cwd
        batch_size = args.batch_size
        repeat_times = args.repeat_times
        soft_update_tau = args.soft_update_tau
        eval_gap = args.eval_gap
        del args

        eval_timer = time.time()
        if_train = True
        while if_train:
            traj_lists = comm_exp.explore(agent)
            traj_list = sum(traj_lists, list())
            del traj_lists

            steps, r_exp = update_buffer(traj_list)
            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
            logging_str = ''.join(f'{n:7.2f}' for n in logging_tuple)

            if learner_id != 0:
                continue

            if time.time() > eval_timer + eval_gap:  # todo plan to Class Recorder
                eval_timer = time.time()
                os.makedirs(f"{cwd}/pod_temp_{ensemble_id:04}", exist_ok=True)
                recorder_save_last_line(cwd, ensemble_id, steps, r_exp, logging_str)

            ensemble.run(cwd, agent)  # todo ensemble

        agent.save_or_load_agent(cwd, if_save=True)
        if agent.if_off_policy:
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)


def train_and_evaluate_en(args):
    args.init_before_training()  # necessary!

    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    '''evaluator'''
    process.append(mp.Process(target=server_evaluator, args=(args,)))

    ensemble_num = len(args.ensemble_gpus)
    learner_pipe = PipeLearner(ensemble_num)
    for ensemble_id in range(ensemble_num):
        learner_num = len(args.ensemble_gpus[ensemble_id])
        for learner_id in range(learner_num):
            '''worker'''
            worker_pipe = PipeWorker(args.env_num, args.worker_num)
            for worker_id in range(args.worker_num):
                process.append(mp.Process(target=worker_pipe.run, args=(args, ensemble_id, learner_id, worker_id)))

            '''learner'''
            process.append(mp.Process(target=learner_pipe.run, args=(args, worker_pipe, ensemble_id, learner_id)))

    [(p.start(), time.sleep(0.1)) for p in process]
    process[0].join()
    process_safely_terminate(process)


'''demo'''


def demo_custom_env2():
    agent_class = [AgentPPO, AgentModPPO][1]

    '''set env'''
    if_demo1 = True  # demo1 is better because of less RAM space
    from elegantrl.envs.Gym import GymEnv
    if if_demo1:
        env = None
        env_class = GymEnv
        env_args = {
            'env_name': 'BipedalWalker-v3',
            'state_norm_std': np.array([0.84, 0.06, 0.17, 0.09, 0.49, 0.55, 0.44, 0.85, 0.29, 0.48, 0.50, 0.48,
                                        0.70, 0.29, 0.07, 0.07, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.14, 0.08]),
            'state_norm_avg': np.array([0.14, 0.00, 0.17, 0.00, 0.24, 0.00, 0.00, 0.00, 0.45, 0.61, 0.00, -0.2,
                                        0.00, 0.50, 0.33, 0.34, 0.35, 0.37, 0.40, 0.46, 0.54, 0.67, 0.88, 1.00])
        }
        env_info = {'env_num': 1,
                    'env_name': 'BipedalWalker-v3',
                    'max_step': 1600,
                    'state_dim': 24,
                    'action_dim': 4,
                    'if_discrete': False,
                    'target_return': 300, }
        # The following code print `env_info` of a standard OpenAI Gym Env.
        # from elegantrl.envs.Gym import get_gym_env_info
        # env_info = get_gym_env_info(env=gym.make('BipedalWalker-v3'), if_print=True)
    else:  # if_demo2
        env = GymEnv(env_name='BipedalWalker-v3')  # env = env_class(*env_args)
        env_class = None
        env_args = None
        env_info = None

    '''hyper-parameters'''
    args = Arguments(agent=agent_class(), env=env,
                     env_func=env_class, env_args=env_args, env_info=env_info)

    args.if_per_or_gae = True
    args.net_dim = int(2 ** 8)
    args.batch_size = int(args.net_dim * 2)
    args.target_step = args.max_step * 2  # 4
    args.repeat_times = 2 ** 3  # 4
    args.reward_scale = 2 ** -1

    args.eval_gap = 2 ** 7
    args.if_allow_break = False

    args.random_seed += GPU_ID
    args.ensemble_gpus = np.arange(GPU_ID, GPU_ID + 4)[:, np.newaxis]
    train_and_evaluate_en(args)


def demo_custom_env3():
    agent_class = [AgentPPO, AgentModPPO][0]

    '''set env'''
    dim = 2
    env_num = 1

    if env_num == 1:
        env = None
        env_class = ChasingEnv
        env_args = {'dim': dim}
        env_info = {'env_num': 1,
                    'env_name': 'GoAfterEnv-v0',
                    'max_step': 1000,
                    'state_dim': dim * 4,
                    'action_dim': dim,
                    'if_discrete': False,
                    'target_return': 6.3, }

        eval_env_class = env_class
        eval_env_args = env_args
    else:
        env = None
        env_class = ChasingVecEnv
        env_args = {'dim': dim, 'env_num': env_num}
        env_info = {'env_num': env_num,
                    'env_name': 'GoAfterVecEnv-v0',
                    'max_step': 1000,
                    'state_dim': dim * 4,
                    'action_dim': dim,
                    'if_discrete': False,
                    'target_return': 6.3, }
        eval_env_class = ChasingEnv
        eval_env_args = {'dim': dim}

    '''hyper-parameters'''
    args = Arguments(agent=agent_class(), env=env,
                     env_func=env_class, env_args=env_args, env_info=env_info)

    args.if_per_or_gae = True
    args.net_dim = int(2 ** 8)
    args.batch_size = int(args.net_dim * 2)
    args.target_step = args.max_step * 4
    args.repeat_times = 2 ** 4

    args.if_allow_break = False
    args.eval_gap = 2 ** 7
    args.eval_times1 = 2 ** 4
    args.eval_times1 = 2 ** 6
    args.eval_env_class = eval_env_class
    args.eval_env_args = eval_env_args

    args.random_seed += GPU_ID
    args.ensemble_gpus = np.arange(GPU_ID, GPU_ID + 4)[:, np.newaxis]
    args.workers_gpus = args.ensemble_gpus
    args.eval_gpu_id = args.ensemble_gpus[0][0]
    train_and_evaluate_en(args)


if __name__ == '__main__':
    GPU_ID = 0
    demo_custom_env2()
    # demo_custom_env3()
