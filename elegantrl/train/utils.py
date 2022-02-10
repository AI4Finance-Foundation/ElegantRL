import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import numpy.random as rd
import torch

from elegantrl.train.config import build_env
from elegantrl.train.evaluator import Evaluator
from elegantrl.train.replay_buffer import ReplayBuffer, ReplayBufferMP


def init_agent(args, gpu_id=0, env=None):
    agent = args.agent
    agent.init(
        net_dim=args.net_dim,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        learning_rate=args.learning_rate,
        if_per_or_gae=args.if_per_or_gae,
        env_num=args.env_num,
        gpu_id=gpu_id,
    )
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        """init states"""
        if args.env_num == 1:
            states = [
                env.reset(),
            ]
            assert isinstance(states[0], np.ndarray)
            assert states[0].shape in {(args.state_dim,), args.state_dim}
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states

    return agent


def init_evaluator(args, agent_id=0):
    eval_env = build_env(
        env=args.eval_env,
        gpu_id=args.eval_gpu_id,
        env_func=args.eval_env_func,
        env_args=args.eval_env_args,
    )
    evaluator = Evaluator(
        cwd=args.cwd,
        agent_id=agent_id,
        eval_env=eval_env,
        eval_gap=args.eval_gap,
        eval_times1=args.eval_times1,
        eval_times2=args.eval_times2,
        target_return=args.target_return,
        if_overwrite=args.if_overwrite,
    )
    evaluator.save_or_load_recoder(if_save=False)
    return evaluator


def init_replay_buffer(args, learner_gpu, agent=None, env=None):
    def get_step_r_exp(ten_reward):
        return len(ten_reward), ten_reward.mean().item()

    if args.if_off_policy:
        if args.worker_num == 1:
            buffer = ReplayBuffer(
                gpu_id=learner_gpu,
                max_len=args.max_memo,
                state_dim=args.state_dim,
                action_dim=1 if args.if_discrete else args.action_dim,
                if_use_per=args.if_per_or_gae,
            )
        else:
            buffer = ReplayBufferMP(
                gpu_id=learner_gpu,
                max_len=args.max_memo,
                state_dim=args.state_dim,
                action_dim=1 if args.if_discrete else args.action_dim,
                buffer_num=1,
                if_use_per=args.if_per_or_gae,
            )

        buffer.save_or_load_history(args.cwd, if_save=False)

        def update_buffer(traj_list):
            steps_r_exp_list = []
            for ten_state, ten_other in traj_list:
                buffer.extend_buffer(ten_state, ten_other)

                steps_r_exp_list.append(get_step_r_exp(ten_reward=ten_other[:, 0]))

            steps_r_exp_list = np.array(steps_r_exp_list)
            steps, r_exp = steps_r_exp_list.mean(axis=0)
            return steps, r_exp

        if_load = buffer.save_or_load_history(args.cwd, if_save=False)
        if (env is not None) and (not if_load):
            update_buffer(agent.explore_env(env, args.target_step))

    else:
        buffer = []

        def update_buffer(traj_list):
            cur_items = list(map(list, zip(*traj_list)))
            cur_items = [torch.cat(item, dim=0) for item in cur_items]
            buffer[:] = cur_items

            steps, r_exp = get_step_r_exp(ten_reward=buffer[1])
            return steps, r_exp

    return buffer, update_buffer


def add_tensor(p0, dst_tensor, src_tensor):  # for `update_buffer` (on-policy)
    p1 = p0 + src_tensor.shape[0]
    if p0 != p1:
        dst_tensor[p0:p1] = src_tensor
    return p1  # pointer


"""multiple process (worker.py evaluator.py)"""


def act_dict_to_device(act_dict, device):
    """
    :param act_dict: net.state_dict()
    :param device: torch.device(f"cuda:{int}")
    :return: act_dict: net.state_dict()
    """
    for key, value in act_dict.items():
        act_dict[key] = value.to(device)
    return act_dict


def trajectory_to_device(trajectory, device):
    trajectory[:] = [
        [item.to(device) for item in item_list] for item_list in trajectory
    ]
    return trajectory


"""ensemble DRL (evaluator.py and leaderboard.py)"""


class PipeEvaluator:  # [ElegantRL.10.21]
    def __init__(self, save_gap, save_dir=None):
        super().__init__()
        self.pipe0, self.pipe1 = mp.Pipe()

        self.save_dir = save_dir  # save_dir = None, means don't save.
        self.save_gap = save_gap
        self.save_timer = time.time()

    def evaluate_and_save_mp(self, agent, steps, r_exp, logging_tuple, cwd):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train, if_save = self.pipe1.recv()
            act_cpu_dict = act_dict_to_device(
                agent.act.state_dict(), torch.device("cpu")
            )
        else:
            if_train, if_save = True, False
            act_cpu_dict = None

        self.pipe1.send((act_cpu_dict, steps, r_exp, logging_tuple))

        if self.save_timer + self.save_gap < time.time() and self.save_dir:
            self.save_timer = time.time()

            """save"""
            episode_return = get_epi_returns(cwd)
            if episode_return:
                """save training temp files"""
                save_path = f"{self.save_dir}/pod_save_{episode_return:09.3f}"
                if not os.path.exists(save_path):
                    with DirLock(save_path):
                        os.mkdir(save_path)
                        agent.save_or_load_agent(save_path, if_save=True)

            """load"""
            load_dir = find_load_dir(cwd)
            if load_dir:
                with DirLock(load_dir):
                    agent.save_or_load_agent(load_dir, if_save=False)
                    os.rmdir(f"{cwd}/load_dir_{load_dir}")
        return if_train, if_save

    def run(self, args, agent_id):
        gpu_id = args.eval_gpu_id

        evaluator = init_evaluator(args, agent_id)
        agent = init_agent(args, gpu_id=gpu_id, env=None)
        act = agent.act
        del agent

        """loop"""
        cwd = args.cwd
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        del args

        torch.set_grad_enabled(False)
        if_save = False
        if_train = True
        if_reach_goal = False
        while if_train:
            act_dict, steps, r_exp, logging_tuple = self.pipe0.recv()

            if act_dict:
                act.load_state_dict(act_dict)
                if_reach_goal, if_save = evaluator.evaluate_and_save(
                    act, steps, r_exp, logging_tuple
                )
            else:
                evaluator.total_step += steps

            if_train = not (
                (if_allow_break and if_reach_goal)
                or evaluator.total_step > break_step
                or os.path.exists(f"{cwd}/stop")
            )
            self.pipe0.send((if_train, if_save))

        print(
            f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}"
        )
        evaluator.save_or_load_recoder(if_save=True)


class DirLock:
    def __init__(self, dir_path):
        self.lock_path = f"{dir_path}_lock"

    def __enter__(self):
        while os.path.exists(self.lock_path):
            time.sleep(0.25)
        os.mkdir(self.lock_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.lock_path):
            os.rmdir(self.lock_path)


def get_epi_returns(cwd):
    recorder_path = f"{cwd}/recorder.npy"

    if os.path.exists(recorder_path):
        recorder = np.load(recorder_path)
        return recorder[-4:, 1].mean()
    else:
        return None


def find_load_dir(cwd):
    load_dir = None
    for name in os.listdir(cwd):
        if name[:9] == "load_dir_":
            load_dir = name[9:]
    return load_dir


"""ensemble LeaderBoard"""


def server_leaderboard(
    ensemble_num, leaderboard_dir="./LeaderBoard"
):  # [ElegantRL.2021.12.12]
    torch.set_grad_enabled(False)  # with torch.no_grad():

    max_pod_num = ensemble_num * 2

    while True:
        time.sleep(ensemble_num * 4)

        """remove pod"""
        dir_names = [
            dir_name
            for dir_name in os.listdir(leaderboard_dir)
            if dir_name.find("pod_save_") >= 0
        ]
        if len(dir_names) > max_pod_num:
            sort_str_list_inplace(dir_names)
            for dir_name in dir_names[:-max_pod_num]:
                remove_dir_path = f"{leaderboard_dir}/{dir_name}"
                with DirLock(remove_dir_path):
                    shutil.rmtree(remove_dir_path)
        dir_names = dir_names[:max_pod_num]

        """get LeaderBoard"""
        if len(dir_names) >= ensemble_num:
            for agent_id in range(ensemble_num):
                pod_dir = f"{leaderboard_dir}/pod_{agent_id:04}"
                load_dir = find_load_dir(pod_dir)
                if load_dir:
                    continue

                """create `load_dir_xxx` in pod_dir"""
                epi_returns = np.array([float(name[9:]) for name in dir_names])
                epi_returns_soft_max = np_soft_max(epi_returns)
                name_id = rd.choice(len(epi_returns_soft_max), p=epi_returns_soft_max)

                src_epi_returns = epi_returns[name_id]
                dst_epi_returns = get_epi_returns(pod_dir)
                if src_epi_returns > dst_epi_returns:
                    with DirLock(pod_dir):
                        os.makedirs(
                            f"{pod_dir}/load_dir_{dir_names[name_id]}", exist_ok=True
                        )


def np_soft_max(raw_x):
    norm_x = (raw_x - raw_x.mean()) / (raw_x.std() + 1e-6)
    exp_x = np.exp(norm_x) + 1e-6
    return exp_x / exp_x.sum()


def sort_str_list_inplace(str_list):
    str_list.sort()

    i = 0
    for str_item in str_list:
        if str_item[9] != "-":
            break
        i += 1

    str_list[:i] = str_list[:i][::-1]
    return str_list


"""utils"""


def get_nd_list(nd_list):
    print_str = "| check_nd_list:"

    item = nd_list
    if hasattr(item, "shape"):
        print_str += f" {item.shape}"
    elif hasattr(item, "__len__"):
        print_str += f" {len(item)}"

        if len(nd_list) > 0:
            item = nd_list[0]
            if hasattr(item, "shape"):
                print_str += f" {item.shape}"
            elif hasattr(item, "__len__"):
                print_str += f" {len(item)}"

                if len(nd_list[0]) > 0:
                    item = nd_list[0][0]
                    if hasattr(item, "shape"):
                        print_str += f" {item.shape}"
                    elif hasattr(item, "__len__"):
                        print_str += f" {len(item)}"

                    else:
                        print_str += " END"
            else:
                print_str += " END"
    else:
        print_str += " END"

    return print_str
