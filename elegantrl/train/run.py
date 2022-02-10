import os
import shutil
import sys
import time

import torch

from elegantrl.train.config import build_env
from elegantrl.train.learner import PipeLearner
from elegantrl.train.utils import init_agent, init_evaluator, init_replay_buffer
from elegantrl.train.utils import server_leaderboard, PipeEvaluator
from elegantrl.train.worker import PipeWorker


def train_and_evaluate(args):
    args.init_before_training()  # necessary!
    learner_gpu = args.learner_gpus[0]

    env = build_env(
        env=args.env, env_func=args.env_func, env_args=args.env_args, gpu_id=learner_gpu
    )
    agent = init_agent(args, gpu_id=learner_gpu, env=env)
    evaluator = init_evaluator(args, agent_id=0)
    buffer, update_buffer = init_replay_buffer(args, learner_gpu, agent, env=env)

    """start training"""
    cwd = args.cwd
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    """start training loop"""
    if_train = True
    torch.set_grad_enabled(False)
    while if_train:
        traj_list = agent.explore_env(env, target_step)
        steps, r_exp = update_buffer(traj_list)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(
            buffer, batch_size, repeat_times, soft_update_tau
        )
        torch.set_grad_enabled(False)

        if_reach_goal, if_save = evaluator.evaluate_and_save(
            agent.act, steps, r_exp, logging_tuple
        )
        if_train = not (
            (if_allow_break and if_reach_goal)
            or evaluator.total_step > break_step
            or os.path.exists(f"{cwd}/stop")
        )

    print(f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}")

    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None
    evaluator.save_or_load_recoder(if_save=True)


def train_and_evaluate_mp(args, python_path=""):
    import multiprocessing as mp

    if_from_ensemble = sys.argv[-1] == "FromEnsemble"
    from collections.abc import Iterable

    if isinstance(args.learner_gpus, int):
        args.learner_gpus = (args.learner_gpus,)
    process = []
    if (not isinstance(args.learner_gpus[0], Iterable)) or if_from_ensemble:
        agent_id = int(sys.argv[-2]) if if_from_ensemble else 0

        args.init_before_training(agent_id=agent_id)  # necessary!

        mp.set_start_method(
            method="spawn", force=True
        )  # force all the multiprocessing to 'spawn' methods

        """evaluator"""
        evaluator_pipe = PipeEvaluator(save_gap=args.save_gap, save_dir=args.save_dir)
        process.append(mp.Process(target=evaluator_pipe.run, args=(args, agent_id)))

        learner_pipe = PipeLearner(args.learner_gpus)
        for learner_id in range(len(args.learner_gpus)):
            """explorer"""
            worker_pipe = PipeWorker(args.env_num, args.worker_num)
            process.extend(
                [
                    mp.Process(
                        target=worker_pipe.run, args=(args, worker_id, learner_id)
                    )
                    for worker_id in range(args.worker_num)
                ]
            )

            """learner"""
            evaluator_temp = evaluator_pipe if learner_id == 0 else None
            process.append(
                mp.Process(
                    target=learner_pipe.run,
                    args=(args, evaluator_temp, worker_pipe, learner_id),
                )
            )

        [(p.start(), time.sleep(0.1)) for p in process]
        process[0].join()
        safely_terminate_process(process)
    else:
        from subprocess import Popen

        python_path = python_path or get_python_path()
        python_proc = sys.argv[0]

        ensemble_dir = args.save_dir
        ensemble_num = len(args.learner_gpus)

        shutil.rmtree(ensemble_dir, ignore_errors=True)
        os.makedirs(ensemble_dir, exist_ok=True)

        proc_leaderboard = mp.Process(
            target=server_leaderboard, args=(ensemble_num, ensemble_dir)
        )
        proc_leaderboard.start()

        print("subprocess Start")
        for agent_id in range(ensemble_num):
            command_str = f"{python_path} {python_proc} {agent_id} FromEnsemble"
            command_list = command_str.split(" ")
            process.append(Popen(command_list))

        for proc in process:
            proc.communicate()
        print("subprocess Stop")

        proc_leaderboard.join()


"""private utils"""


def get_python_path():  # useless
    from subprocess import check_output

    python_path = check_output("which python3", shell=True).strip()
    python_path = python_path.decode("utf-8")
    print(f"| get_python_path: {python_path}")
    return python_path


def safely_terminate_process(process):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)


def check_subprocess():
    import subprocess

    timer = time.time()
    print("subprocess Start")

    process = [subprocess.Popen("sleep 3".split(" ")) for _ in range(4)]
    [proc.communicate() for proc in process]
    print("subprocess Stop:", time.time() - timer)
