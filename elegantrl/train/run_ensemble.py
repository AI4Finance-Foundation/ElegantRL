from elegantrl.train.run_parallel import *
from elegantrl.train.run_tutorial import *
import random as rd
'''run.py'''


def train_and_evaluate_em(args):  # keep
    args.init_before_training()  # necessary!
    args_cwd = args.cwd

    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    # todo ensemble
    ensemble_num = len(args.ensemble_gpus)

    for agent_id in range(ensemble_num):
        # todo ensemble
        args.cwd = f'{args_cwd}/ensemble_{agent_id:02}'
        args.learner_gpus = args.ensemble_gpus[agent_id]
        args.eval_gpu_id = args.learner_gpus[0]
        args.random_seed += agent_id * len(args.learner_gpus) * args.worker_num
        os.makedirs(args.cwd, exist_ok=True)

        '''learner'''
        learner_num = len(args.learner_gpus)
        learner_pipe = PipeLearner(learner_num)
        for learner_id in range(learner_num):
            '''explorer'''
            worker_pipe = PipeWorker(args.env_num, args.worker_num)
            for worker_id in range(args.worker_num):
                proc = mp.Process(target=worker_pipe.run, args=(args, worker_id, learner_id))
                proc.start()
                process.append(proc)

            '''evaluator'''
            if learner_id == learner_num - 1:
                evaluator_pipe = PipeEvaluator()
                proc = mp.Process(target=evaluator_pipe.run, args=(args, agent_id))
                proc.start()
                process.append(proc)
            else:
                evaluator_pipe = None

            proc = mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe, learner_id, agent_id))
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
        if time.time() < self.ensemble_timer + self.ensemble_gap:
            return
        self.ensemble_timer = time.time()

        lock_signal = f'{cwd}/lock'
        os.makedirs(lock_signal, exist_ok=True)
        agent.save_or_load_agent(cwd, if_save=True)
        os.rmdir(lock_signal)

        '''update episode_return_dir_name'''
        recorder = np.load(f"{cwd}/recorder.npy")  # `evaluator.py save_learning_curve()`
        r_avg = recorder[-8:, 1].mean()
        new_r_dir_name = f'episode_return_{r_avg:08.3f}'
        old_r_dir_name = self.get_episode_return_dir_name(cwd)

        os.makedirs(f"{cwd}/{new_r_dir_name}", exist_ok=True)
        os.rmdir(f"{cwd}/{old_r_dir_name}") if old_r_dir_name else None

        '''build ensemble_rs'''
        ensemble_rs = list()
        for i in range(self.ensemble_num):
            r_file = self.get_episode_return_dir_name(f"{cwd[:-2]}{i:02}")
            ensemble_rs.append(float(r_file.split('_')[-1]) if r_file else -2 ** 16)
        ensemble_rs = np.array(ensemble_rs)

        '''move training files'''
        if r_avg == np.max(ensemble_rs):
            move_id = self.agent_id
        elif r_avg == np.min(ensemble_rs):
            move_id = np.argmax(ensemble_rs)
        else:
            soft_max_rs = self.np_soft_max(ensemble_rs)
            move_id = rd.choice(self.ensemble_num, p=soft_max_rs)

        if move_id != self.agent_id:
            other_cwd = cwd[:-2] + f'{move_id}:02'

            while os.path.exists(f"{other_cwd}/lock"):
                time.sleep(1)
            agent.save_or_load_agent(other_cwd, if_save=False)
        print(f"{' '*20}{self.agent_id:2}<-{move_id}    {repr(ensemble_rs.round(2))}")

    @staticmethod
    def get_episode_return_dir_name(cwd):
        r_files = [name
                   for name in os.listdir(cwd)
                   if name.find('episode_return_') == 0]
        return r_files[0] if len(r_files) else None

    @staticmethod
    def np_soft_max(raw_x):
        norm_x = (raw_x - raw_x.mean()) / (raw_x.std() + 1e-6)
        exp_x = np.exp(norm_x) + 1e-6
        return exp_x / exp_x.sum()


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

    def run(self, args, comm_eva, comm_exp, learner_id=0, agent_id=0):
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
        del args

        if_train = True
        while if_train:
            ensemble.run(cwd, agent)  # todo ensemble

            traj_lists = comm_exp.explore(agent)
            if self.learner_num > 1:
                data = self.comm_data(traj_lists, learner_id, round_id=-1)
                traj_lists.extend(data)
            traj_list = sum(traj_lists, list())

            if sys.platform == 'win32':  # Avoid CUDA runtime error (801)
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


'''run'''
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.agents.AgentA2C import AgentA2C
from elegantrl.train.config import Arguments


def demo_continuous_action_on_policy():  # [ElegantRL.2021.11.11]
    env_name = ['Pendulum-v1', 'LunarLanderContinuous-v2',
                'BipedalWalker-v3', 'BipedalWalkerHardcore-v3'][ENV_ID]
    agent_class = [AgentPPO, AgentA2C][DRL_ID]
    args = Arguments(env=build_env(env_name), agent=agent_class())
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
    args.ensemble_gpus = ((0,), (1,), (2,), (3,))
    args.ensemble_gap = 2 ** 8
    args.cwd = './temp'
    args.target_return = 320
    args.if_allow_break = True

    args.learner_gpus = (GPU_ID,)  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    if_use_single_process = 0
    if if_use_single_process:
        train_and_evaluate(args)  # single process
    else:
        train_and_evaluate_em(args)  # multiple process


if __name__ == '__main__':
    sys.argv.extend('3 1 0'.split(' '))
    GPU_ID = eval(sys.argv[1])
    ENV_ID = eval(sys.argv[2])
    DRL_ID = eval(sys.argv[3])
    demo_continuous_action_on_policy()
