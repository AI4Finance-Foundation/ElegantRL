import ray
import torch
import os
import time
import numpy as np
import numpy.random as rd
import datetime
from ray_elegantrl.buffer import ReplayBuffer, ReplayBufferMP
from ray_elegantrl.evaluate import RecordEpisode, RecordEvaluate, Evaluator
from ray_elegantrl.config import default_config

"""
Modify [ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)
by https://github.com/GyChou
"""


class Arguments:
    def __init__(self, configs=default_config):
        self.gpu_id = configs['gpu_id']  # choose the GPU for running. gpu_id is None means set it automatically
        # current work directory. cwd is None means set it automatically
        self.cwd = configs['cwd'] if 'cwd' in configs.keys() else None
        # current work directory with time.
        self.if_cwd_time = configs['if_cwd_time'] if 'cwd' in configs.keys() else False
        # initialize random seed in self.init_before_training()

        self.random_seed = 0
        # id state_dim action_dim reward_dim target_reward horizon_step
        self.env = configs['env']
        # Deep Reinforcement Learning algorithm
        self.agent = configs['agent']
        self.agent['agent_name'] = self.agent['class_name']().__class__.__name__
        self.trainer = configs['trainer']
        self.interactor = configs['interactor']
        self.buffer = configs['buffer']
        self.evaluator = configs['evaluator']

        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)

        '''if_per_explore'''
        if self.buffer['if_on_policy']:
            self.if_per_explore = False
        else:
            self.if_per_explore = True

    def init_before_training(self, if_main=True):
        '''set gpu_id automatically'''
        if self.gpu_id is None:  # set gpu_id automatically
            import sys
            self.gpu_id = sys.argv[-1][-4]
        else:
            self.gpu_id = str(self.gpu_id)
        if not self.gpu_id.isdigit():  # set gpu_id as '0' in default
            self.gpu_id = '0'

        '''set cwd automatically'''
        if self.cwd is None:
            if self.if_cwd_time:
                curr_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            else:
                curr_time = 'current'
            self.cwd = f'./logs/{self.env["id"]}/{self.agent["agent_name"]}/exp_{curr_time}_cuda:{self.gpu_id}'

        if if_main:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


def make_env(env_dict, id=None):
    import gym
    import pybullet_envs
    import gym_carla_feature
    if 'params_name' in env_dict:
        env_dict['params_name']['params']['port'] = env_dict['params_name']['params']['port'] + id * 4
        env = gym.make(env_dict['id'], **env_dict['params_name'])
    else:
        env = gym.make(env_dict['id'])
    return env


@ray.remote
class InterActor(object):

    def __init__(self, id, args):
        self.id = id
        args.init_before_training(if_main=False)
        self.env = make_env(args.env, self.id)
        self.env_max_step = args.env['max_step']
        self.reward_scale = args.interactor['reward_scale']
        self._horizon_step = args.interactor['horizon_step'] // args.interactor['rollout_num']
        self.gamma = args.interactor['gamma'] if type(args.interactor['gamma']) is np.ndarray else np.ones(
            args.env['reward_dim']) * args.interactor['gamma']
        self.action_dim = args.env['action_dim']
        self.if_discrete_action = args.env['if_discrete_action']
        if args.agent['agent_name'] in ['AgentPPO']:
            self.modify_action = lambda x: np.tanh(x)
        else:
            self.modify_action = lambda x: x
        self.buffer = ReplayBuffer(
            max_len=args.buffer['max_buf'] // args.interactor['rollout_num'] + args.env['max_step'],
            if_on_policy=args.buffer['if_on_policy'],
            state_dim=args.env['state_dim'],
            action_dim=1 if self.if_discrete_action else args.env['action_dim'],
            reward_dim=args.env['reward_dim'],
            if_per=False,
            if_gpu=False)

        self.record_episode = RecordEpisode()

    @ray.method(num_returns=1)
    def explore_env(self, select_action, policy):
        self.buffer.empty_buffer_before_explore()
        actual_step = 0
        while actual_step < self._horizon_step:
            state = self.env.reset()
            for i in range(self.env_max_step):
                action = select_action(state, policy)
                next_s, reward, done, _ = self.env.step(self.modify_action(action))
                done = True if i == (self.env_max_step - 1) else done
                self.buffer.append_buffer(state,
                                          action,
                                          reward * self.reward_scale,
                                          np.zeros(self.gamma.shape) if done else self.gamma)
                actual_step += 1
                if done:
                    break
                state = next_s
        self.buffer.update_now_len_before_sample()
        return actual_step, \
               self.buffer.buf_state[:self.buffer.now_len], \
               self.buffer.buf_action[:self.buffer.now_len], \
               self.buffer.buf_reward[:self.buffer.now_len], \
               self.buffer.buf_gamma[:self.buffer.now_len]

    @ray.method(num_returns=1)
    def random_explore_env(self, r_horizon_step=None):
        self.buffer.empty_buffer_before_explore()
        if r_horizon_step is None:
            r_horizon_step = self._horizon_step
        else:
            r_horizon_step = max(min(r_horizon_step, self.buffer.max_len - 1), self._horizon_step)
        actual_step = 0
        while actual_step < r_horizon_step:
            state = self.env.reset()
            for _ in range(self.env_max_step):
                action = rd.randint(self.action_dim) if self.if_discrete_action else rd.uniform(-1, 1,
                                                                                                size=self.action_dim)
                next_s, reward, done, _ = self.env.step(self.modify_action(action))
                self.buffer.append_buffer(state,
                                          action,
                                          reward * self.reward_scale,
                                          np.zeros(self.gamma.shape) if done else self.gamma)
                actual_step += 1
                if done:
                    break
                state = next_s
        self.buffer.update_now_len_before_sample()
        return self.buffer.buf_state[:self.buffer.now_len], \
               self.buffer.buf_action[:self.buffer.now_len], \
               self.buffer.buf_reward[:self.buffer.now_len], \
               self.buffer.buf_gamma[:self.buffer.now_len]

    def exploite_env(self, policy, eval_times):
        self.record_episode.clear()
        eval_record = RecordEvaluate()
        for _ in range(eval_times):
            state = self.env.reset()
            for _ in range(self.env_max_step):
                action = policy(torch.as_tensor((state,), dtype=torch.float32).detach_())
                next_s, reward, done, info = self.env.step(action.detach().numpy()[0])
                self.record_episode.add_record(reward, info)
                if done:
                    break
                state = next_s
            eval_record.add(self.record_episode.get_result())
            self.record_episode.clear()
        return eval_record.results


class Trainer(object):

    def __init__(self, args_trainer, agent, buffer):
        self.agent = agent
        self.buffer = buffer
        self.sample_step = args_trainer['sample_step']
        self.batch_size = args_trainer['batch_size']
        self.policy_reuse = args_trainer['policy_reuse']

    def train(self):
        self.agent.act.to(device=self.agent.device)
        self.agent.cri.to(device=self.agent.device)
        train_record = self.agent.update_net(self.buffer, self.sample_step, self.batch_size, self.policy_reuse)
        if self.buffer.if_on_policy:
            self.buffer.empty_buffer_before_explore()
        return train_record


def beginer(config, params=None):
    args = Arguments(config)
    args.init_before_training()
    args_id = ray.put(args)
    #######Init######
    agent = args.agent['class_name'](args.agent)
    agent.init(args.agent['net_dim'],
               args.env['state_dim'],
               args.env['action_dim'],
               args.env['reward_dim'],
               args.buffer['if_per'])
    interactors = [InterActor.remote(i, args_id) for i in range(args.interactor['rollout_num'])]
    buffer_mp = ReplayBufferMP(
        max_len=args.buffer['max_buf'] + args.env['max_step'] * args.interactor['rollout_num'],
        state_dim=args.env['state_dim'],
        action_dim=1 if args.env['if_discrete_action'] else args.env['action_dim'],
        reward_dim=args.env['reward_dim'],
        if_on_policy=args.buffer['if_on_policy'],
        if_per=args.buffer['if_per'],
        rollout_num=args.interactor['rollout_num'])
    trainer = Trainer(args.trainer, agent, buffer_mp)
    evaluator = Evaluator(args)
    rollout_num = args.interactor['rollout_num']

    #######Random Explore Before Interacting#######
    if args.if_per_explore:
        episodes_ids = [interactors[i].random_explore_env.remote() for i in range(rollout_num)]
        assert len(episodes_ids) > 0
        for i in range(len(episodes_ids)):
            done_id, episodes_ids = ray.wait(episodes_ids)
            buf_state, buf_action, buf_reward, buf_gamma = ray.get(done_id[0])
            buffer_mp.extend_buffer(buf_state, buf_action, buf_reward, buf_gamma, i)

    #######Interacting Begining#######
    start_time = time.time()
    policy_id = ray.put(agent.act.to('cpu'))
    while (evaluator.record_totalstep < evaluator.break_step) or (evaluator.record_satisfy_reward):
        #######Explore Environment#######
        episodes_ids = [interactors[i].explore_env.remote(agent.select_action, policy_id) for i in
                        range(rollout_num)]
        assert len(episodes_ids) > 0
        sample_step = 0
        for i in range(len(episodes_ids)):
            done_id, episodes_ids = ray.wait(episodes_ids)
            actual_step, buf_state, buf_action, buf_reward, buf_gamma = ray.get(done_id[0])
            sample_step += actual_step
            buffer_mp.extend_buffer(buf_state, buf_action, buf_reward, buf_gamma, i)
        evaluator.update_totalstep(sample_step)
        #######Training#######
        trian_record = trainer.train()
        evaluator.tb_train(trian_record)
        #######Evaluate#######
        policy_id = ray.put(agent.act.to('cpu'))
        evalRecorder = RecordEvaluate()
        if_eval = True
        #######pre-eval#######
        if evaluator.pre_eval_times > 0:
            eval_results = ray.get(
                [interactors[i].exploite_env.remote(policy_id, eval_times=evaluator.pre_eval_times) for i in
                 range(rollout_num)])
            for eval_result in eval_results:
                evalRecorder.add_many(eval_result)
            eval_record = evalRecorder.eval_result()
            if eval_record['reward'][0]['max'] < evaluator.target_reward:
                if_eval = False
                evaluator.tb_eval(eval_record)
        #######eval#######
        if if_eval:
            eval_results = ray.get(
                [interactors[i].exploite_env.remote(policy_id, eval_times=(evaluator.eval_times))
                 for i in range(rollout_num)])
            for eval_result in eval_results:
                evalRecorder.add_many(eval_result)
            eval_record = evalRecorder.eval_result()
            evaluator.tb_eval(eval_record)
        #######Save Model#######
        evaluator.analyze_result(eval_record)
        evaluator.iter_print(trian_record, eval_record, use_time=(time.time() - start_time))
        evaluator.save_model(agent.act, agent.cri)
        start_time = time.time()

    print(f'#######Experiment Finished!\t TotalTime:{evaluator.total_time:8.0f}s #######')
