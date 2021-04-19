import numpy as np
import torch
from tensorboardX import SummaryWriter
from copy import deepcopy

"""
By https://github.com/GyChou
"""


class TensorBoard:
    _writer = None

    @classmethod
    def get_writer(cls, load_path=None):
        if cls._writer:
            return cls._writer
        cls._writer = SummaryWriter(load_path)
        return cls._writer


class RecordEpisode:
    def __init__(self):
        self.l_reward = []
        self.record = {}

    def add_record(self, reward, info):
        self.l_reward.append(reward)

    def get_result(self):
        results = {}
        #######Reward#######
        results['reward'] = {}
        rewards = np.array(self.l_reward)
        results['reward'][0] = rewards.sum()
        if len(rewards.shape) > 1:
            for i in range(rewards.shape[1]):
                results['reward'][i + 1] = rewards[:, i].sum()

        #######Total#######
        results['total'] = {}
        results['total']['step'] = rewards.shape[0]
        return results

    def clear(self):
        self.l_reward = []
        self.record = {}


def calc(np_array):
    if len(np_array.shape) > 1:
        np_array = np_array.sum(dim=1)
    return {'avg': np_array.mean(),
            'std': np_array.std(),
            'max': np_array.max(),
            'min': np_array.min(),
            'mid': np.median(np_array)}


class RecordEvaluate:

    def __init__(self):
        self.results = {}

    def add(self, result):
        if len(self.results) == 0:
            self.results = result
            for k in result.keys():
                for i, v in result[k].items():
                    self.results[k][i] = [v]
        else:
            for k in result.keys():
                for i, v in result[k].items():
                    self.results[k][i].append(v)

    def add_many(self, results):
        if len(self.results) == 0:
            self.results = deepcopy(results)
        else:
            for k in results.keys():
                for i, v in results[k].items():
                    self.results[k][i] += results[k][i]

    def eval_result(self):
        result = {}
        for k in self.results.keys():
            result[k] = {}
            for i, v in self.results[k].items():
                result[k][i] = calc(np.array(self.results[k][i]))
        return result

    def clear(self):
        self.results = {}


class Evaluator():
    def __init__(self, args):
        self.cwd = args.cwd
        self.writer = TensorBoard.get_writer(args.cwd)
        self.target_reward = args.env['target_reward']
        self.eval_times = args.evaluator['eval_times']
        self.break_step = args.evaluator['break_step']
        self.satisfy_reward_stop = args.evaluator['satisfy_reward_stop']
        self.pre_eval_times = args.evaluator['pre_eval_times']
        self.device = torch.device('cpu')

        self.record_totalstep = 0
        self.curr_step = 0
        self.record_satisfy_reward = False
        self.curr_max_avg_reward = -1e10
        self.if_save_model = False
        self.total_time = 0

    def update_totalstep(self, totalstep):
        self.curr_step = totalstep
        self.record_totalstep += totalstep

    def analyze_result(self, result):
        avg_reward = result['reward'][0]['avg']
        if avg_reward > self.curr_max_avg_reward:
            self.curr_max_avg_reward = avg_reward
            self.if_save_model = True
            if (self.curr_max_avg_reward > self.target_reward) and (self.satisfy_reward_stop):
                self.record_satisfy_reward = True

    def tb_train(self, train_record):
        for key, value in train_record.items():
            self.writer.add_scalar(f'algorithm/{key}', value, self.record_totalstep - self.curr_step)

    def tb_eval(self, eval_record):
        for k in eval_record.keys():
            for i in eval_record[k].keys():
                for key, value in eval_record[k][i].items():
                    self.writer.add_scalar(f'{k}_{i}/{key}', value, self.record_totalstep - self.curr_step)

    def iter_print(self, train_record, eval_record, use_time):
        print_info = f"|  {'Step':>8}  {'MaxR':>8} |" + \
                     f"  {'avgR':>8}  {'stdR':>8}" + \
                     f"  {'avgS':>6}  {'stdS':>4} |"
        for key in train_record.keys():
            print_info += f"   {key:>8}"
        print_info += " |"
        print(print_info)
        print_info = f"|  {self.record_totalstep:8.2e}  {self.curr_max_avg_reward:8.2f} |" + \
                     f"  {eval_record['reward'][0]['avg']:8.2f}  {eval_record['reward'][0]['std']:8.2f}" + \
                     f"  {eval_record['total']['step']['avg']:6.0f}  {eval_record['total']['step']['std']:4.0f} |"
        for key in train_record.keys():
            print_info += f"   {train_record[key]:8.2f}"
        print_info += " |"
        print(print_info)
        self.total_time += use_time
        print_info = f"| UsedTime:{use_time:8.3f}s  TotalTime:{self.total_time:8.0f}s"
        if self.if_save_model:
            print_info += " |  Save model!"
        print(print_info)

    def save_model(self, act, cri):
        if self.if_save_model:
            if next(act.parameters()).is_cuda:
                act = act.to(self.device)
            if next(cri.parameters()).is_cuda:
                cri = cri.to(self.device)
            act_save_path = f'{self.cwd}/actor.pth'
            torch.save(act.state_dict(), act_save_path)
            cri_save_path = f'{self.cwd}/critic.pth'
            torch.save(cri.state_dict(), cri_save_path)
        self.if_save_model = False
