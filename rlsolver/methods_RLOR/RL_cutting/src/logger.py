import numpy as np
from pathlib import Path
import time
import torch


def get_filename(env_config, policy_params, critic_params, rnd_params, ppo_tag = False):
    start_idx = env_config['idx_list'][0]
    end_idx = env_config['idx_list'][-1]


    if ppo_tag:
        file_dir = f"records/" \
                   f"{env_config['load_dir'][10:]}/" \
                   f"idx_{start_idx}_{end_idx}/" \
                   f"ppo_actor_{policy_params['model']}_critic_{critic_params['model']}_rnd_{rnd_params['model']}/"
    else:
        file_dir = f"records/" \
                   f"{env_config['load_dir'][10:]}/" \
                   f"idx_{start_idx}_{end_idx}/" \
                   f"actor_{policy_params['model']}_critic_{critic_params['model']}_rnd_{rnd_params['model']}/"
    filetime = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"{filetime}.txt"

    return file_dir, file_name, filetime


class RewardLogger(object):
    # todo: when actor critic are combined and ppo config is implemented, get rid of ppo tag
    def __init__(self, env_config, policy_params, critic_params, rnd_params, hyperparameters, ppo_tag = False):
        file_dir, file_name,filetime = get_filename(env_config, policy_params, critic_params, rnd_params, ppo_tag)
        Path(file_dir).mkdir(parents=True, exist_ok=True)
        self.file_dir = file_dir
        self.model_dir = f"{self.file_dir}/models_{filetime}/"
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self.filepath = file_dir + file_name

        with open(self.filepath, "w+") as f:
            f.write(str(policy_params) + "\n")
            f.write(str(critic_params) + "\n")
            f.write(str(rnd_params) + "\n")
            f.write(str(hyperparameters) + "\n")

        # not used but may implement plotting functionality later
        self.reward_record = []

    def record(self, reward_sums):
        """
        :param records: list of trajectory sum of rewards
        :return:
        """
        self.reward_record.extend(reward_sums)
        reward_sums_str = "\n".join(list(map(str, reward_sums)))
        with open(self.filepath, "a") as f:
            f.write(reward_sums_str)

    def save_ppo(self, ppo, ite):
        t = time.strftime("%Y%m%d-%H%M%S")
        actor_filename = f'actor_{ite}_{t}.pt'
        critic_filename = f'critic_{ite}_{t}.pt'

        actor_statedict, critic_statedict = ppo.get_checkpoint()
        torch.save(actor_statedict, self.model_dir + actor_filename)
        if critic_statedict != None:
            torch.save(critic_statedict, self.model_dir + critic_filename)

    def save_rnd(self, rnd, ite):
        t = time.strftime("%Y%m%d-%H%M%S")
        rndt_filename = f'rndtarget_{ite}_{t}.pt'
        rndp_filename = f'rndpred_{ite}_{t}.pt'
        rndt_statedict, rndp_statedict = rnd.get_checkpoint()

        if rndt_statedict != None and rndp_statedict != None:
            torch.save(rndt_statedict, self.model_dir + rndt_filename)
            torch.save(rndp_statedict, self.model_dir + rndp_filename)


