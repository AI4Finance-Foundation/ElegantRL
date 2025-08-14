import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../')
sys.path.append(os.path.dirname(rlsolver_path))

import torch

from rlsolver.methods.eco_s2v.rl4co.envs.graph import MaxCutEnv
from rlsolver.methods.eco_s2v.rl4co.envs.graph.maxcut.inference_generator import MaxCutGenerator
from rlsolver.methods.eco_s2v.rl4co.models import S2VModel

from rlsolver.methods.eco_s2v.config import *


def run(graph_dir=DATA_DIR, n_sims=NUM_INFERENCE_SIMS):
    torch.set_grad_enabled(False)
    device = INFERENCE_DEVICE

    checkpoint_path = RL4CO_CHECKOUT_DIR
    new_model_checkpoint = S2VModel.load_from_checkpoint(checkpoint_path, strict=False, map_location=device)

    policy_new = new_model_checkpoint.policy.to(device)
    generator = MaxCutGenerator(file=graph_dir, device=device)
    env = MaxCutEnv(generator).to(device)
    td_init = env.reset(batch_size=[n_sims]).to(device)

    out = policy_new(td_init.clone(), env, phase="test", decode_type="greedy")
    out['reward'] = out['reward'] * generator.n_spins
    print(out['reward'])
    print(torch.mean(out['reward']))
    print(torch.max(out['reward']))
