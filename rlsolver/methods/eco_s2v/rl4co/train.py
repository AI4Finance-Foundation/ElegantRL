import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../../')
sys.path.append(os.path.dirname(rlsolver_path))

from rlsolver.methods.eco_s2v.rl4co.envs.graph import MaxCutEnv, MaxCutGenerator
from rlsolver.methods.eco_s2v.rl4co.models import S2VModelPolicy, S2VModel
from rlsolver.methods.eco_s2v.rl4co.utils import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary

from rlsolver.methods.eco_s2v.config import *


def run(save_loc, graph_save_loc):
    # Instantiate generator and environment
    pre_fix = save_loc + "/" + NEURAL_NETWORK_PREFIX

    generator = MaxCutGenerator(n_spins=NUM_TRAIN_NODES)
    env = MaxCutEnv(generator)

    # Create policy and RL model
    policy = S2VModelPolicy(env_name=env.name, embed_dim=64, num_encoder_layers=6)

    model = S2VModel(env, policy, batch_size=2, train_data_size=9000, optimizer_kwargs={"lr": 1e-4})

    checkpoint_callback = ModelCheckpoint(
        dirpath=pre_fix,  # 保存路径
        filename=NEURAL_NETWORK_PREFIX + "_{step:06d}",  # 按步数保存
        every_n_train_steps=50,  # 每 50 步保存一次
        save_top_k=-1,  # 保存所有模型
        save_last=False,  # 保存最后一个模型
        mode="max",  # 最大化 reward
    )
    rich_model_summary = RichModelSummary(max_depth=3)  # model summary callback
    callbacks = [checkpoint_callback, rich_model_summary]

    trainer = RL4COTrainer(max_epochs=2, accelerator="gpu",
                           precision="16-mixed", callbacks=callbacks,
                           devices=[TRAIN_GPU_ID], default_root_dir=save_loc + "/rl4co")
    trainer.fit(model)
