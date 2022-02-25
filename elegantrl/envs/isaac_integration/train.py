# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import isaacgym

import os
from typing import Dict

from elegantrl.envs.isaac_integration.utils.utils import (
    set_np_formatting,
    set_seed,
)

from elegantrl.envs.isaac_integration.utils.rlgames_utils import (
    RLGPUEnv,
    RLGPUAlgoObserver,
    get_rlgames_env_creator,
)

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

main_config = {
    "task_name": "Cartpole",
    "experiment": "",
    "num_envs": "",
    "seed": 42,
    "torch_deterministic": False,
    "max_iterations": "",
    "physics_engine": "physx",
    "pipeline": "gpu",
    "sim_device": "cuda:0",
    "rl_device": "cuda:0",
    "graphics_device_id": 0,
    "num_threads": 4,
    "solver_type": 1,
    "num_subscenes": 4,
    "test": False,
    "checkpoint": "",
    "multi_gpu": False,
    "headless": False,
}

task_config = {
    "name": "Cartpole",
    "physics_engine": "physx",
    "env": {
        "numEnvs": 512,
        "envSpacing": 4.0,
        "resetDist": 3.0,
        "maxEffort": 400.0,
        "clipObservations": 5.0,
        "clipActions": 1.0,
        "asset": {"assetRoot": "assets", "assetFileName": "cartpole.urdf"},
        "enableCameraSensors": False,
    },
    "sim": {
        "dt": 0.0166,
        "substeps": 2,
        "up_axis": "z",
        "use_gpu_pipeline": True,
        "gravity": [0.0, 0.0, -9.81],
        "physx": {
            "num_threads": 4,
            "solver_type": 1,
            "use_gpu": True,
            "num_position_iterations": 4,
            "num_velocity_iterations": 0,
            "contact_offset": 0.02,
            "rest_offset": 0.001,
            "bounce_threshold_velocity": 0.2,
            "max_depenetration_velocity": 100.0,
            "default_buffer_size_multiplier": 2.0,
            "max_gpu_contact_pairs": 1048576,
            "num_subscenes": 4,
            "contact_collection": 0,
        },
    },
    "task": {"randomize": False},
}

train_config = {
    "params": {
        "seed": 42,
        "algo": {"name": "a2c_continuous"},
        "model": {"name": "continuous_a2c_logstd"},
        "network": {
            "name": "actor_critic",
            "separate": False,
            "space": {
                "continuous": {
                    "mu_activation": "None",
                    "sigma_activation": "None",
                    "mu_init": {"name": "default"},
                    "sigma_init": {"name": "const_initializer", "val": 0},
                    "fixed_sigma": True,
                }
            },
            "mlp": {
                "units": [32, 32],
                "activation": "elu",
                "initializer": {"name": "default"},
                "regularizer": {"name": "None"},
            },
        },
        "load_checkpoint": False,
        "load_path": "",
        "config": {
            "name": "Cartpole",
            "full_experiment_name": "Cartpole",
            "env_name": "rlgpu",
            "ppo": True,
            "mixed_precision": False,
            "normalize_input": True,
            "normalize_value": True,
            "num_actors": 512,
            "reward_shaper": {"scale_value": 0.1},
            "normalize_advantage": True,
            "gamma": 0.99,
            "tau": 0.95,
            "learning_rate": 0.0003,
            "lr_schedule": "adaptive",
            "kl_threshold": 0.008,
            "score_to_win": 20000,
            "max_epochs": 100,
            "save_best_after": 50,
            "save_frequency": 25,
            "grad_norm": 1.0,
            "entropy_coef": 0.0,
            "truncate_grads": True,
            "e_clip": 0.2,
            "horizon_length": 16,
            "minibatch_size": 8192,
            "mini_epochs": 8,
            "critic_coef": 4,
            "clip_value": True,
            "seq_len": 4,
            "bounds_loss_coef": 0.0001,
        },
    }
}


def prepare_training(main_config: Dict, task_config: Dict, train_config: Dict):
    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    main_config["seed"] = set_seed(
        seed=main_config["seed"], torch_deterministic=main_config["torch_deterministic"]
    )

    # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
    # We use the helper function here to specify the environment config.
    create_rlgpu_env = get_rlgames_env_creator(
        task_config=task_config,
        task_name=main_config["task_name"],
        sim_device=main_config["sim_device"],
        rl_device=main_config["rl_device"],
        graphics_device_id=main_config["graphics_device_id"],
        headless=main_config["headless"],
        multi_gpu=main_config["multi_gpu"],
    )

    # register the rl-games adapter to use inside the runner
    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: create_rlgpu_env(**kwargs),
        },
    )

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = Runner(RLGPUAlgoObserver())
    runner.load(yaml_conf=train_config)
    runner.reset()

    # dump config dict
    experiment_dir = os.path.join("runs", train_config["params"]["config"]["name"])
    os.makedirs(experiment_dir, exist_ok=True)

    runner.run(
        {
            "train": not main_config["test"],
            "play": main_config["test"],
        }
    )


if __name__ == "__main__":
    prepare_training(
        main_config=main_config, task_config=task_config, train_config=train_config
    )
