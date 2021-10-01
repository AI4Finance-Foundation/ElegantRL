# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from rlgpu.tasks.cartpole import Cartpole
from rlgpu.tasks.cartpole_y_up import CartpoleYUp
from rlgpu.tasks.ball_balance import BallBalance
from rlgpu.tasks.quadcopter import Quadcopter
from rlgpu.tasks.ant import Ant
from rlgpu.tasks.humanoid import Humanoid
from rlgpu.tasks.franka import FrankaCabinet
from rlgpu.tasks.shadow_hand import ShadowHand
from rlgpu.tasks.ingenuity import Ingenuity
from rlgpu.tasks.anymal import Anymal
from rlgpu.tasks.base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython

from rlgpu.utils.config import warn_task_name

from isaacgym import rlgpu
from rlgpu.utils.config import warn_task_name

import json


def parse_task(args, cfg, cfg_train, sim_params):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    if args.task_type == "C++":
        if args.device == "cpu":
            print("C++ CPU")
            task = rlgpu.create_task_cpu(args.task, json.dumps(cfg_task))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(device_id, -1, args.physics_engine, sim_params)
            else:
                task.init(device_id, device_id, args.physics_engine, sim_params)
            env = VecTaskCPU(task, rl_device, False, cfg_train.get("clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))
        else:
            print("C++ GPU")

            task = rlgpu.create_task_gpu(args.task, json.dumps(cfg_task))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(device_id, -1, args.physics_engine, sim_params)
            else:
                task.init(device_id, device_id, args.physics_engine, sim_params)
            env = VecTaskGPU(task, rl_device, cfg_train.get("clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))

    elif args.task_type == "Python":
        print("Python")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless)
        except NameError as e:
            print(e)
            warn_task_name()
        env = VecTaskPython(task, rl_device)

    return task, env
