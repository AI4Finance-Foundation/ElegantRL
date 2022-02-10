"""
This script lists out all OpenAI gym environments that can be tested on. Some of them
(Ant, Hopper, etc.) require additional external dependencies (mujoco_py, etc.).
"""

from gym import envs

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

for env_id in env_ids:
    print(env_id)
