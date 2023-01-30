import sys

from elegantrl.envs.IsaacGym import *


def create_isaac_vec_environment(env_name: str):
    isaac_env = IsaacVecEnv(env_name)
    del isaac_env


if __name__ == "__main__":
    env_name = sys.argv[1]
    create_isaac_vec_environment(env_name)
