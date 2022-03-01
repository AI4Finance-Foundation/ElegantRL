import sys

from elegantrl.envs.isaac_integration.Isaac_Envs import *


def create_isaac_vec_environment(env_name: str):
    isaac_env = IsaacVecEnv(env_name)
    del isaac_env


if __name__ == "__main__":
    env_name = sys.argv[1]
    create_isaac_vec_environment(env_name)
