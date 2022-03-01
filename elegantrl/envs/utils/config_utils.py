import os
import yaml
from elegantrl.envs.isaac_tasks import isaacgym_task_map


def load_task_config(env_name: str):
    if env_name not in isaacgym_task_map:
        handle_illegal_environment(env_name)
    config_root = os.path.join(os.getcwd(), "./elegantrl/envs/isaac_configs")
    config_filename = os.path.join(config_root, env_name + ".yaml")
    with open(config_filename) as config_file:
        task_config = yaml.load(config_file, Loader=yaml.SafeLoader)
    return task_config


def handle_illegal_environment(illegal_name: str):
    legal_environment_names = ""
    for env_name in isaacgym_task_map:
        legal_environment_names += env_name + "\n"
    raise NameError(
        f"Incorrect environment name '{illegal_name}' specified for Isaac Gym training.\n"
        + "Choose from one of the following:\n"
        + legal_environment_names
    )
