import os
import yaml
from elegantrl.envs.isaac_tasks import isaacgym_task_map
from typing import Dict


def load_task_config(env_name: str) -> Dict:
    """Loads the configuration for this Isaac Gym environment name from the
    isaac_configs folder.

    Args:
        env_name (str): the name of the Isaac Gym environment whose config we'd like to
            load.

    Returns:
        Dict: the Isaac Gym environment config.
    """
    if env_name not in isaacgym_task_map:
        handle_illegal_environment(env_name)
    config_root = os.path.join(os.getcwd(), "./elegantrl/envs/isaac_configs")
    config_filename = os.path.join(config_root, env_name + ".yaml")
    with open(config_filename) as config_file:
        task_config = yaml.load(config_file, Loader=yaml.SafeLoader)
    return task_config


def handle_illegal_environment(illegal_name: str):
    """Handles an illegal Isaac Gym environment name by returning an error.

    Args:
        illegal_name (str): the name of the illegal environment.

    Raises:
        NameError: the error for trying to instantiate an incorrect environment name.
    """
    legal_environment_names = ""
    for env_name in isaacgym_task_map:
        legal_environment_names += env_name + "\n"
    raise NameError(
        f"Incorrect environment name '{illegal_name}' specified for Isaac Gym training.\n"
        + "Choose from one of the following:\n"
        + legal_environment_names
    )


def get_isaac_env_args(env_name: str) -> Dict:
    """Retrieves env_args for an Isaac Gym environment.

    Args:
        env_name (str): the name of the desired Isaac Gym environment.

    Returns:
        Dict: the env_args of the Isaac Gym environment.
    """
    task_config = load_task_config(env_name)
    env_config = task_config["env"]
    return {
        "env_num": env_config["numEnvs"],
        "env_name": task_config["name"],
        "max_step": get_max_step_from_config(env_config),
        "state_dim": env_config["numObservations"],
        "action_dim": env_config["numActions"],
        "if_discrete": False,
        "target_return": 10**10,
        "device_id": 0,  # set by worker
        "if_print": False,  # don't print out args by default
    }


def get_max_step_from_config(env_config: Dict) -> int:
    """Retrieves max_step from the hard-coded Isaac Gym environment config.

    Args:
        env_config (Dict): the environment config of the Isaac Gym task.

    Returns:
        int: the maximum episode length in steps.
    """
    if "maxEpisodeLength" in env_config:
        return env_config["maxEpisodeLength"]
    elif "episodeLength" in env_config:
        return env_config["episodeLength"]
    else:
        return 1000
