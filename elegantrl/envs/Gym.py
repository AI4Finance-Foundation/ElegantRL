import gym  # not necessary
import numpy as np

gym.logger.set_level(40)  # Block warning


class PendulumEnv(gym.Wrapper):
    def __init__(self, gym_env_id="Pendulum-v1", target_return=-200):
        # Pendulum-v0 gym.__version__ == 0.17.0
        # Pendulum-v1 gym.__version__ == 0.21.0
        super().__init__(env=gym.make(gym_env_id))

        # from elegantrl.envs.Gym import get_gym_env_info
        # get_gym_env_info(env, if_print=True)  # use this function to print the env information
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 200  # the max step of each episode
        self.state_dim = 3  # feature number of state
        self.action_dim = 1  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = target_return  # episode return is between (-1600, 0)

    def reset(self):
        return self.env.reset()

    def step(self, action: np.ndarray):
        # PendulumEnv set its action space as (-2, +2). It is bad.  # https://github.com/openai/gym/wiki/Pendulum-v0
        # I suggest to set action space as (-1, +1) when you design your own env.
        return self.env.step(action * 2)  # state, reward, done, info_dict


def get_gym_env_args(env, if_print) -> dict:  # [ElegantRL.2021.12.12]
    """get a dict `env_args` about a standard OpenAI gym env information.

    env_args = {
        'env_num': 1,
        'env_name': env_name,            # [str] the environment name, such as XxxXxx-v0
        'max_step': max_step,            # [int] the steps in an episode. (from env.reset to done).
        'state_dim': state_dim,          # [int] the dimension of state
        'action_dim': action_dim,        # [int] the dimension of action
        'if_discrete': if_discrete,      # [bool] action space is discrete or continuous
        'target_return': target_return,  # [float] We train agent to reach this target episode return.
    }

    :param env: a standard OpenAI gym env
    :param if_print: [bool] print the dict about env inforamtion.
    :return: env_args [dict]
    """

    env_num = getattr(env, "env_num") if hasattr(env, "env_num") else 1

    if isinstance(env, gym.Env):
        env_name = getattr(env, "env_name", None)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        state_shape = env.observation_space.shape
        state_dim = (
            state_shape[0] if len(state_shape) == 1 else state_shape
        )  # sometimes state_dim is a list

        target_return = getattr(env, "target_return", None)
        target_return_default = getattr(env.spec, "reward_threshold", None)
        if target_return is None:
            target_return = target_return_default
        if target_return is None:
            target_return = 2**16

        max_step = getattr(env, "max_step", None)
        max_step_default = getattr(env, "_max_episode_steps", None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2**10

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(
            env.action_space, gym.spaces.Box
        ):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            assert not any(env.action_space.high - 1)
            assert not any(env.action_space.low + 1)
        else:
            raise RuntimeError(
                "\n| Error in get_gym_env_info()"
                "\n  Please set these value manually: if_discrete=bool, action_dim=int."
                "\n  And keep action_space in (-1, 1)."
            )
    else:
        env_name = env.env_name
        max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
        target_return = env.target_return

    env_args = {
        "env_num": env_num,
        "env_name": env_name,
        "max_step": max_step,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "if_discrete": if_discrete,
        "target_return": target_return,
    }
    if if_print:
        env_args_repr = repr(env_args)
        env_args_repr = env_args_repr.replace(",", ",\n   ")
        env_args_repr = env_args_repr.replace("{", "{\n    ")
        env_args_repr = env_args_repr.replace("}", ",\n}")
        print(f"env_args = {env_args_repr}")
    return env_args
