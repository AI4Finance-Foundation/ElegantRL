import gym

gym.logger.set_level(40)  # Block warning

class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        self.env = gym.make(env) if isinstance(env, str) else env
        super().__init__(self.env)

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(self.env, if_print)

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        return state.astype(np.float32)

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        state, reward, done, info_dict = self.env.step(action * self.action_max)
        return state.astype(np.float32), reward, done, info_dict


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    assert isinstance(env, gym.Env)

    env_name = getattr(env, 'env_name', None)
    env_name = env.unwrapped.spec.id if env_name is None else env_name

    if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
        raise RuntimeError("| <class 'gym.spaces.discrete.Discrete'> does not support environment with discrete observation (state) space.")
    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_return = getattr(env.spec, 'reward_threshold', 2 ** 16)

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # for discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # for continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name:  {env_name}, action if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return
