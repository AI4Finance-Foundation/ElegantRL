from env import *


def check_pendulum_env():
    env = PendulumEnv()
    assert isinstance(env.env_name, str)
    assert isinstance(env.state_dim, int)
    assert isinstance(env.action_dim, int)
    assert isinstance(env.if_discrete, bool)

    state = env.reset()
    assert state.shape == (env.state_dim,)

    action = np.random.uniform(-1, +1, size=env.action_dim)
    state, reward, done, info_dict = env.step(action)
    assert isinstance(state, np.ndarray)
    assert state.shape == (env.state_dim,)
    assert isinstance(state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info_dict, dict) or (info_dict is None)


if __name__ == '__main__':
    check_pendulum_env()
    print('| Finish checking.')
