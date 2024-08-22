import os
from elegantrl.train.evaluator import Evaluator
from elegantrl.envs.CustomGymEnv import PendulumEnv

EnvArgsPendulum = {'env_name': 'Pendulum-v1', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}


def test_get_rewards_and_steps():
    print("\n| test_get_rewards_and_steps()")
    from elegantrl.train.evaluator import get_rewards_and_steps
    from elegantrl.agents.net import Actor

    env = PendulumEnv()

    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    actor = Actor(dims=[8, 8], state_dim=state_dim, action_dim=action_dim)

    if_render = False
    rewards, steps = get_rewards_and_steps(env=env, actor=actor, if_render=if_render)
    assert isinstance(rewards, float)
    assert isinstance(steps, int)

    if os.name == 'nt':  # if the operating system is Windows NT
        if_render = True
        print("\"libpng warning: iCCP: cHRM chunk does not match sRGB\" → It doesn't matter to see this warning.")
        rewards, steps = get_rewards_and_steps(env=env, actor=actor, if_render=if_render)
        assert isinstance(rewards, float)
        assert isinstance(steps, int)


if __name__ == '__main__':
    print("\n| test_evaluator.py")
    test_get_rewards_and_steps()
