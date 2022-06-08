from elegantrl_helloworld.config import Arguments
from elegantrl_helloworld.run import train_and_evaluate
from elegantrl_helloworld.env import get_gym_env_args, PendulumEnv
import yaml

def train_ppo_in_pendulum(gpu_id=0):
    env_name = "Pendulum"
    alg = "PPO"
    with open("config.yml", 'r') as f:
       hyp = yaml.safe_load(f)[alg][env_name]
    env = PendulumEnv()
    env_func = PendulumEnv
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(env_func, env_args, hyp)
    
    train_and_evaluate(args)
    
def train_ppo_in_lunar_lander_or_bipedal_walker(gpu_id=0):
    from elegantrl_helloworld.agent import AgentPPO
    agent_class = AgentPPO
    env_name = ["LunarLanderContinuous-v2", "BipedalWalker-v3"][1]

    if env_name == "LunarLanderContinuous-v2":
        import gym
        env = gym.make(env_name)
        env_func = gym.make
        env_args = get_gym_env_args(env, if_print=True)

        args = Arguments(agent_class, env_func, env_args)

        '''reward shaping'''
        args.gamma = 0.99
        args.reward_scale = 2 ** -1

        '''network update'''
        args.target_step = args.max_step * 8
        args.num_layer = 3
        args.batch_size = 2 ** 7
        args.repeat_times = 2 ** 4
        args.lambda_entropy = 0.04

        '''evaluate'''
        args.eval_gap = 2 ** 6
        args.eval_times = 2 ** 5
        args.break_step = int(4e5)

        args.learner_gpus = gpu_id
        train_agent(args)
        evaluate_agent(args)
        print('| The cumulative returns of LunarLanderContinuous-v2 is ∈ (-1800, (-300, 200), 310+)')

        """
        | Arguments Keep cwd: ./LunarLanderContinuous-v2_PPO_4

        | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
        | `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action.
        | `ObjC` denotes the objective of Critic network. Or call it loss function of critic network.
        | `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network.
        """

    elif env_name == "BipedalWalker-v3":
        import gym
        env = gym.make(env_name)
        env_func = gym.make
        env_args = get_gym_env_args(env, if_print=True)

        args = Arguments(agent_class, env_func, env_args)

        '''reward shaping'''
        args.reward_scale = 2 ** -1
        args.gamma = 0.98

        '''network update'''
        args.target_step = args.max_step
        args.net_dim = 2 ** 8
        args.num_layer = 3
        args.batch_size = 2 ** 8
        args.repeat_times = 2 ** 4

        '''evaluate'''
        args.eval_gap = 2 ** 6
        args.eval_times = 2 ** 4
        args.break_step = int(1e6)

        args.learner_gpus = gpu_id
        args.random_seed += gpu_id
        train_agent(args)
        evaluate_agent(args)
        print('| The cumulative returns of BipedalWalker-v3 is ∈ (-150, (-100, 280), 320+)')

        """
        | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
        | `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action.
        | `ObjC` denotes the objective of Critic network. Or call it loss function of critic network.
        | `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network.
        """

if __name__ == "__main__":
    train_ppo_in_pendulum()
    train_ppo_in_lunar_lander_or_bipedal_walker()
