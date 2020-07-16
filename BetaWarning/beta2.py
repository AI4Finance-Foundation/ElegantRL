from AgentRun import *

'''
test DQN DDQN
'''


def run__demo(gpu_id, cwd='RL_BasicAC'):
    from AgentZoo import AgentBasicAC

    args = Arguments(class_agent=AgentBasicAC)
    args.gpu_id = gpu_id

    args.env_name = "Pendulum-v0"
    args.cwd = './{}/Pendulum_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__off_policy(**vars(args))


run__demo(gpu_id=1)
