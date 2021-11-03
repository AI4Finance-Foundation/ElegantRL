import sys
from elegantrl.demo import *


def demo_pixel_level_on_policy():  # 2021-09-07
    env_name = ['CarRacingFix', ][ENV_ID]
    agent_class = [AgentPPO, AgentSharePPO, AgentShareA2C][0]
    # args = Arguments(env=build_env(env_name, if_print=True), agent=agent_class())
    args = Arguments(env=env_name, agent=agent_class())

    if env_name == 'CarRacingFix':
        args.state_dim = (112, 112, 6)
        args.action_dim = 6
        args.max_step = 512
        args.if_discrete = False
        args.target_return = 950

        "Step 12e5,  Reward 300,  UsedTime 10ks PPO"
        "Step 20e5,  Reward 700,  UsedTime 25ks PPO"
        "Step 40e5,  Reward 800,  UsedTime 50ks PPO"
        args.agent.ratio_clip = 0.5
        args.agent.explore_rate = 0.75
        args.agent.if_use_cri_target = True

        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.repeat_times = 2 ** 4
        args.learning_rate = 2 ** -17
        args.soft_update_tau = 2 ** -11
        args.batch_size = args.net_dim * 4
        args.if_per_or_gae = True
        args.agent.lambda_gae_adv = 0.96

        args.eval_gap = 2 ** 9
        args.eval_times1 = 2 ** 2
        args.eval_times1 = 2 ** 4
        args.if_allow_break = False
        args.break_step = int(2 ** 22)

        # args.worker_num = 6  # about 96 cores
        args.worker_num = 2  # about 32 cores
        args.target_step = int(args.max_step * 12 / args.worker_num)

    args.learner_gpus = (GPU_ID,)  # single GPU
    args.eval_gpu_id = GPU_ID
    train_and_evaluate_mp(args)


if __name__ == '__main__':
    GPU_ID = 0  # eval(sys.argv[1])
    ENV_ID = 0  # eval(sys.argv[2])
    demo_pixel_level_on_policy()
