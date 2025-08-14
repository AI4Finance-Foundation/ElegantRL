from models.ppo import PPOPolicy


def build_ppo(policy_params, critic_params, policy_filepath = None, critic_filepath = None):
    # todo: add more hyperparameters
    epochs = 6
    lr = 0.005
    eps_clip = 0.2

    entropy_coeff = 0.005 # for starter config
    # entropy_coeff = 0.01
    return PPOPolicy(policy_params, critic_params, epochs, lr, eps_clip, entropy_coeff, policy_filepath, critic_filepath)
