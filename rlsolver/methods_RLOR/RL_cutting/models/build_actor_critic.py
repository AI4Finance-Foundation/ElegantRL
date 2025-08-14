import torch

from models.pg_actors import AttentionPolicy, RNNPolicy, DensePolicy, RandomPolicy, DoubleAttentionPolicy
from models.pg_critics import DenseCritic, NoCritic


def build_actor(policy_params):
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """returns the type of model within policy_params
    """
    # determine the type of model
    if policy_params['model'] == 'dense':
        actor = DensePolicy(**policy_params['model_params']).to(mydevice)
    elif policy_params['model'] == 'rnn':
        actor = RNNPolicy(**policy_params['model_params']).to(mydevice)
    elif policy_params['model'] == 'attention':
        actor = AttentionPolicy(**policy_params['model_params']).to(mydevice)
    elif policy_params['model'] == 'random':
        actor = RandomPolicy(**policy_params['model_params']).to(mydevice)
    elif policy_params['model'] == 'double_attention':
        actor = DoubleAttentionPolicy(**policy_params['model_params']).to(mydevice)
    else:
        raise NotImplementedError

    return actor

def build_critic(critic_params):
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if critic_params['model'] == 'dense':
        critic = DenseCritic(**critic_params['model_params']).to(mydevice)
    elif critic_params['model'] == 'None':
        critic = NoCritic()
    else:
        raise NotImplementedError
    return critic


