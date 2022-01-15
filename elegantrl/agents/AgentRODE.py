class AgentREDQ(AgentBase):  # [ElegantRL.2021.11.11]
    """
    Bases: ``AgentBase``
    
    Randomized Ensemble Double Q-learning algorithm. “Randomized Ensembled Double Q-Learning: Learning Fast Without A Model”. Xinyue Chen et al.. 2021.
    
    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param reward_scale: scale the reward to get a appropriate scale Q value
    :param gamma: the discount factor of Reinforcement Learning
    :param learning_rate: learning rate of optimizer
    :param if_per_or_gae: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    :param G: Update to date ratio
    :param M: subset size of critics
    :param N: ensemble number of critics
    """
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassCri = Critic
        self.get_obj_critic = self.get_obj_critic_raw
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False
        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda
    