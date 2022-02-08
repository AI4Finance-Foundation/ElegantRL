from elegantrl.agents.AgentBase import AgentBase


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

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=3e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
        G=20,
        M=2,
        N=10,
    ):
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        self.traj_list = [[] for _ in range(env_num)]
        self.G = G
        self.M = M
        self.N = N
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu"
        )
        self.cri_list = [
            self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
            for i in range(self.N)
        ]
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target_list = [deepcopy(self.cri_list[i]) for i in range(N)]
        self.cri_optim_list = [
            torch.optim.Adam(self.cri_list[i].parameters(), learning_rate)
            for i in range(self.N)
        ]
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate)
        assert isinstance(if_per_or_gae, bool)
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env
        self.alpha_log = torch.zeros(
            1, requires_grad=True, device=self.device
        )  # trainable parameter
        self.alpha_optim = torch.optim.Adam([self.alpha_log], lr=learning_rate)
        self.target_entropy = np.log(action_dim)
        self.criterion = torch.nn.MSELoss()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            batch = buffer.sample_batch(batch_size)
            state = torch.Tensor(batch["obs1"]).to(self.device)
            next_s = torch.Tensor(batch["obs2"]).to(self.device)
            action = torch.Tensor(batch["acts"]).to(self.device)
            reward = torch.Tensor(batch["rews"]).unsqueeze(1).to(self.device)
            mask = torch.Tensor(batch["done"]).unsqueeze(1).to(self.device)
            # state, next_s, actions, reward, mask = buffer.sample_batch(batch_size)
            # print(batch_size,reward.shape,mask.shape,action.shape, state.shape, next_s.shape)
            next_a, next_log_prob = self.act.get_action_logprob(
                next_s
            )  # stochastic policy
            g = torch.Generator()
            g.manual_seed(torch.randint(high=10000000, size=(1,))[0].item())
            a = torch.randperm(self.N, generator=g)
            # a = np.random.choice(self.N, self.M, replace=False)
            # print(a[:M])
            q_tmp = [self.cri_target_list[a[j]](next_s, next_a) for j in range(self.M)]
            q_prediction_next_cat = torch.cat(q_tmp, 1)
            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            next_q_with_log_prob = min_q - alpha * next_log_prob
            y_q = reward + (1 - mask) * self.gamma * next_q_with_log_prob
        q_values = [
            self.cri_list[j](state, action) for j in range(self.N)
        ]  # todo ensemble
        q_values_cat = torch.cat(q_values, dim=1)
        y_q = y_q.expand(-1, self.N) if y_q.shape[1] == 1 else y_q
        obj_critic = self.criterion(q_values_cat, y_q) * self.N
        return obj_critic, state
        # return y_q, state,action

    def select_actions(self, state, size, env):
        """
        Select continuous actions for exploration

        :param state: states.shape==(batch_size, state_dim, )
        :return: actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """
        state = state.to(self.device)
        actions = self.act.get_action(state)
        return actions.detach().cpu()

    def cri_multi_train(self, k):
        q_values = self.cri_list[k](self.state, self.action)
        obj = self.criterion(q_values, self.y_q)
        self.cri_optim_list[k].zero_grad()
        obj.backward()
        self.cri_optim_list[k].step()

    def update_net(self, buffer, batch_size, soft_update_tau):
        # buffer.update_now_len()
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        for i in range(self.G):
            alpha = self.alpha_log.cpu().exp().item()
            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            # self.y_q, self.state,self.action = self.get_obj_critic(buffer, batch_size, alpha)
            for q_i in range(self.N):
                self.cri_optim_list[q_i].zero_grad()
            obj_critic.backward()
            if ((i + 1) % self.G == 0) or i == self.G - 1:
                a_noise_pg, logprob = self.act.get_action_logprob(
                    state
                )  # policy gradient
                """objective of alpha (temperature parameter automatic adjustment)"""
                cri_tmp = []
                for j in range(self.N):
                    self.cri_list[j].requires_grad_(False)
                    cri_tmp.append(self.cri_list[j](state, a_noise_pg))
                q_value_pg = torch.cat(cri_tmp, 1)
                q_value_pg = torch.mean(q_value_pg, dim=1, keepdim=True)
                obj_actor = (-q_value_pg + logprob * alpha).mean()  # todo ensemble
                self.act_optim.zero_grad()
                obj_actor.backward()
                for j in range(self.N):
                    self.cri_list[j].requires_grad_(True)
                obj_alpha = -(self.alpha_log * (logprob - 1).detach()).mean()
                self.optim_update(self.alpha_optim, obj_alpha)
            for q_i in range(self.N):
                self.cri_optim_list[q_i].step()
            if ((i + 1) % self.G == 0) or i == self.G - 1:
                self.act_optim.step()
            for q_i in range(self.N):
                self.soft_update(
                    self.cri_target_list[q_i], self.cri_list[q_i], soft_update_tau
                )
        return obj_actor, alpha
