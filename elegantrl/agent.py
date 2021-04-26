import os
import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
from elegantrl.net import QNet, UAQNet

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""

def quantile_huber_loss(x,y, device, kappa=1):

    batch_size = x.shape[0] 
    num_quant = x.shape[1]

    #Get x and y to repeat here
    x = x.unsqueeze(2).repeat(1,1,num_quant)
    y = y.unsqueeze(2).repeat(1,1,num_quant).transpose(1,2)

    tau_hat = torch.linspace(0.0, 1.0 - 1. / num_quant, num_quant) + 0.5 / num_quant
    tau_hat = tau_hat.to(device)
    tau_hat = tau_hat.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1,num_quant)
    
    diff = y-x

    if kappa == 0:
        huber_loss = diff.abs()
    else:
        huber_loss = 0.5 * diff.abs().clamp(min=0.0, max=kappa).pow(2)
        huber_loss += kappa * (diff.abs() - diff.abs().clamp(min=0.0, max=kappa))

    quantile_loss = (tau_hat - (diff < 0).float()).abs() * huber_loss

    return quantile_loss.mean(2).mean(0).sum()


class AgentBase:
    def __init__(self):
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8  # 5e-3 ~= 2 ** -8
        self.state = None  # set for self.update_buffer(), initialize before training
        self.device = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None
        self.criterion = None
        self.get_obj_critic = None

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        `int net_dim` the dimension of networks (the width of neural networks)
        `int state_dim` the dimension of state (the number of state vector)
        `int action_dim` the dimension of action (the number of discrete action)
        `bool if_per` Prioritized Experience Replay for sparse reward
        """

    def select_action(self, state) -> np.ndarray:
        """Select actions for exploration

        :array state: state.shape==(state_dim, )
        :return array action: action.shape==(action_dim, ), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0]
        return action.cpu().numpy()

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        """actor explores in env, then stores the env transition to ReplayBuffer

        :env: RL training environment. env.reset() env.step()
        :buffer: Experience Replay Buffer.
        :int target_step: explored target_step number of step in env
        :float reward_scale: scale reward, 'reward * reward_scale'
        :float gamma: discount factor, 'mask = 0.0 if done else gamma'
        :return int target_step: collected target_step number of step in env
        """
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        `buffer` Experience replay buffer.
        :int target_step: explore target_step number of step in env
        `int batch_size` sample batch_size of data for Stochastic Gradient Descent
        :float repeat_times: the times of sample batch = int(target_step * repeat_times) in off-policy
        :return float obj_a: the objective value of actor
        :return float obj_c: the objective value of critic
        """

    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


'''Value-based Methods (DQN variances)'''


class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.1  # the probability of choosing action randomly in epsilon-greedy
        self.action_dim = None  # chose discrete action randomly in epsilon-greedy

    def init(self, net_dim, state_dim, action_dim, if_per=False, turbulence_threshold=300):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNet(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.act = self.cri  # to keep the same from Actor-Critic framework
        self.turbulence_threshold = turbulence_threshold

        self.criterion = torch.nn.MSELoss(reduction='none' if if_per else 'mean')
        if if_per:
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.get_obj_critic = self.get_obj_critic_raw

    def select_action(self, state) -> int:  # for discrete action space
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = rd.randint(self.action_dim)  # choosing action randomly
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
            action = self.act(states)[0]
            a_int = action.argmax(dim=0).cpu().numpy()
        

        return a_int

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)

            other = (reward * reward_scale, 0.0 if done else gamma, action)  # action is an int
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        q_value = obj_critic = None
        for i in range(int(target_step * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return q_value.mean().item(), obj_critic.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.type(torch.long))
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.type(torch.long))
        obj_critic = (self.criterion(q_value, q_label) * is_weights).mean()
        return obj_critic, q_value
    
    def get_best_act(self, state):
        a_tensor = self.act(state)
        a_tensor = a_tensor.argmax(dim=1)
        return a_tensor.detach().cpu().numpy()[0] 


class AgentUADQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.action_dim = None  # chose discrete action randomly in epsilon-greedy
        self.n_quantiles = 20
        self.aleatoric_penalty = 0.5

    def init(self, net_dim, state_dim, action_dim, kappa=1, prior=0.01):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = UAQNet(state_dim, action_dim*self.n_quantiles, action_dim*self.n_quantiles, net_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.act = self.cri  # to keep the same from Actor-Critic framework

        self.anchor1 = [p.data.clone() for p in list(self.cri.output_1.parameters())]
        self.anchor2 = [p.data.clone() for p in list(self.cri.output_2.parameters())]

        self.criterion = quantile_huber_loss
        self.kappa = kappa
        self.prior = prior
        self.get_obj_critic = self.get_obj_critic_raw

    @torch.no_grad()
    def select_action(self, state) -> int:  # for discrete action space
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        net1, net2 = self.act(states)
        net1 = net1.view(self.action_dim, self.n_quantiles)
        net2 = net2.view(self.action_dim, self.n_quantiles)
        action_means = torch.mean((net1+net2)/2,dim=1)
        epistemic_uncertainties = torch.mean((net1-net2)**2,dim=1)/2
        aleatoric_uncertainties = []
        for i in range(self.action_dim):
            aleatoric_uncertainties.append(np.cov(net1[i], net2[i])[0][1])
        aleatoric_uncertainties = torch.tensor(aleatoric_uncertainties)
        action_means = action_means - self.aleatoric_penalty * aleatoric_uncertainties
        samples = torch.distributions.multivariate_normal.MultivariateNormal(action_means,covariance_matrix=torch.diagflat(epistemic_uncertainties)).sample()
        action = samples.argmax().item()
        return action

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)

            other = (reward * reward_scale, 0.0 if done else gamma, action)  # action is an int
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        q_value = obj_critic = None
        for i in range(int(target_step * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return q_value.mean().item(), obj_critic.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            target1,target2 = self.cri_target(next_s)
            target1 = target1.view(batch_size,self.action_dim,self.n_quantiles)
            target2 = target2.view(batch_size,self.action_dim,self.n_quantiles)

        best_action_idx = torch.mean((target1+target2)/2,dim=2).max(1, True)[1].unsqueeze(2)
        q_label = 0.5*target1.gather(1, best_action_idx.repeat(1,1,self.n_quantiles))+ 0.5*target2.gather(1, best_action_idx.repeat(1,1,self.n_quantiles))

        # Calculate TD target
        td_target = reward.unsqueeze(2).repeat(1,1,self.n_quantiles) \
            + mask.unsqueeze(2).repeat(1,1,self.n_quantiles) * q_label

        out1,out2 = self.cri(state)
        out1 = out1.view(batch_size,self.action_dim,self.n_quantiles) 
        out2 = out2.view(batch_size,self.action_dim,self.n_quantiles) 

        q_value1 = out1.gather(1, action.unsqueeze(2).repeat(1,1,self.n_quantiles).type(torch.long))
        q_value2 = out2.gather(1, action.unsqueeze(2).repeat(1,1,self.n_quantiles).type(torch.long))

        loss1 = self.criterion(q_value1.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss2 = self.criterion(q_value2.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)

        quantile_loss = loss1+loss2

        diff1=[]
        for i, p in enumerate(self.cri.output_1.parameters()):
            diff1.append(torch.sum((p - self.anchor1[i])**2))

        diff2=[]
        for i, p in enumerate(self.cri.output_2.parameters()):
            diff2.append(torch.sum((p - self.anchor2[i])**2))

        diff1 = torch.stack(diff1).sum()
        diff2 = torch.stack(diff2).sum()

        anchor_loss = self.prior*(diff1+diff2)

        loss = quantile_loss + anchor_loss
        
        out_combined = torch.mean((out1+out2)/2,dim=2)

        q_value = out_combined.gather(1, action.type(torch.long))
        return loss, q_value
    
    def get_best_act(self, state):
        net1,net2 = self.act(state)
        net1 = net1.view(self.action_dim,self.n_quantiles)
        net2 = net2.view(self.action_dim,self.n_quantiles)
        action_means = torch.mean((net1+net2)/2,dim=1)
        action = action_means.argmax()
        return action.detach().cpu().numpy()

'''Utils'''


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """output a OU-noise

        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise
