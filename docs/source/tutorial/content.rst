Content
========

Related code please refer to https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl/tutorial

Networks: *net.py*
------------------

class QNet(*nn.Module*)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
       def __init__(self, mid_dim, state_dim, action_dim):
           super().__init__()
           self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim))

       def forward(self, state):
           return self.net(state)  # q value

- __init__(*self, mid_dim, state_dim, action_dim*)

The network has four layers with ReLU activation functions, where the input size is ``state_dim`` and the output size is ``action_dim``.

- forward(*self, state*)

Take ``state`` as the input and output Q values.

class QNetTwin(*nn.Module*) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   class QNetTwin(nn.Module):  # Double DQN
       def __init__(self, mid_dim, state_dim, action_dim):
           super().__init__()
           self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU())
           self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, action_dim))  # q1 value
           self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, action_dim))  # q2 value

       def forward(self, state):
           tmp = self.net_state(state)
           return self.net_q1(tmp)  # one Q value

       def get_q1_q2(self, state):
           tmp = self.net_state(state)
           return self.net_q1(tmp), self.net_q2(tmp)  # two Q values

- __init__(*self, mid_dim, state_dim, action_dim*)

There are three networks:

The **net_state** network has two layers,  where the input size is ``state_dim`` and the output size is ``mid_dim``.

The **net_q1** and **net_q2** network has two layers,  where the input size is ``mid_dim`` and the output size is ``action_dim``.

The **net_state** network is connected to both the **net_q1** network and **net_q2** network, all with ReLU activation functions.

- forward(*self, state*)

Take ``state`` as the input and output one Q value.

- get_q1_q2(*self, state*)

Take ``state`` as the input and output two Q values.

class Actor(nn.Module)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:
   
   class Actor(nn.Module):
       def __init__(self, mid_dim, state_dim, action_dim):
           super().__init__()
           self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, action_dim))

       def forward(self, state):
           return self.net(state).tanh()  # action.tanh()

       def get_action(self, state, action_std):
           action = self.net(state).tanh()
           noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
           return (action + noise).clamp(-1.0, 1.0)

- __init__(*self, mid_dim, state_dim, action_dim*)

The network has four layers with ReLU and Hardswish activation functions, where the input size is ``state_dim`` and the output size is ``action_dim``, with ReLU and Hardswish activation functions.

- forward(*self, state*)

Take ``state`` as the input, and apply an additional *tanh()* at last, then output one Q value.

- get_action(*self, state, action_std*)
           
class ActorSAC(nn.Module)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:
   
   class ActorSAC(nn.Module):
      def __init__(self, mid_dim, state_dim, action_dim):
          super().__init__()
          self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
          self.net_a_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                         nn.Linear(mid_dim, action_dim))  # the average of action
          self.net_a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                         nn.Linear(mid_dim, action_dim))  # the log_std of action
          self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

      def forward(self, state):
          tmp = self.net_state(state)
          return self.net_a_avg(tmp).tanh()  # action

      def get_action(self, state):
          t_tmp = self.net_state(state)
          a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
          a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
          return torch.normal(a_avg, a_std).tanh()  # re-parameterize

      def get_action_logprob(self, state):
          t_tmp = self.net_state(state)
          a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
          a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
          a_std = a_std_log.exp()

          noise = torch.randn_like(a_avg, requires_grad=True)
          a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()

          log_prob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
          log_prob = log_prob + (-a_tan.pow(2) + 1.000001).log()  # fix log_prob using the derivative of action.tanh()
          return a_tan, log_prob.sum(1, keepdim=True)

- __init__(*self, mid_dim, state_dim, action_dim*)

There are three networks:

The **net_state** network has two layers,  where the input size is ``state_dim`` and the output size is ``mid_dim``.

The **net_a_avg** and **net_a_std** network has two layers,  where the input size is ``mid_dim`` and the output size is ``action_dim`` and Hardswish activation functions.

The **net_state** network is connected to both the **net_q1** network and **net_q2** network, with ReLU activation functions.

- forward(*self, state*)

Take ``state`` as the input and output an action.

- get_action(*self, state*)

Take ``state`` as the input and output re-parameterize action.

- get_action_logprob(*self, state*)

Take ``state`` as the input and output a action and log probability of that action.

class ActorPPO(nn.Module)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   class ActorPPO(nn.Module):
       def __init__(self, mid_dim, state_dim, action_dim):
           super().__init__()
           self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, action_dim), )

           # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
           self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
           self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

       def forward(self, state):
           return self.net(state).tanh()  # action.tanh()

       def get_action(self, state):
           a_avg = self.net(state)
           a_std = self.a_std_log.exp()

           noise = torch.randn_like(a_avg)
           action = a_avg + noise * a_std
           return action, noise

       def get_logprob_entropy(self, state, action):
           a_avg = self.net(state)
           a_std = self.a_std_log.exp()

           delta = ((a_avg - action) / a_std).pow(2) * 0.5
           logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

           dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
           return logprob, dist_entropy

       def get_old_logprob(self, _action, noise):  # noise = action - a_noise
           delta = noise.pow(2) * 0.5
           return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob
   
- __init__(*self, mid_dim, state_dim, action_dim*)

The network has four layers with ReLU and Hardswish activation functions, where the input size is ``state_dim`` and the output size is ``action_dim``, with ReLU Hardswish activation functions.

- forward(*self, state*)

Take ``state`` as the input and output an action.

- get_action(*self, state*)

Take ``state`` as the input and output re-parameterize action and a noise.

- get_logprob_entropy(*self, state, action*)

Take ``state`` and ``action`` as the input and output log probability and the policy's entropy.

- get_old_logprob(*self, state, action*)

Take ``_action`` and ``noise`` as the input, compute and return the old log probability.


class ActorDiscretePPO(nn.Module)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:
   
   class ActorDiscretePPO(nn.Module):
       def __init__(self, mid_dim, state_dim, action_dim):
           super().__init__()
           self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, action_dim))
           self.action_dim = action_dim
           self.soft_max = nn.Softmax(dim=-1)
           self.Categorical = torch.distributions.Categorical

       def forward(self, state):
           return self.net(state)  # action_prob without softmax

       def get_action(self, state):
           a_prob = self.soft_max(self.net(state))
           # action = Categorical(a_prob).sample()
           samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
           action = samples_2d.reshape(state.size(0))
           return action, a_prob

       def get_logprob_entropy(self, state, a_int):
           a_prob = self.soft_max(self.net(state))
           dist = self.Categorical(a_prob)
           return dist.log_prob(a_int), dist.entropy().mean()

       def get_old_logprob(self, a_int, a_prob):
           dist = self.Categorical(a_prob)
           return dist.log_prob(a_int)
           
- __init__(*self, mid_dim, state_dim, action_dim*)

The network has four layers with ReLU and Hardswish activation functions, where the input size is ``state_dim`` and the output size is ``action_dim``, with ReLU Hardswish activation functions.

- forward(*self, state*)

Take ``state`` as the input and output an action.

- get_action(*self, state*)

Take ``state`` as the input and output re-parameterize action and a noise.

- get_logprob_entropy(*self, state, a_int*)

Take ``state`` and ``a_int`` as the input and output log probability and the policy's entropy.

- get_old_logprob(*self, a_int, a_prob*)

Take ``a_int`` and ``a_prob`` as the input, compute and return the old log probability.
 

class Critic(nn.Module)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   class Critic(nn.Module):
       def __init__(self, mid_dim, state_dim, action_dim):
           super().__init__()
           self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))

       def forward(self, state, action):
           return self.net(torch.cat((state, action), dim=1))  # q value

- __init__(*self, mid_dim, state_dim, action_dim*)

The network has four layers with ReLU activation functions, where the input size is ``state_dim + action_dim`` and the output size is ``1``, with ReLU and Hardswish activation functions.

- forward(*self, state, action*)

Take ``state, action`` as the inputs and output the Q value of that ``action``.

class CriticAdv(nn.Module)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:
   
   class CriticAdv(nn.Module):
       def __init__(self, mid_dim, state_dim, _action_dim):
           super().__init__()
           self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))

       def forward(self, state):
           return self.net(state)  # advantage value

- __init__(*self, mid_dim, state_dim, action_dim*)

The network has four layers with ReLU activation functions, where the input size is ``state_dim`` and the output size is ``1``, with ReLU and Hardswish functions.

- forward(*self, state*)

Take ``state`` as the input and output a Q value.
           
class CriticTwin(nn.Module)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   class CriticTwin(nn.Module):  # shared parameter
       def __init__(self, mid_dim, state_dim, action_dim):
           super().__init__()
           self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())  # concat(state, action)
           self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, 1))  # q1 value
           self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, 1))  # q2 value

       def forward(self, state, action):
           tmp = self.net_sa(torch.cat((state, action), dim=1))
           return self.net_q1(tmp)  # one Q value

       def get_q1_q2(self, state, action):
           tmp = self.net_sa(torch.cat((state, action), dim=1))
           return self.net_q1(tmp), self.net_q2(tmp)  # two Q values

- __init__(*self, mid_dim, state_dim, action_dim*)

There are three networks:

The **net_state** network has two layers,  where the input size is ``state_dim`` and the output size is ``mid_dim``.

The **net_q1** and **net_q2** network has two layers, where the input size is ``mid_dim`` and the output size is ``1``, with Hardswish activation functions.

The **net_state** network is connected to both the **net_q1** network and **net_q2** network, with ReLU activation functions.

- forward(*self, state*)

Take ``state`` as the input and output one Q value.

- get_q1_q2(*self, state*)

Take ``state`` as the input and output two Q values.

Agents: *agent.py*
------------------

class AgentBase
^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   class AgentBase:
       def __init__(self):
           self.state = None
           self.device = None
           self.action_dim = None
           self.if_off_policy = None
           self.explore_noise = None
           self.trajectory_list = None

           self.criterion = torch.nn.SmoothL1Loss()
           self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
           self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

       def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_per_or_gae=False, gpu_id=0):
           # explict call self.init() for multiprocessing
           self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
           self.action_dim = action_dim

           self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
           self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
           self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
           self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

           self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
           self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
           del self.ClassCri, self.ClassAct

       def select_action(self, state) -> np.ndarray:
           states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
           action = self.act(states)[0]
           action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
           return action.detach().cpu().numpy()

       def explore_env(self, env, target_step) -> list:
           state = self.state

           trajectory_list = list()
           for _ in range(target_step):
               action = self.select_action(state)
               next_s, reward, done, _ = env.step(action)
               trajectory_list.append((state, (reward, done, *action)))

               state = env.reset() if done else next_s
           self.state = state
           return trajectory_list

       @staticmethod
       def optim_update(optimizer, objective):
           optimizer.zero_grad()
           objective.backward()
           optimizer.step()

       @staticmethod
       def soft_update(target_net, current_net, tau):
           for tar, cur in zip(target_net.parameters(), current_net.parameters()):
               tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

       def save_or_load_agent(self, cwd, if_save):
           def load_torch_file(model_or_optim, _path):
               state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
               model_or_optim.load_state_dict(state_dict)

           name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                            ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
           name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
           if if_save:
               for name, obj in name_obj_list:
                   save_path = f"{cwd}/{name}.pth"
                   torch.save(obj.state_dict(), save_path)
           else:
               for name, obj in name_obj_list:
                   save_path = f"{cwd}/{name}.pth"
                   load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

- init(*self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_per_or_gae=False, gpu_id=0*)

Initialize the device type, actor and critic network, copy of actor and critic, and Adam optimizer for both network.

- select_action(*self, state*) -> *np.ndarray*

Take ``state`` as input and return the action selected by the actor net.

- explore_env(*self, env, target_step*) -> *list*

Explore the environment ``env`` for ``target_step`` steps, and return the state, action, reward information for each step in a list.

- optim_update(*optimizer, objective*)

- soft_update(*target_net, current_net, tau*)

Update the ``target_net`` data using the ``current_net`` with a learning rate of ``tau``.

- save_or_load_agent(*self, cwd, if_save*)

Load or save the agent in the ``cwd`` directory.

class AgentDQN(AgentBase)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class AgentDQN(AgentBase):
        def __init__(self):
            super().__init__()
            self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy
            self.if_use_cri_target = True
            self.ClassCri = QNet

        def select_action(self, state) -> int:  # for discrete action space
            if rd.rand() < self.explore_rate:  # epsilon-greedy
                a_int = rd.randint(self.action_dim)  # choosing action randomly
            else:
                states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
                action = self.act(states)[0]
                a_int = action.argmax(dim=0).detach().cpu().numpy()
            return a_int

        def explore_env(self, env, target_step) -> list:
            state = self.state

            trajectory_list = list()
            for _ in range(target_step):
                action = self.select_action(state)  # assert isinstance(action, int)
                next_s, reward, done, _ = env.step(action)
                trajectory_list.append((state, (reward, done, action)))

                state = env.reset() if done else next_s
            self.state = state
            return trajectory_list

        def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
            buffer.update_now_len()
            obj_critic = q_value = None
            for _ in range(int(buffer.now_len / batch_size * repeat_times)):
                obj_critic, q_value = self.get_obj_critic(buffer, batch_size)
                self.optim_update(self.cri_optim, obj_critic)
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
            return obj_critic.item(), q_value.mean().item()

        def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q

            q_value = self.cri(state).gather(1, action.long())
            obj_critic = self.criterion(q_value, q_label)
            return obj_critic, q_value

- __init__(self)

Inherit the init from AgentBase class, set the explore rate to 0.25, and set the critic net to **QNet**.

- select_action(*self, state*) -> *int*

Take an input ``state`` and return the index of the best action index.

- explore_env(*self, env, target_step*) -> *list*

Explore the environment ``env`` for ``target_step`` steps, and return the state, action, reward information for each step in a list.

- update_net(*self, buffer, batch_size, repeat_times, soft_update_tau*) -> *tuple*

Update the network and q values for (``buffer`` length / ``batch_size`` * ``repeat_times``) times.

- get_obj_critic(*self, buffer, batch_size*) -> (*torch.Tensor, torch.Tensor*)

Return the critic network and q values.

class AgentDoubleDQN(AgentDQN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:
    
    class AgentDoubleDQN(AgentDQN):
        def __init__(self):
            super().__init__()
            self.softMax = torch.nn.Softmax(dim=1)
            self.ClassCri = QNetTwin

        def select_action(self, state) -> int:  # for discrete action space
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            actions = self.act(states)
            if rd.rand() < self.explore_rate:  # epsilon-greedy
                a_prob = self.softMax(actions)[0].detach().cpu().numpy()
                a_int = rd.choice(self.action_dim, p=a_prob)  # choose action according to Q value
            else:
                action = actions[0]
                a_int = action.argmax(dim=0).detach().cpu().numpy()
            return a_int

        def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q

            q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            return obj_critic, q1

- __init__(*self*)

Initialize the critic network as **QNetTwin**.

- select_action(*self, state*) -> *int*

Take an input ``state`` and return the index of the best action index.

- get_obj_critic(*self, buffer, batch_size*) -> (*torch.Tensor, torch.Tensor*)

Return the critic network and q values from the first Q-net.

class AgentDDPG(AgentBase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class AgentDDPG(AgentBase):
        def __init__(self):
            super().__init__()
            self.explore_noise = 0.1  # explore noise of action
            self.if_use_cri_target = self.if_use_act_target = True
            self.ClassCri = Critic
            self.ClassAct = Actor

        def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
            buffer.update_now_len()
            obj_critic = obj_actor = None
            for _ in range(int(buffer.now_len / batch_size * repeat_times)):
                obj_critic, state = self.get_obj_critic(buffer, batch_size)
                self.optim_update(self.cri_optim, obj_critic)
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

                action_pg = self.act(state)  # policy gradient
                obj_actor = -self.cri(state, action_pg).mean()
                self.optim_update(self.act_optim, obj_actor)
                self.soft_update(self.act_target, self.act, soft_update_tau)
            return obj_actor.item(), obj_critic.item()

        def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_q = self.cri_target(next_s, self.act_target(next_s))
                q_label = reward + mask * next_q
            q_value = self.cri(state, action)
            obj_critic = self.criterion(q_value, q_label)
            return obj_critic, state
            
- __init__(*self*)

Set the explore rate to 0.25, initialize the critic network as **Critic** and actor network as **Actor**.

- update_net(*self, buffer, batch_size, repeat_times, soft_update_tau*)

Update the network and q values for (``buffer`` length / ``batch_size`` * ``repeat_times``) times.

- get_obj_critic(*self, buffer, batch_size*)

Return the critic network and states.

class AgentTD3(AgentBase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class AgentTD3(AgentBase):
        def __init__(self):
            super().__init__()
            self.explore_noise = 0.1  # standard deviation of exploration noise
            self.policy_noise = 0.2  # standard deviation of policy noise
            self.update_freq = 2  # delay update frequency
            self.if_use_cri_target = self.if_use_act_target = True
            self.ClassCri = CriticTwin
            self.ClassAct = Actor

        def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
            buffer.update_now_len()
            obj_critic = obj_actor = None
            for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
                obj_critic, state = self.get_obj_critic(buffer, batch_size)
                self.optim_update(self.cri_optim, obj_critic)

                action_pg = self.act(state)  # policy gradient
                obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target instead of cri for stable training
                self.optim_update(self.act_optim, obj_actor)
                if update_c % self.update_freq == 0:  # delay update
                    self.soft_update(self.cri_target, self.cri, soft_update_tau)
                    self.soft_update(self.act_target, self.act, soft_update_tau)
            return obj_critic.item() / 2, obj_actor.item()

        def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
                q_label = reward + mask * next_q

            q1, q2 = self.cri.get_q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
            return obj_critic, state

class AgentSAC(AgentBase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class AgentSAC(AgentBase):
        def __init__(self):
            super().__init__()
            self.ClassCri = CriticTwin
            self.ClassAct = ActorSAC
            self.if_use_cri_target = True
            self.if_use_act_target = False

            self.alpha_log = None
            self.alpha_optim = None
            self.target_entropy = None

        def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_use_per=False, gpu_id=0, env_num=1):
            super().init(net_dim, state_dim, action_dim, learning_rate, _if_use_per, gpu_id)

            self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                        requires_grad=True, device=self.device)  # trainable parameter
            self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
            self.target_entropy = np.log(action_dim)

        def select_action(self, state):
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            actions = self.act.get_action(states)
            return actions.detach().cpu().numpy()[0]

        def explore_env(self, env, target_step):
            trajectory = list()

            state = self.state
            for _ in range(target_step):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                trajectory.append((state, (reward, done, *action)))
                state = env.reset() if done else next_state
            self.state = state
            return trajectory

        def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
            buffer.update_now_len()

            alpha = self.alpha_log.exp().detach()
            obj_critic = obj_actor = None
            for _ in range(int(buffer.now_len * repeat_times / batch_size)):
                '''objective of critic (loss function of critic)'''
                with torch.no_grad():
                    reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                    next_a, next_log_prob = self.act_target.get_action_logprob(next_s)
                    next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
                    q_label = reward + mask * (next_q + next_log_prob * alpha)
                q1, q2 = self.cri.get_q1_q2(state, action)
                obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
                self.optim_update(self.cri_optim, obj_critic)
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

                '''objective of alpha (temperature parameter automatic adjustment)'''
                action_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
                obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
                self.optim_update(self.alpha_optim, obj_alpha)

                '''objective of actor'''
                alpha = self.alpha_log.exp().detach()
                with torch.no_grad():
                    self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
                obj_actor = -(torch.min(*self.cri_target.get_q1_q2(state, action_pg)) + log_prob * alpha).mean()
                self.optim_update(self.act_optim, obj_actor)

                self.soft_update(self.act_target, self.act, soft_update_tau)

            return obj_critic.item(), obj_actor.item(), alpha.item()

class AgentModSAC(AgentSAC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
        def __init__(self):
            super().__init__()
            self.if_use_act_target = True
            self.if_use_cri_target = True
            self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

        def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
            buffer.update_now_len()

            alpha = self.alpha_log.exp().detach()
            update_a = 0
            obj_actor = None
            for update_c in range(1, int(buffer.now_len * repeat_times / batch_size)):
                '''objective of critic (loss function of critic)'''
                with torch.no_grad():
                    reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

                    next_a, next_log_prob = self.act_target.get_action_logprob(next_s)
                    next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
                    q_label = reward + mask * (next_q + next_log_prob * alpha)
                q1, q2 = self.cri.get_q1_q2(state, action)
                obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
                self.obj_c = 0.995 * self.obj_c + 0.0025 * obj_critic.item()  # for reliable_lambda
                self.optim_update(self.cri_optim, obj_critic)
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

                a_noise_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
                '''objective of alpha (temperature parameter automatic adjustment)'''
                obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
                self.optim_update(self.alpha_optim, obj_alpha)
                with torch.no_grad():
                    self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
                alpha = self.alpha_log.exp().detach()

                '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
                reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
                if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
                if if_update_a:  # auto TTUR
                    update_a += 1

                    q_value_pg = torch.min(*self.cri.get_q1_q2(state, a_noise_pg))
                    obj_actor = -(q_value_pg + log_prob * alpha).mean()
                    self.optim_update(self.act_optim, obj_actor)
                    self.soft_update(self.act_target, self.act, soft_update_tau)
            return self.obj_c, obj_actor.item(), alpha.item()

class AgentPPO(AgentBase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class AgentPPO(AgentBase):
        def __init__(self):
            super().__init__()
            self.ClassCri = CriticAdv
            self.ClassAct = ActorPPO

            self.if_off_policy = False
            self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
            self.lambda_entropy = 0.02  # could be 0.01~0.05
            self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
            self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

        def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, gpu_id=0, env_num=1):
            super().init(net_dim, state_dim, action_dim, learning_rate, if_use_gae, gpu_id)
            self.trajectory_list = list()
            self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

        def select_action(self, state):
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            actions, noises = self.act.get_action(states)
            return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

        def explore_env(self, env, target_step):
            state = self.state

            trajectory_temp = list()
            last_done = 0
            for i in range(target_step):
                action, noise = self.select_action(state)
                next_state, reward, done, _ = env.step(np.tanh(action))
                trajectory_temp.append((state, reward, done, action, noise))
                if done:
                    state = env.reset()
                    last_done = i
                else:
                    state = next_state
            self.state = state

            '''splice list'''
            trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]
            self.trajectory_list = trajectory_temp[last_done:]
            return trajectory_list

        def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
            with torch.no_grad():
                buf_len = buffer[0].shape[0]
                buf_state, buf_action, buf_noise, buf_reward, buf_mask = [ten.to(self.device) for ten in buffer]
                # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer

                '''get buf_r_sum, buf_logprob'''
                bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
                buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
                buf_value = torch.cat(buf_value, dim=0)
                buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

                buf_r_sum, buf_advantage = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
                buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
                del buf_noise, buffer[:]

            '''PPO: Surrogate objective of Trust Region'''
            obj_critic = obj_actor = None
            for _ in range(int(buf_len / batch_size * repeat_times)):
                indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

                state = buf_state[indices]
                action = buf_action[indices]
                r_sum = buf_r_sum[indices]
                logprob = buf_logprob[indices]
                advantage = buf_advantage[indices]

                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = advantage * ratio
                surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                self.optim_update(self.act_optim, obj_actor)

                value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
                obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
                self.optim_update(self.cri_optim, obj_critic)
                self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

            a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1))
            return obj_critic.item(), obj_actor.item(), a_std_log.mean().item()  # logging_tuple

        def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
            buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

            pre_r_sum = 0
            for i in range(buf_len - 1, -1, -1):
                buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
                pre_r_sum = buf_r_sum[i]
            buf_advantage = buf_r_sum - (buf_mask * buf_value[:, 0])
            return buf_r_sum, buf_advantage

        def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
            buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
            buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

            pre_r_sum = 0
            pre_advantage = 0  # advantage value of previous step
            for i in range(buf_len - 1, -1, -1):
                buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
                pre_r_sum = buf_r_sum[i]
                buf_advantage[i] = ten_reward[i] + ten_mask[i] * (pre_advantage - ten_value[i])  # fix a bug here
                pre_advantage = ten_value[i] + buf_advantage[i] * self.lambda_gae_adv
            return buf_r_sum, buf_advantage
            
.. autoclass:: elegantrl.tutorial.agent.AgentPPO
   :members:

.. autoclass:: elegantrl.agent.AgentPPO
   :members:

class AgentDiscretePPO(AgentBase)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class AgentDiscretePPO(AgentPPO):
        def __init__(self):
            super().__init__()
            self.ClassAct = ActorDiscretePPO

        def explore_env(self, env, target_step):
            state = self.state

            trajectory_temp = list()
            last_done = 0
            for i in range(target_step):
                # action, noise = self.select_action(state)
                # next_state, reward, done, _ = env.step(np.tanh(action))
                action, a_prob = self.select_action(state)  # different from `action, noise`
                a_int = int(action)  # different
                next_state, reward, done, _ = env.step(a_int)  # different from `np.tanh(action)`
                trajectory_temp.append((state, reward, done, a_int, a_prob))
                if done:
                    state = env.reset()
                    last_done = i
                else:
                    state = next_state
            self.state = state

            '''splice list'''
            trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]
            self.trajectory_list = trajectory_temp[last_done:]
            return trajectory_list

class ReplayBuffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class ReplayBuffer:
        def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
            self.now_len = 0
            self.next_idx = 0
            self.if_full = False
            self.max_len = max_len
            self.data_type = torch.float32
            self.action_dim = action_dim
            self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

            other_dim = 1 + 1 + self.action_dim
            self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)

            if isinstance(state_dim, int):  # state is pixel
                self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
            elif isinstance(state_dim, tuple):
                self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
            else:
                raise ValueError('state_dim')

        def extend_buffer(self, state, other):  # CPU array to CPU array
            size = len(other)
            next_idx = self.next_idx + size

            if next_idx > self.max_len:
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
                self.if_full = True

                next_idx = next_idx - self.max_len
                self.buf_state[0:next_idx] = state[-next_idx:]
                self.buf_other[0:next_idx] = other[-next_idx:]
            else:
                self.buf_state[self.next_idx:next_idx] = state
                self.buf_other[self.next_idx:next_idx] = other
            self.next_idx = next_idx

        def sample_batch(self, batch_size) -> tuple:
            indices = rd.randint(self.now_len - 1, size=batch_size)
            r_m_a = self.buf_other[indices]
            return (r_m_a[:, 0:1],
                    r_m_a[:, 1:2],
                    r_m_a[:, 2:],
                    self.buf_state[indices],
                    self.buf_state[indices + 1])

        def update_now_len(self):
            self.now_len = self.max_len if self.if_full else self.next_idx

        def save_or_load_history(self, cwd, if_save, buffer_id=0):
            save_path = f"{cwd}/replay_{buffer_id}.npz"

            if if_save:
                self.update_now_len()
                state_dim = self.buf_state.shape[1]
                other_dim = self.buf_other.shape[1]
                buf_state = np.empty((self.max_len, state_dim), dtype=np.float16)  # sometimes np.uint8
                buf_other = np.empty((self.max_len, other_dim), dtype=np.float16)

                temp_len = self.max_len - self.now_len
                buf_state[0:temp_len] = self.buf_state[self.now_len:self.max_len].detach().cpu().numpy()
                buf_other[0:temp_len] = self.buf_other[self.now_len:self.max_len].detach().cpu().numpy()

                buf_state[temp_len:] = self.buf_state[:self.now_len].detach().cpu().numpy()
                buf_other[temp_len:] = self.buf_other[:self.now_len].detach().cpu().numpy()

                np.savez_compressed(save_path, buf_state=buf_state, buf_other=buf_other)
                print(f"| ReplayBuffer save in: {save_path}")
            elif os.path.isfile(save_path):
                buf_dict = np.load(save_path)
                buf_state = buf_dict['buf_state']
                buf_other = buf_dict['buf_other']

                buf_state = torch.as_tensor(buf_state, dtype=torch.float32, device=self.device)
                buf_other = torch.as_tensor(buf_other, dtype=torch.float32, device=self.device)
                self.extend_buffer(buf_state, buf_other)
                self.update_now_len()
                print(f"| ReplayBuffer load: {save_path}")


Environment: *env.py*
---------------------

class PreprocessEnv(gym.Wrapper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class PreprocessEnv(gym.Wrapper):  # environment wrapper
        def __init__(self, env, if_print=True):
            self.env = gym.make(env) if isinstance(env, str) else env
            super().__init__(self.env)

            (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
            self.if_discrete, self.target_return) = get_gym_env_info(self.env, if_print)

        def reset(self) -> np.ndarray:
            state = self.env.reset()
            return state.astype(np.float32)

        def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
            state, reward, done, info_dict = self.env.step(action * self.action_max)
            return state.astype(np.float32), reward, done, info_dict

Main: *run.py*
--------------

class Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class Arguments:
        def __init__(self, agent=None, env=None, if_off_policy=True):
            self.agent = agent  # DRL algorithm
            self.env = env  # env for training

            self.cwd = None  # current work directory. None means set automatically
            self.if_remove = True  # remove the cwd folder? (True, False, None)
            self.break_step = 2 ** 20  # terminate training after 'total_step > break_step'
            self.if_allow_break = True  # terminate training when reaching a target reward

            self.visible_gpu = '0'  # e.g., os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
            self.worker_num = 2  # #rollout workers per GPU
            self.num_threads = 8  # cpu_num to evaluate model, torch.set_num_threads(self.num_threads)

            '''Arguments for training'''
            self.gamma = 0.99  # discount factor
            self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
            self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
            self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

            if if_off_policy:  # (off-policy)
                self.net_dim = 2 ** 8  # the network width
                self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
                self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
                self.target_step = 2 ** 10  # collect target_step, then update network
                self.max_memo = 2 ** 20  # capacity of replay buffer
                self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.
            else:
                self.net_dim = 2 ** 9  # the network width
                self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
                self.repeat_times = 2 ** 3  # collect target_step, then update network
                self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
                self.max_memo = self.target_step  # capacity of replay buffer
                self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.

            '''Arguments for evaluate'''
            self.eval_env = None  # the environment for evaluating. None means set automatically.
            self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
            self.eval_times = 2  # number of times that get episode return in first
            self.random_seed = 0  # initialize random seed in self.init_before_training()

        def init_before_training(self, if_main):
            if self.cwd is None:
                agent_name = self.agent.__class__.__name__
                self.cwd = f'./{agent_name}_{self.env.env_name}_{self.visible_gpu}'

            if if_main:
                import shutil  # remove history according to bool(if_remove)
                if self.if_remove is None:
                    self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
                elif self.if_remove:
                    shutil.rmtree(self.cwd, ignore_errors=True)
                    print(f"| Remove cwd: {self.cwd}")
                os.makedirs(self.cwd, exist_ok=True)

            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            torch.set_num_threads(self.num_threads)
            torch.set_default_dtype(torch.float32)

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)


def train_and_evaluate(args, agent_id=0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    def train_and_evaluate(args, agent_id=0):
        args.init_before_training(if_main=True)

        '''init: Agent'''
        env = args.env
        agent = args.agent
        agent.init(args.net_dim, env.state_dim, env.action_dim, args.learning_rate, args.if_per_or_gae)
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init Evaluator'''
        eval_env = deepcopy(env) if args.eval_env is None else args.eval_env
        evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                            args.eval_times, args.eval_gap)

        '''init ReplayBuffer'''
        if agent.if_off_policy:
            buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                                action_dim=1 if env.if_discrete else env.action_dim)
            buffer.save_or_load_history(args.cwd, if_save=False)

            def update_buffer(_trajectory):
                ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
                ary_other = torch.as_tensor([item[1] for item in _trajectory])
                ary_other[:, 0] = ary_other[:, 0] * reward_scale  # ten_reward
                ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma

                buffer.extend_buffer(ten_state, ary_other)

                _steps = ten_state.shape[0]
                _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, action)
                return _steps, _r_exp
        else:
            buffer = list()

            def update_buffer(_trajectory):
                _trajectory = list(map(list, zip(*_trajectory)))  # 2D-list transpose
                ten_state = torch.as_tensor(_trajectory[0])
                ten_reward = torch.as_tensor(_trajectory[1], dtype=torch.float32) * reward_scale
                ten_mask = (1.0 - torch.as_tensor(_trajectory[2], dtype=torch.float32)) * gamma  # _trajectory[2] = done
                ten_action = torch.as_tensor(_trajectory[3])
                ten_noise = torch.as_tensor(_trajectory[4], dtype=torch.float32)

                buffer[:] = (ten_state, ten_action, ten_noise, ten_reward, ten_mask)

                _steps = ten_reward.shape[0]
                _r_exp = ten_reward.mean()
                return _steps, _r_exp

        '''start training'''
        cwd = args.cwd
        gamma = args.gamma
        break_step = args.break_step
        batch_size = args.batch_size
        target_step = args.target_step
        reward_scale = args.reward_scale
        repeat_times = args.repeat_times
        if_allow_break = args.if_allow_break
        soft_update_tau = args.soft_update_tau
        del args

        agent.state = env.reset()
        if agent.if_off_policy:
            trajectory = agent.explore_env(env, target_step)
            update_buffer(trajectory)

        if_train = True
        while if_train:
            with torch.no_grad():
                trajectory = agent.explore_env(env, target_step)
                steps, r_exp = update_buffer(trajectory)

            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

            with torch.no_grad():
                if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
                if_train = not ((if_allow_break and if_reach_goal)
                                or evaluator.total_step > break_step
                                or os.path.exists(f'{cwd}/stop'))
        print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')
        agent.save_or_load_agent(cwd, if_save=True)
        buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None


class Evaluator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class Evaluator:
        def __init__(self, cwd, agent_id, device, env, eval_times, eval_gap, ):
            self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
            self.recorder_path = f'{cwd}/recorder.npy'
            self.r_max = -np.inf
            self.total_step = 0

            self.env = env
            self.cwd = cwd
            self.device = device
            self.agent_id = agent_id
            self.eval_gap = eval_gap
            self.eval_times = eval_times
            self.target_return = env.target_return

            self.used_time = None
            self.start_time = time.time()
            self.eval_time = 0
            print(f"{'#' * 80}\n"
                f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
                f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                f"{'expR':>8}{'objC':>7}{'etc.':>7}")

        def evaluate_and_save(self, act, steps, r_exp, log_tuple) -> bool:
            self.total_step += steps  # update total training steps

            if time.time() - self.eval_time < self.eval_gap:
                return False  # if_reach_goal

            self.eval_time = time.time()
            rewards_steps_list = [get_episode_return_and_step(self.env, act, self.device) for _ in
                                range(self.eval_times)]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

            if r_avg > self.r_max:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                act_save_path = f'{self.cwd}/actor.pth'
                torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
                print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

            self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                    f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                    f"{'UsedTime':>8}  ########\n"
                    f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                    f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                    f"{self.used_time:>8}  ########")

            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")
            return if_reach_goal

        @staticmethod
        def get_r_avg_std_s_avg_std(rewards_steps_list):
            rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
            r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
            r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
            return r_avg, r_std, s_avg, s_std

def get_episode_return_and_step(env, act, device)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    def get_episode_return_and_step(env, act, device) -> (float, int):
        episode_return = 0.0  # sum of rewards in an episode
        episode_step = 1
        max_step = env.max_step
        if_discrete = env.if_discrete

        state = env.reset()
        for episode_step in range(max_step):
            s_tensor = torch.as_tensor((state,), device=device)
            a_tensor = act(s_tensor)
            if if_discrete:
                a_tensor = a_tensor.argmax(dim=1)
            action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
            state, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                break
        episode_return = getattr(env, 'episode_return', episode_return)
        return episode_return, episode_step


def get_gym_env_info(env, if_print)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
        assert isinstance(env, gym.Env)

        env_name = getattr(env, 'env_name', None)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            raise RuntimeError("| <class 'gym.spaces.discrete.Discrete'> does not support environment with discrete observation (state) space.")
        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        target_return = getattr(env.spec, 'reward_threshold', 2 ** 16)

        max_step = getattr(env, 'max_step', None)
        max_step_default = getattr(env, '_max_episode_steps', None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2 ** 10

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # for discrete action space
            action_dim = env.action_space.n
            action_max = int(1)
        elif isinstance(env.action_space, gym.spaces.Box):  # for continuous action space
            action_dim = env.action_space.shape[0]
            action_max = float(env.action_space.high[0])
            assert not any(env.action_space.high + env.action_space.low)
        else:
            raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

        print(f"\n| env_name:  {env_name}, action if_discrete: {if_discrete}"
            f"\n| state_dim: {state_dim}, action_dim: {action_dim}, action_max: {action_max}"
            f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
<<<<<<< HEAD
        return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return
=======
        return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return
>>>>>>> f9fda245867c2c799640ee9bfe40316904ccd5c4
