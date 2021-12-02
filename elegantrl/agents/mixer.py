import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.nn as nn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QTran(nn.Module):
    def __init__(self, args):
        super(QTran, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        q_input_size = self.state_dim + self.args.rnn_hidden_dim + self.n_actions

        self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                nn.ReLU(),
                                nn.Linear(self.embed_dim, self.embed_dim),
                                nn.ReLU(),
                                nn.Linear(self.embed_dim, 1))

        # V(s)
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                nn.ReLU(),
                                nn.Linear(self.embed_dim, self.embed_dim),
                                nn.ReLU(),
                                nn.Linear(self.embed_dim, 1))
        ae_input = self.args.rnn_hidden_dim + self.n_actions
        self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                nn.ReLU(),
                                                nn.Linear(ae_input, ae_input))

    def forward(self, batch, hidden_states, actions=None):
        bs = batch.batch_size
        ts = batch.max_seq_length

        states = batch["state"].reshape(bs * ts, self.state_dim)

        if self.arch == "coma_critic":
            if actions is None:
                # Use the actions taken by the agents
                actions = batch["actions_onehot"].reshape(bs * ts, self.n_agents * self.n_actions)
            else:
                # It will arrive as (bs, ts, agents, actions), we need to reshape it
                actions = actions.reshape(bs * ts, self.n_agents * self.n_actions)
            inputs = th.cat([states, actions], dim=1)
        elif self.arch == "qtran_paper":
            if actions is None:
                # Use the actions taken by the agents
                actions = batch["actions_onehot"].reshape(bs * ts, self.n_agents, self.n_actions)
            else:
                # It will arrive as (bs, ts, agents, actions), we need to reshape it
                actions = actions.reshape(bs * ts, self.n_agents, self.n_actions)

            hidden_states = hidden_states.reshape(bs * ts, self.n_agents, -1)
            agent_state_action_input = th.cat([hidden_states, actions], dim=2)
            agent_state_action_encoding = self.action_encoding(agent_state_action_input.reshape(bs * ts * self.n_agents, -1)).reshape(bs * ts, self.n_agents, -1)
            agent_state_action_encoding = agent_state_action_encoding.sum(dim=1) # Sum across agents

            inputs = th.cat([states, agent_state_action_encoding], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["state"].reshape(bs * ts, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs, v_outputs
class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        return th.sum(agent_qs, dim=2, keepdim=True)
        
class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot