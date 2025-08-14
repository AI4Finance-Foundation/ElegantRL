import numpy as np
import torch

from models.build_actor_critic import build_actor, build_critic


class PPOPolicy:  # TODO: PPO is usually used with tanh activation functions, but they don't work with parallelization?
    def __init__(self, policy_params, critic_params, epochs, lr, eps_clip, entropy_coeff, policy_filepath= None, critic_filepath = None):

        self.policy = build_actor(policy_params)
        self.critic = build_critic(critic_params)

        if critic_params["model"] != "None":
            self.uses_critic = True
            self.optimizer = torch.optim.Adam([
                {'params': self.policy.model.parameters(), 'lr': lr},
                {'params': self.critic.model.parameters(), 'lr': lr}
            ])
        else:
            # todo: PPO without critic is undefined
            self.uses_critic = False
            self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99, last_epoch=-1, verbose=True)

        self.policy_old = build_actor(policy_params)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.epochs = epochs

        # todo: create parameters for these hardcoded values
        self.eps_clip = eps_clip
        self.entropy_coeff = entropy_coeff

        if policy_filepath != None:
            self.load(policy_filepath, critic_filepath)


    def get_checkpoint(self):
        if self.uses_critic:
            return self.policy_old.state_dict(), self.critic.state_dict()
        return self.policy_old.state_dict(), None

    def load(self, policy_filepath, critic_filepath = None):
        self.policy_old.load_state_dict(torch.load(policy_filepath))
        self.policy.load_state_dict(torch.load(policy_filepath))
        if critic_filepath != None:
            self.critic.load_state_dict(torch.load(critic_filepath))


    # functions used to make rollout decisions
    def _compute_prob_torch(self, state):
        return self.policy_old._compute_prob_torch(state)

    def compute_prob(self, state):
        return self.policy_old.compute_prob(state)

    def _compute_loss(self, memory):
        # computes the loss of a single epoch, given a set of trajectories
        loss = 0
        entropies = []
        for i, state in enumerate(memory.states):
            # compute advantage
            if self.uses_critic:
                baseline = self.critic._compute_value_torch(state)
                advantage = memory.values[i] - baseline.detach()
            else:
                advantage = memory.values[i]

            if len(memory.intrinsic_values) > 0:
                advantage += memory.intrinsic_values[i]

            action = int(memory.actions[i])
            prob = self.policy._compute_prob_torch(state)
            prob_old = self.policy_old._compute_prob_torch(state).detach()

            dist_entropy = torch.distributions.Categorical(prob).entropy()

            k = len(prob)
            action_onehot = np.zeros(k)
            action_onehot[action] = 1
            action_onehot = torch.FloatTensor(action_onehot)

            prob_selected = torch.matmul(prob, action_onehot)
            prob_selected_old = torch.matmul(prob_old, action_onehot)

            ratio = prob_selected / prob_selected_old

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            if self.uses_critic:
                mse = (memory.values[i] - baseline)**2
                loss += torch.min(surr1, surr2) - 0.5 * mse + 0.01 * dist_entropy
            else:
                loss += torch.min(surr1, surr2) + 0.01 * dist_entropy
            entropies.append(dist_entropy.detach())
        loss = -loss / len(memory.states)
        assert loss.requires_grad == True
        return loss

    def train(self, memory):
        # from here, memory.advantages must be computed, however rewards are preprocessed
        # todo: preprocessing needs to be normalized across the code
        for _ in range(self.epochs):
            print("epoch")
            loss = self._compute_loss(memory)
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
        self.scheduler.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


