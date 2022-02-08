from copy import deepcopy

import numpy as np
import torch

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorBiConv, CriticBiConv, ShareBiConv
from typing import Tuple


class AgentStep1AC(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = ActorBiConv
        self.ClassCri = CriticBiConv
        self.if_use_cri_target = False
        self.if_use_act_target = False
        self.explore_noise = 2**-8
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )
        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.MSELoss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw
        self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        action = self.act.get_action(state.to(self.device), self.explore_noise)
        return action.detach().cpu()

    def update_net(
        self, buffer, batch_size, repeat_times, soft_update_tau
    ) -> Tuple[float, float]:
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        for update_c in range(1, int(buffer.now_len / batch_size * repeat_times)):
            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.obj_critic = (
                0.99 * self.obj_critic + 0.01 * obj_critic.item()
            )  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            reliable_lambda = np.exp(-self.obj_critic**2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

            obj_actor = -self.cri(state, self.act(state)).mean()  # policy gradient
            self.optim_update(self.act_optim, obj_actor)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            q_label, action, state = buffer.sample_batch_one_step(batch_size)

        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            q_label, action, state, is_weights = buffer.sample_batch_one_step(
                batch_size
            )

        q_value = self.cri(state, action)
        td_error = self.criterion(
            q_value, q_label
        )  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value


class AgentShareStep1AC(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = ShareBiConv
        self.ClassCri = self.ClassAct
        self.if_use_cri_target = True
        self.if_use_act_target = True
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )
        self.act = self.cri = self.ClassAct(net_dim, state_dim, action_dim).to(
            self.device
        )
        if self.if_use_act_target:
            self.act_target = self.cri_target = deepcopy(self.act)
        else:
            self.act_target = self.cri_target = self.act

        self.cri_optim = torch.optim.Adam(
            [
                {"params": self.act.enc_s.parameters(), "lr": learning_rate * 1.25},
                {
                    "params": self.act.enc_a.parameters(),
                },
                {"params": self.act.mid_n.parameters(), "lr": learning_rate * 1.25},
                {
                    "params": self.act.dec_a.parameters(),
                },
                {
                    "params": self.act.dec_q.parameters(),
                },
            ],
            lr=learning_rate,
        )
        self.act_optim = self.cri_optim

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.MSELoss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        action = self.act.get_action(state.to(self.device), self.explore_noise)
        return action.detach().cpu()

    def update_net(
        self, buffer, batch_size, repeat_times, soft_update_tau
    ) -> Tuple[float, float]:
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        update_a = 0
        for update_c in range(1, int(buffer.now_len / batch_size * repeat_times)):
            """objective of critic"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.obj_critic = (
                0.995 * self.obj_critic + 0.005 * obj_critic.item()
            )  # for reliable_lambda
            reliable_lambda = np.exp(-self.obj_critic**2)  # for reliable_lambda

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                action_pg = self.act(state)  # policy gradient
                obj_actor = -self.act_target.critic(state, action_pg).mean()

                obj_united = obj_critic + obj_actor * reliable_lambda
            else:
                obj_united = obj_critic

            self.optim_update(self.cri_optim, obj_united)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return obj_critic.item(), obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            q_label, action, state = buffer.sample_batch_one_step(batch_size)

        q_value = self.act.critic(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            q_label, action, state, is_weights = buffer.sample_batch_one_step(
                batch_size
            )

        q_value = self.act.critic(state, action)
        td_error = self.criterion(
            q_value, q_label
        )  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value
