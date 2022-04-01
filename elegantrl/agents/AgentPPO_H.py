import numpy as np
import numpy.random as rd
import torch
from elegantrl.agents.AgentPPO import AgentPPO   


class AgentPPO_H(AgentPPO):
    def __init__(
        self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None
    ):
        AgentPPO.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)
        self.lambda_h_term = getattr(args, "lambda_h_term", 2**-3)

        self.h_term_buffer = list()
        self.h_term_r_min_max = (0.0, 1.0)

    def update_net(self, buffer):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [
                ten.to(self.device) for ten in buffer
            ]
            buf_len = buf_state.shape[0]

            """get buf_r_sum, buf_logprob"""
            bs = self.batch_size  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [
                self.cri_target(buf_state[i : i + bs]) for i in range(0, buf_len, bs)
            ]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(
                buf_len, buf_reward, buf_mask, buf_value
            )  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value

            # done_list = [0, ] + list(torch.where(buf_mask.squeeze(1) == 0)[0].detach().cpu().numpy())  # H term
            self.get_buf_h_term(buf_state, buf_action, buf_r_sum)
            del (
                buf_noise,
                buf_reward,
                buf_mask,
                buf_value,
            )

        """update network"""
        obj_critic = None
        obj_actor = None
        assert buf_len >= self.batch_size
        for param_group in self.cri_optimizer.param_groups:
            param_group["lr"] *= 0.9996

        for i in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(
                buf_len,
                size=(self.batch_size,),
                requires_grad=False,
                device=self.device,
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            """PPO: Surrogate objective of Trust Region"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state, action
            )  # it is obj_actor
            obj_entropy *= self.lambda_entropy

            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()

            obj_h_term = self.get_obj_h_term()

            obj_actor = obj_surrogate + obj_entropy + obj_h_term
            self.optimizer_update(self.act_optimizer, obj_actor)

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_buf_h_term(self, buf_state, buf_action, buf_r_sum):
        buf_r_norm = buf_r_sum - buf_r_sum.mean()
        buf_r_diff = (
            torch.where(buf_r_norm[:-1] * buf_r_norm[1:] <= 0)[0].detach().cpu().numpy()
            + 1
        )
        buf_r_diff = list(buf_r_diff) + [
            buf_r_norm.shape[0],
        ]

        step_i = 0
        min_len = 16
        positive_list = list()
        for step_j in buf_r_diff:
            if buf_r_norm[step_i] > 0 and step_i + min_len < step_j:
                positive_list.append((step_i, step_j))
            step_i = step_j

        for step_i, step_j in positive_list:
            index = np.arange(step_i, step_j)

            ten_state = buf_state[index]
            ten_action = buf_action[index]
            ten_r_sum = buf_r_sum[index]

            q_avg = ten_r_sum.mean().item()
            q_min = ten_r_sum.min().item()
            q_max = ten_r_sum.max().item()

            self.h_term_buffer.append(
                (ten_state, ten_action, ten_r_sum, q_avg, q_min, q_max)
            )

        q_arg_sort = np.argsort([item[3] for item in self.h_term_buffer])
        self.h_term_buffer = [
            self.h_term_buffer[i]
            for i in q_arg_sort[max(0, len(self.h_term_buffer) // 4 - 1) :]
        ]

        q_min = np.min(np.array([item[4] for item in self.h_term_buffer]))
        q_max = np.max(np.array([item[5] for item in self.h_term_buffer]))
        self.h_term_r_min_max = (q_min, q_max)

    def get_obj_h_term(self):
        list_len = len(self.h_term_buffer)
        rd_list = rd.choice(list_len, replace=False, size=max(2, list_len // 2))

        ten_state = list()
        ten_action = list()
        ten_r_sum = list()
        for i in rd_list:
            ten_state.append(self.h_term_buffer[i][0])
            ten_action.append(self.h_term_buffer[i][1])
            ten_r_sum.append(self.h_term_buffer[i][2])
        ten_state = torch.vstack(ten_state)  # ten_state.shape == (-1, state_dim)
        ten_action = torch.vstack(ten_action)  # ten_action.shape == (-1, action_dim)
        ten_r_sum = torch.hstack(ten_r_sum)  # ten_r_sum.shape == (-1, )

        """rd sample"""
        ten_size = ten_state.shape[0]
        indices = torch.randint(
            ten_size, size=(ten_size // 2,), requires_grad=False, device=self.device
        )
        ten_state = ten_state[indices]
        ten_action = ten_action[indices]
        ten_r_sum = ten_r_sum[indices]

        """hamilton"""
        ten_logprob = self.act.get_logprob(ten_state, ten_action)
        ten_hamilton = ten_logprob.exp().prod(dim=1)

        n_min, n_max = self.h_term_r_min_max
        ten_r_norm = (ten_r_sum - n_min) / (n_max - n_min)
        return -(ten_hamilton * ten_r_norm).mean() * self.lambda_h_term
