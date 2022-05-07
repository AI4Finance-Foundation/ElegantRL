from elegantrl.agents.net import QNetDuel
from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.train.config import Arguments

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''


class AgentDuelingDQN(AgentDQN):
    """
    Dueling network.
    """

    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
        self.act_class = getattr(self, "act_class", QNetDuel)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        AgentDQN.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)
