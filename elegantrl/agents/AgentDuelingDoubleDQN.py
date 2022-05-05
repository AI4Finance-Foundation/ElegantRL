from elegantrl.agents.net import QNetTwinDuel
from elegantrl.agents.AgentDoubleDQN import AgentDoubleDQN

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''


class AgentDuelingDoubleDQN(AgentDoubleDQN):
    """
    Dueling Double Deep Q network. (D3QN)
    """

    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        self.act_class = getattr(self, "act_class", QNetTwinDuel)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        AgentDoubleDQN.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)
