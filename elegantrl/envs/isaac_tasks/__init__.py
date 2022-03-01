# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from elegantrl.envs.isaac_tasks.allegro_hand import AllegroHand
from elegantrl.envs.isaac_tasks.ant import Ant
from elegantrl.envs.isaac_tasks.anymal import Anymal
from elegantrl.envs.isaac_tasks.anymal_terrain import AnymalTerrain
from elegantrl.envs.isaac_tasks.ball_balance import BallBalance
from elegantrl.envs.isaac_tasks.cartpole import Cartpole
from elegantrl.envs.isaac_tasks.franka_cabinet import FrankaCabinet
from elegantrl.envs.isaac_tasks.humanoid import Humanoid
from elegantrl.envs.isaac_tasks.ingenuity import Ingenuity
from elegantrl.envs.isaac_tasks.quadcopter import Quadcopter
from elegantrl.envs.isaac_tasks.shadow_hand import ShadowHand
from elegantrl.envs.isaac_tasks.trifinger import Trifinger

# Mappings from strings to environments
isaacgym_task_map = {
    "AllegroHand": AllegroHand,
    "Ant": Ant,
    "Anymal": Anymal,
    "AnymalTerrain": AnymalTerrain,
    "BallBalance": BallBalance,
    "Cartpole": Cartpole,
    "FrankaCabinet": FrankaCabinet,
    "Humanoid": Humanoid,
    "Ingenuity": Ingenuity,
    "Quadcopter": Quadcopter,
    "ShadowHand": ShadowHand,
    "Trifinger": Trifinger,
}
