import sys
from tutorial_DDPG import *

GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
train_ddpg_for_pendulum(gpu_id=GPU_ID)
