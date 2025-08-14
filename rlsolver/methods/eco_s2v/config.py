import os
from enum import Enum

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../')
from rlsolver.methods.config import GraphType
from rlsolver.methods.util import calc_device


class Alg(Enum):
    eco = 'eco'
    s2v = 's2v'
    eco_torch = 'eco_torch'
    eeco = 'eeco'
    jumanji = 'jumanji'
    rl4co = 'rl4co'


TRAIN_INFERENCE = 0  # 0: train, 1: inference
assert TRAIN_INFERENCE in [0, 1]

ALG = Alg.eeco  # Alg
GRAPH_TYPE = GraphType.BA

# params of training
TRAIN_GPU_ID = 0
SAMPLE_GPU_ID_IN_ECO_S2V = -1 if ALG in [Alg.eco, Alg.s2v, Alg.eeco] else None
USE_TWO_DEVICES_IN_ECO_S2V = True if ALG in [Alg.eco, Alg.s2v, Alg.eeco] else False
BUFFER_GPU_ID = SAMPLE_GPU_ID_IN_ECO_S2V if USE_TWO_DEVICES_IN_ECO_S2V else TRAIN_GPU_ID
NUM_TRAIN_NODES = 20
NUM_TRAIN_SIMS = 2 ** 8
NUM_VALIDATION_NODES = NUM_TRAIN_NODES
VALIDATION_SEED = 10
NUM_VALIDATION_SIMS = 2 ** 4
TEST_SAMPLING_SPEED = True  # False by default

# params of inference
INFERENCE_GPU_ID = 0
NUM_GENERATED_INSTANCES_IN_SELECT_BEST = 10  # select_best_neural_network
NUM_TRAINED_NODES_IN_INFERENCE = 20  # also used in select_best_neural_network
NUM_INFERENCE_NODES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 2000, 3000, 4000, 5000, 10000]
USE_TENSOR_CORE_IN_INFERENCE = True if ALG == Alg.eeco else False
INFERENCE_PREFIXES = [GRAPH_TYPE.value + "_" + str(i) + "_" for i in NUM_INFERENCE_NODES]
# PREFIXES = ["BA_100_", "BA_200_", "BA_300_", "BA_400_", "BA_500_", "BA_600_", "BA_700_", "BA_800_", "BA_900_",
#             "BA_1000_", "BA_1100_", "BA_1200_", "BA_2000_", "BA_3000_", "BA_4000_",
#             "BA_5000_"]  # Replace with your desired prefixes
NUM_INFERENCE_SIMS = 50
MINI_INFERENCE_SIMS = int(0.5 * NUM_INFERENCE_SIMS)  # 如果NUM_INFERENCE_SIMS太大导致GPU内存爆掉，分拆成MINI_INFERENCE_SIMS个环境，跑多次凑够NUM_INFERENCE_SIMS
USE_LOCAL_SEARCH = True if ALG == Alg.eeco else False
LOCAL_SEARCH_FREQUENCY = 10
NEURAL_NETWORK_SAVE_PATH = rlsolver_path + "/methods/eco_s2v/pretrained_agent/" + ALG.value + "_" + GRAPH_TYPE.value + "_" + str(NUM_TRAINED_NODES_IN_INFERENCE) + "spin_best.pth"
DATA_DIR = rlsolver_path + "/data/syn_" + GRAPH_TYPE.value
NEURAL_NETWORK_DIR = rlsolver_path + "/methods/eco_s2v/pretrained_agent/tmp"
NEURAL_NETWORK_SUBFOLDER = "s2v_BA_20spin_s"
NEURAL_NETWORK_FOLDER = rlsolver_path + "/methods/eco_s2v/pretrained_agent/tmp/" + NEURAL_NETWORK_SUBFOLDER
NEURAL_NETWORK_PREFIX = ALG.value + "_" + GRAPH_TYPE.value + "_" + str(NUM_TRAIN_NODES) + "spin"

UPDATE_FREQUENCY = 32
TRAIN_DEVICE = calc_device(TRAIN_GPU_ID)
SAMPLE_DEVICE_IN_ECO_S2V = None if SAMPLE_GPU_ID_IN_ECO_S2V is None else calc_device(SAMPLE_GPU_ID_IN_ECO_S2V)
INFERENCE_DEVICE = calc_device(INFERENCE_GPU_ID)
BUFFER_DEVICE = calc_device(BUFFER_GPU_ID)

if GRAPH_TYPE == GraphType.BA:
    if NUM_TRAIN_NODES == 20:
        NB_STEPS = 500  # 25000
        REPLAY_BUFFER_SIZE = 300
    elif NUM_TRAIN_NODES == 40:
        NB_STEPS = 250000
        REPLAY_BUFFER_SIZE = 5000
    elif NUM_TRAIN_NODES == 60 or NUM_TRAIN_NODES == 80:
        NB_STEPS = 500000
        REPLAY_BUFFER_SIZE = 5000
    elif NUM_TRAIN_NODES == 100:
        NB_STEPS = 800000
        REPLAY_BUFFER_SIZE = 10000
    elif NUM_TRAIN_NODES >= 200:
        NB_STEPS = 1000000
        REPLAY_BUFFER_SIZE = 10 * NUM_TRAIN_NODES * NUM_TRAIN_SIMS
    else:
        raise ValueError("parameters are not set")
elif GRAPH_TYPE == GraphType.ER:
    if NUM_TRAIN_NODES == 20:
        NB_STEPS = 250000
        REPLAY_BUFFER_SIZE = 5000
    elif NUM_TRAIN_NODES == 40:
        NB_STEPS = 250000
        REPLAY_BUFFER_SIZE = 5000
    elif NUM_TRAIN_NODES == 60:
        NB_STEPS = 500000
        REPLAY_BUFFER_SIZE = 5000
    elif NUM_TRAIN_NODES == 100:
        NB_STEPS = 800000
        REPLAY_BUFFER_SIZE = 10000
    elif NUM_TRAIN_NODES >= 200:
        NB_STEPS = 1000000
        REPLAY_BUFFER_SIZE = 70000
    else:
        raise ValueError("parameters are not set")
FINAL_EXPLORATION_STEP = int(0.8 * NB_STEPS)
NUM_TEST_OBJ = 5000
TEST_OBJ_FREQUENCY = max(1, int(NB_STEPS / NUM_TEST_OBJ))
SAVE_NETWORK_FREQUENCY = 10 if NUM_TRAIN_NODES <= 100 else 500  # seconds
if NUM_TRAIN_NODES <= 80:
    UPDATE_TARGET_FREQUENCY = 1000
    REPLAY_START_SIZE = 50
elif NUM_TRAIN_NODES <= 100:
    UPDATE_TARGET_FREQUENCY = 2500
    REPLAY_START_SIZE = 1500
else:
    UPDATE_TARGET_FREQUENCY = 4000
    REPLAY_START_SIZE = 3000  # NUM_TRAIN_NODES*2*NUM_TRAIN_SIMS

# jumanji
JUMANJI_NB_STEPS = 10000
HERIZON_LENGTH = int(NUM_TRAIN_NODES / 2)
JUMANJI_TEST_OBJ_FREQUENCY = 10  # 每次test的时间间隔

# rl4co
RL4CO_GRAPH_DIR = rlsolver_path + "/data/syn_BA/BA_100_ID0.txt"
RL4CO_CHECKOUT_DIR = rlsolver_path + "/methods/eco_s2v/pretrained_agent/tmp/rl4co_BA_20spin/rl4co_BA_20spin_step=000250.ckpt"
