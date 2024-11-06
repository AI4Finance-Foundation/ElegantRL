import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))
from rlsolver.methods.eco_s2v.config.config import *
from rlsolver.methods.eco_s2v.train_and_test import (train_eco,
                                                     train_s2v,
                                                     inference)

# 这也是test中保存结果的地方
# save_loc=f"../../result/{GRAPH_TYPE}_{NODES}spin_eco"
# save_loc = f"../../result"
save_loc = RESULT_DIR

train_network = False
inference_network = True

if train_network:
    if ALG_NAME == "eco":
        train_eco.run(save_loc=RESULT_DIR, graph_save_loc=DATA_DIR)
    elif ALG_NAME == "s2v":
        train_s2v.run(save_loc=RESULT_DIR, graph_save_loc=DATA_DIR)
    else:
        raise ValueError('Algorithm not recognized')

if inference_network:
    inference.run(save_loc=RESULT_DIR, graph_save_loc=DATA_DIR, network_save_path=NETWORK_SAVE_PATH,
                  batched=True, max_batch_size=None, max_parallel_jobs=2, prefixes=INFERENCE_PREFIXES)
