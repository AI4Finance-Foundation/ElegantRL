import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import rlsolver.methods.eco_dqn.train_and_test.inference_eco as test
import rlsolver.methods.eco_dqn.train_and_test.train_eco as train

from rlsolver.methods.eco_dqn.config.eco_config import *



save_loc=f"../../result/{GRAPH_TYPE}_{NODES}spin/eco"

train.run(save_loc)

test.run(save_loc, graph_save_loc="../../data/syn_BA", batched=True, max_batch_size=None)
