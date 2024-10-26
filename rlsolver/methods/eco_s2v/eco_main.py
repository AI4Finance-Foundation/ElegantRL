import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import rlsolver.methods.eco_s2v.train_and_test.inference_eco as inference
import rlsolver.methods.eco_s2v.train_and_test.train_eco as train

from rlsolver.methods.eco_s2v.config.eco_config import *

# 这也是test中保存结果的地方
# save_loc=f"../../result/{GRAPH_TYPE}_{NODES}spin_eco"
save_loc = f"../../result"

train_then_inference = False
if train_then_inference:
    train.run(save_loc, graph_save_loc=GRAPHSAVELOC)
    inference.run(save_loc, graph_save_loc=GRAPHSAVELOC, batched=True, max_batch_size=None, just_test=False)

only_inference = True
if only_inference:
    inference.run(save_loc, network_save_path=NETWORKSAVEPATH,
                  graph_save_loc=GRAPHSAVELOC, batched=True, max_batch_size=None)
