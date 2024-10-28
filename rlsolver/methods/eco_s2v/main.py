import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))
from rlsolver.methods.eco_s2v.config.config import *
import rlsolver.methods.eco_s2v.train_and_test.inference as test

if ALGNAME == 'eco':
    import rlsolver.methods.eco_s2v.train_and_test.train_eco as train
elif ALGNAME == 's2v':
    import rlsolver.methods.eco_s2v.train_and_test.train_s2v as train
else:
    raise ValueError('Algorithm not recognized')

# 这也是test中保存结果的地方
# save_loc=f"../../result/{GRAPH_TYPE}_{NODES}spin_eco"
save_loc = f"../../result"

only_train = False
only_inference = True

if only_train:
    train.run(save_loc, graph_save_loc=GRAPHSAVELOC)

if only_inference:
    test.run(save_loc, network_save_path=NETWORKSAVEPATH,
                  graph_save_loc=GRAPHSAVELOC, batched=True, max_batch_size=None)

