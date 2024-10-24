import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import rlsolver.methods.eco_and_s2v_dqn.train_and_test.inference_eco as test
import rlsolver.methods.eco_and_s2v_dqn.train_and_test.train_eco as train

from rlsolver.methods.eco_and_s2v_dqn.config.eco_config import *

just_test = JUSTTEST
#这也是test中保存结果的地方
save_loc=f"../../result/{GRAPH_TYPE}_{NODES}spin_eco"


if not just_test:
    train.run(save_loc, graph_save_loc=GRAPHSAVELOC)
    test.run(save_loc, graph_save_loc=GRAPHSAVELOC, batched=True, max_batch_size=None,just_test=False)

else:
    test.run(save_loc, network_save_path = NETWORKSAVEPATH,
             graph_save_loc=GRAPHSAVELOC, batched=True, max_batch_size=None)

