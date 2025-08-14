import os
import sys
import time

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../')
sys.path.append(os.path.dirname(rlsolver_path))
from rlsolver.methods.eco_s2v.config import *

start_time = time.time()
if TRAIN_INFERENCE == 0:
    if ALG == Alg.eco:
        from rlsolver.methods.eco_s2v.train_and_inference.train_eco import run
    elif ALG == Alg.s2v:
        from rlsolver.methods.eco_s2v.train_and_inference.train_s2v import run
    elif ALG == Alg.eco_torch:
        from rlsolver.methods.eco_s2v.train_and_inference.train_eco_torch import run
    elif ALG == Alg.eeco:
        from rlsolver.methods.eco_s2v.train_and_inference.train_eeco import run
    elif ALG == Alg.jumanji:
        from rlsolver.methods.eco_s2v.jumanji.train_and_inference.train import run
    elif ALG == Alg.rl4co:
        from rlsolver.methods.eco_s2v.rl4co.train import run
    else:
        raise ValueError('Algorithm not recognized')
    run(save_loc=NEURAL_NETWORK_DIR)

if TRAIN_INFERENCE == 1:
    if ALG == Alg.eeco:
        from rlsolver.methods.eco_s2v.train_and_inference.inference_eeco import run

        run(graph_folder=DATA_DIR,
            n_sims=NUM_INFERENCE_SIMS,
            mini_sims=MINI_INFERENCE_SIMS)
    elif ALG == Alg.eco or ALG == Alg.s2v:
        from rlsolver.methods.eco_s2v.train_and_inference.inference import run

        run(save_loc=NEURAL_NETWORK_DIR, graph_save_loc=DATA_DIR, network_save_path=NEURAL_NETWORK_SAVE_PATH,
            batched=True, max_batch_size=None, max_parallel_jobs=1, prefixes=INFERENCE_PREFIXES)
    elif ALG == Alg.jumanji:
        from rlsolver.methods.eco_s2v.jumanji.train_and_inference.inference import run

        run(graph_folder=DATA_DIR, n_sims=NUM_INFERENCE_SIMS, mini_sims=MINI_INFERENCE_SIMS)
    elif ALG == Alg.rl4co:
        from rlsolver.methods.eco_s2v.rl4co.inference import run

        run(graph_dir=RL4CO_GRAPH_DIR, n_sims=NUM_INFERENCE_SIMS)
    else:
        raise ValueError('Algorithm not recognized')
running_duration = time.time() - start_time
print("running_duration: ", running_duration)
