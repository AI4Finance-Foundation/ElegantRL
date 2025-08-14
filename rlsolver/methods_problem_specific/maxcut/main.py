import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
from rlsolver.methods_problem_specific.maxcut_BLS.BLS import BLSMaxCut
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util_obj import obj_maxcut
from rlsolver.methods.util_result import write_graph_result

def run_bls_trial(args):
    trial_id, filename = args
    filename = os.path.abspath(filename)

    graph = read_nxgraph(filename)
    num_nodes = graph.number_of_nodes()
    for u, v, d in graph.edges(data=True):
        if "weight" in d:
            d["weight"] = int(d["weight"])

    params = {
        "L0_ratio":      0.01,
        "T":              1000,
        "phi_min":         3,
        "phi_max_ratio":  0.1,
        "P0":             0.8,
        "Q":              0.5,
        "max_iters": 10_000 * num_nodes,
    }

    start_time = time.time()
    solver = BLSMaxCut(graph, params)
    solution, best_val = solver.run(target=None, time_limit=120)
    running_duration = time.time() - start_time

    return trial_id, best_val, solution, running_duration

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    filename = str(project_root / "data" / "syn_BA" / "BA_100_ID0.txt").replace('\\', '/')

    print("Experimental file path:", filename)
    if not os.path.exists(filename):
        print("❌ The file does not exist, please check the path and file.！")
        sys.exit(1)

    num_trials = 20
    workers = min(4, num_trials)

    trial_args = [(trial_id, filename) for trial_id in range(1, num_trials + 1)]
    print(f"Running {num_trials} independent BLS trials in parallel...")
    with ProcessPoolExecutor(max_workers=workers) as exe:
        results = list(exe.map(run_bls_trial, trial_args))

    vals = []
    sols = []
    durations = []
    for trial_id, best_val, solution, running_duration in results:
        print(f"Trial {trial_id:2d}: best_val={best_val}, time={running_duration:.2f}s")
        vals.append(best_val)
        sols.append(solution)
        durations.append(running_duration)

    best_val = max(vals)
    best_idx = vals.index(best_val)
    best_solution = sols[best_idx]
    avg = sum(vals) / len(vals)
    std = (sum((x - avg) ** 2 for x in vals) / len(vals)) ** 0.5
    avg_time = sum(durations) / len(durations)


    print("\n=== Summary ===")
    print(" Best =", best_val)
    print(" Avg  =", f"{avg:.2f}")
    print(" Std  =", f"{std:.2f}")
    print(" Avg time =", f"{avg_time:.2f}s")


    graph = read_nxgraph(filename)
    num_nodes = graph.number_of_nodes()
    for u, v, d in graph.edges(data=True):
        if "weight" in d:
            d["weight"] = int(d["weight"])
    alg_name = "BLS"
    write_graph_result(
        best_val,
        avg_time,
        num_nodes,
        alg_name,
        [bool(x) for x in best_solution],

        filename
    )

    init_solution = [0] * num_nodes
    init_score = obj_maxcut(init_solution, graph)
    print("init_score, final score of bls:", init_score, best_val)
    print("solution:", best_solution)
    print("average running_duration:", avg_time)

