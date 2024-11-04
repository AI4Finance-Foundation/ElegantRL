import torch as th
from typing import List
from enum import Enum, unique
# from L2A.graph_utils import GraphList, obtain_num_nodes
import os

@unique
class Problem(Enum):
    maxcut = "maxcut"
    graph_partitioning = "graph_partitioning"
    minimum_vertex_cover = "minimum_vertex_cover"
    number_partitioning = "number_partitioning"
    bilp = "bilp"
    maximum_independent_set = "maximum_independent_set"
    knapsack = "knapsack"
    set_cover = "set_cover"
    graph_coloring = "graph_coloring"
    tsp = "tsp"
PROBLEM = Problem.maxcut

@unique
class GraphType(Enum):
    BA: str = "BA"  # "barabasi_albert"
    ER: str = "ER"  # "erdos_renyi"
    PL: str = "PL"  # "powerlaw"

def calc_device(gpu_id: int):
    return th.device(f"cuda:{gpu_id}" if th.cuda.is_available() and gpu_id >= 0 else "cpu")

GPU_ID: int = 0  # -1: cpu, >=0: gpu

DATA_FILENAME = "../data/syn_BA/BA_100_ID0.txt"  # one instance
DIRECTORY_DATA = "../data/syn_BA"  # used in multi instances
PREFIXES = ["BA_100_ID0"]  # used in multi instances

DEVICE: th.device = calc_device(GPU_ID)

GRAPH_TYPE = GraphType.PL
GRAPH_TYPES: List[GraphType] = [GraphType.ER, GraphType.PL, GraphType.BA]
    # graph_types = ["erdos_renyi", "powerlaw", "barabasi_albert"]
NUM_IDS = 30  # ID0, ..., ID29



INF = 1e6

# RUNNING_DURATIONS = [600, 1200, 1800, 2400, 3000, 3600]  # store results
RUNNING_DURATIONS = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]  # store results

# None: write results when finished.
# others: write results in mycallback. seconds, the interval of writing results to txt files
GUROBI_INTERVAL = None  # None: not using interval, i.e., not using mycallback function, write results when finished. If not None such as 100, write result every 100s
GUROBI_TIME_LIMITS = [1 * 3600]  # seconds
# GUROBI_TIME_LIMITS = [600, 1200, 1800, 2400, 3000, 3600]  # seconds
# GUROBI_TIME_LIMITS2 = list(range(10 * 60, 1 * 3600 + 1, 10 * 60))  # seconds
GUROBI_VAR_CONTINUOUS = False  # True: relax it to LP, and return x values. False: sovle the primal MILP problem
GUROBI_MILP_QUBO = 1  # 0: MILP, 1: QUBO. default: QUBO, since using QUBO is generally better than MILP.
assert GUROBI_MILP_QUBO in [0, 1]



