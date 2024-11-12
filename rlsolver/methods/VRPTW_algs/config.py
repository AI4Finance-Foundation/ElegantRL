import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import os
import numpy as np
from enum import Enum, unique

@unique
class Alg(Enum):
    impact_heuristic = 'impact_heuristic'
    column_generation = 'column_generation'

class Config:
    DATA_DIR = "../../data"
    RESULT_DIR = "../../result"
    INF = 1e6

    ADD_DEST_AS_SAME_ORIG = True  # add a dest node, which is the same as orig except id

    SORT_BY_CUMULATIVE_TRAVEL_COST = False

    # params in CG
    USE_PATHS_SET_IN_CG = True  # default by true. It will be a little faster if True.
    ADD_ONE_PATH_EACH_ITERATION_IN_CG = False
    MAX_NUM_ITERATIONS_IN_CG = INF
    USE_CHECK_WIDTH_IN_CG = False
    CHECK_WIDTH_IN_CG = 12
    CHECK_DIFF_THRESHOLD_IN_CG = 0
    USE_ESPPRC_IMPACT_AS_INIT_IN_CG = 0  # default: 0
    ADD_NUM_VISITED_NODES_FOR_DOMINATE_IN_CG = True   # the performance in True and False is almost the same
    assert USE_ESPPRC_IMPACT_AS_INIT_IN_CG in [0, 1]

    INSTANCE_NAME = "c101"
    NUM_PURE_CUSTOMERS = 10  # excluding orig or dest. it <= 100 if selecting instance 'c101'
    NUM_CUSTOMERS = NUM_PURE_CUSTOMERS + 2 if ADD_DEST_AS_SAME_ORIG else NUM_PURE_CUSTOMERS + 1

    ORIG_ID = 0
    # ORIG_NAME = str(ORIG_ID)
    # ORIG_NAME = "orig" + str(ORIG_ID)
    ORIG_NAME = str(ORIG_ID) + "-orig"
    DEST_ID = NUM_PURE_CUSTOMERS + 1 if ADD_DEST_AS_SAME_ORIG else NUM_PURE_CUSTOMERS
    # DEST_NAME = str(DEST_ID)
    # DEST_NAME = "dest" + str(DEST_ID)
    DEST_NAME = str(DEST_ID) + "-dest"
    CONNECT_ORIG_DEST = True  # default: True
    IDS_OF_PURE_CUSTOMERS = list(range(1, NUM_PURE_CUSTOMERS + 1))
    IDS_OF_CUSTOMERS = list(range(NUM_PURE_CUSTOMERS + 2)) if ADD_DEST_AS_SAME_ORIG else list(range(NUM_PURE_CUSTOMERS + 1))  # including orig and dest
    NAMES_OF_CUSTOMERS = [str(i) for i in IDS_OF_CUSTOMERS]
    NAMES_OF_CUSTOMERS[ORIG_ID] = ORIG_NAME
    NAMES_OF_CUSTOMERS[DEST_ID] = DEST_NAME
    VELOCITY = 1

    # input dir: solomon-instances
    INSTANCE_FILENAME = DATA_DIR + "/" + "solomon-instances" + "/" + INSTANCE_NAME + ".txt"
    RESULT_FILENAME = RESULT_DIR + "/" + INSTANCE_NAME + "-" + str(NUM_PURE_CUSTOMERS) + "-customers.txt"
    B1 = B2 = B3 = Bs = Be = Br = 1 / 3  # used in impact_heuristic
    assert B1 + B2 + B3 == 1 and B1 >= 0 and B2 >= 0 and B3 >= 0

    X = []
    Y = []
    TRAVEL_DURATION_MATRIX = []
    TRAVEL_DIST_MATRIX = []

    TIME_WINDOW_START = {}
    TIME_WINDOW_END = {}
    SERVICE_DURATION = {}
    TIME_WINDOW_START_OF_DEPOT = 0
    TIME_WINDOW_END_OF_DEPOT = 0
    SERVICE_DURATION_OF_DEPOT = 0
    NUM_VEHICLES = 0
    VEHICLE_CAPACITY = 6
    DEMANDS = []


    ALG = Alg.impact_heuristic


# if not os.path.exists(Config.ROUTES_FILENAME):
#     print("No other_routes generated for this instance and number of customers.")
#     print("Run: 'python main.py' and input desired instance and" +\
#             "number of customers.")
#     print("Exit")
#     exit(1)

def calc_Euclidean_distance(x, y, i, j):
    dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
    return dist

def calc_travel_dist(x, y, i, j):
    dist = calc_Euclidean_distance(x, y, i, j)
    res = (int) (round(dist))
    return res

def calc_travel_time(x, y, i, j):
    dist = calc_Euclidean_distance(x, y, i, j)
    travel_time = (int) (round(dist / Config.VELOCITY))
    return travel_time

def update_config(num_vehicles, vehicle_capacity, x, y, demands, time_window_start, time_window_end, service_time):
    Config.NUM_VEHICLES = num_vehicles
    Config.VEHICLE_CAPACITY = vehicle_capacity
    Config.X = x
    Config.Y = y
    Config.DEMANDS = demands
    Config.TIME_WINDOW_START = time_window_start
    Config.TIME_WINDOW_END = time_window_end
    Config.TIME_WINDOW_START_OF_DEPOT = time_window_start[0]
    Config.TIME_WINDOW_END_OF_DEPOT = time_window_end[0]
    Config.SERVICE_DURATION = service_time
    Config.SERVICE_DURATION_OF_DEPOT = service_time[0]

    Config.TRAVEL_DIST_MATRIX = []
    Config.TRAVEL_DURATION_MATRIX = []
    num_customers = Config.NUM_PURE_CUSTOMERS + 2 if Config.ADD_DEST_AS_SAME_ORIG else Config.NUM_PURE_CUSTOMERS + 1
    for i in range(num_customers):
        row = [0] * num_customers
        Config.TRAVEL_DIST_MATRIX.append(row)
        Config.TRAVEL_DURATION_MATRIX.append(row)
    for i in range(num_customers):
        for j in range(i + 1, num_customers):
            travel_dist = calc_travel_dist(x, y, i, j)
            travel_duration = calc_travel_time(x, y, i, j)
            Config.TRAVEL_DIST_MATRIX[i][j] = travel_dist
            Config.TRAVEL_DIST_MATRIX[j][i] = travel_dist
            Config.TRAVEL_DURATION_MATRIX[i][j] = travel_duration
            Config.TRAVEL_DURATION_MATRIX[j][i] = travel_duration
    Config.TRAVEL_DIST_MATRIX = np.array(Config.TRAVEL_DIST_MATRIX)
    Config.TRAVEL_DURATION_MATRIX = np.array(Config.TRAVEL_DURATION_MATRIX)
    print()


