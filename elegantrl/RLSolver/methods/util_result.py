import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import copy
from torch.autograd import Variable
import os
import functools
import time
import torch.nn as nn
import numpy as np
from typing import List, Union, Tuple, Optional
import networkx as nx
import pandas as pd
import torch as th
from torch import Tensor
from os import system
import math
from enum import Enum
import tqdm
import re
from util import (calc_result_file_name,
calc_txt_files_with_prefix
                  )
from config import (NUM_IDS,
                    RUNNING_DURATIONS,
                    )
# from methods.simulated_annealing import simulated_annealing_set_cover, simulated_annealing
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from util import (plot_fig_over_durations,
                  )


# write a tensor/list/np.array (dim: 1) to a txt file.
# The nodes start from 0, and the label of classified set is 0 or 1 in our codes, but the nodes written to file start from 1, and the label is 1 or 2
def write_result(result: Union[Tensor, List, np.array],
                 filename: str = './result/result.txt',
                 obj: Union[int, float] = None,
                 running_duration: Union[int, float] = None):
    # assert len(result.shape) == 1
    # N = result.shape[0]
    num_nodes = len(result)
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(filename, 'w', encoding="UTF-8") as file:
        if obj is not None:
            file.write(f'// obj: {obj}\n')
        if running_duration is not None:
            file.write(f'// running_duration: {running_duration}\n')
        for node in range(num_nodes):
            file.write(f'{node + 1} {int(result[node] + 1)}\n')

def write_result3(obj, running_duration, num_nodes, alg_name, solution, filename: str):
    add_tail = '_' if running_duration is None else '_' + str(int(running_duration)) if 'data' in filename else None
    new_filename = calc_result_file_name(filename, add_tail)
    print("result filename: ", new_filename)
    with open(new_filename, 'w', encoding="UTF-8") as new_file:
        prefix = '// '
        new_file.write(f"{prefix}obj: {obj}\n")
        new_file.write(f"{prefix}running_duration: {running_duration}\n")
        new_file.write(f"// num_nodes: {num_nodes}\n")
        new_file.write(f"{prefix}alg_name: {alg_name}\n")
        for i in range(len(solution)):
            new_file.write(f"{i + 1} {solution[i] + 1}\n")

def write_result_set_cover(obj, running_duration, num_items: int, num_sets: int, alg_name, filename: str):
    add_tail = '_' + str(int(running_duration)) if 'data' in filename else None
    new_filename = calc_result_file_name(filename, add_tail)
    with open(new_filename, 'w', encoding="UTF-8") as new_file:
        prefix = '// '
        new_file.write(f"{prefix}obj: {obj}\n")
        new_file.write(f"{prefix}running_duration: {running_duration}\n")
        new_file.write(f"// num_sets: {num_sets}\n")
        new_file.write(f"// num_items: {num_items}\n")
        new_file.write(f"{prefix}alg_name: {alg_name}\n")

# return: num_nodes, ID, running_duration:, obj,
def read_result_comments(filename: str):
    num_nodes, ID, running_duration, obj = None, None, None, None
    ID = int(filename.split('ID')[1].split('_')[0])
    with open(filename, 'r') as file:
        # lines = []
        line = file.readline()
        while line is not None and line != '':
            if '//' in line:
                if 'num_nodes:' in line:
                    num_nodes = int(line.split('num_nodes:')[1])
                    break
                if 'running_duration:' in line:
                    running_duration = obtain_first_number(line)
                if 'obj:' in line:
                    obj = float(line.split('obj:')[1])
                if 'obj_bound:' in line:
                    obj_bound = float(line.split('obj_bound:')[1])

            line = file.readline()
    return num_nodes, ID, running_duration, obj, obj_bound

def read_result_comments_multifiles2(dir: str, prefixes: str, max_ID: int):
    objs = {}
    running_durations = {}
    obj_bounds = {}
    # for prefix in prefixes:
    files = calc_txt_files_with_prefix(dir, prefixes)
    for i in range(len(files)):
        file = files[i]
        num_nodes, ID, running_duration, obj, obj_bound = read_result_comments(file)
        if ID >= max_ID + 1:
            continue
        if num_nodes == 200:
            print("ID: ", ID)
            aaa = 1
        if str(num_nodes) not in objs.keys():
            objs[str(num_nodes)] = [obj]
            running_durations[str(num_nodes)] = [running_duration]
        else:
            objs[str(num_nodes)].append(obj)
            running_durations[str(num_nodes)].append(running_duration)
        if str(num_nodes) not in obj_bounds.keys():
            obj_bounds[str(num_nodes)] = [obj_bound]
        else:
            obj_bounds[str(num_nodes)].append(obj_bound)

    label = f"num_nodes={num_nodes}"
    print(f"objs: {objs}, running_durations: {running_durations}")
    # objs = [(key, objs[key]) for key in sorted(objs.keys())]
    objs = dict(sorted(objs.items(), key=lambda x: x[0]))
    obj_bounds = dict(sorted(obj_bounds.items(), key=lambda x: x[0]))
    running_durations = dict(sorted(running_durations.items(), key=lambda x: x[0]))

    avg_objs = {}
    avg_obj_bounds = {}
    avg_running_durations = {}
    std_objs = {}
    std_obj_bounds = {}
    std_running_durations = {}
    for key, value in objs.items():
        avg_objs[key] = np.average(value)
        std_objs[key] = np.std(value)
    for key, value in obj_bounds.items():
        avg_obj_bounds[key] = np.average(value)
        std_obj_bounds[key] = np.std(value)
    for key, value in running_durations.items():
        avg_running_durations[key] = np.average(value)
        std_running_durations[key] = np.std(value)

    return objs, obj_bounds, running_durations, avg_objs, avg_obj_bounds, avg_running_durations, std_objs, std_obj_bounds, std_running_durations



# def calc_txt_files_with_prefix(directory: str, prefix: str):
#     res = []
#     files = os.listdir(directory)
#     for file in files:
#         if prefix in file and ('.txt' in file or '.msc' in file):
#             res.append(directory + '/' + file)
#     return res


# e.g., s = "// time_limit: ('TIME_LIMIT', <class 'float'>, 36.0, 0.0, inf, inf)",
# then returns 36
def obtain_first_number(s: str):
    res = ''
    pass_first_digit = False
    for i in range(len(s)):
        if s[i].isdigit() or s[i] == '.':
            res += s[i]
            pass_first_digit = True
        elif pass_first_digit:
            break
    value = int(float(res))
    return value



def read_result_comments_multifiles(dir: str, prefixes: str, running_durations: List[int]):
    res = {}
    num_nodess = set()
    # for prefix in prefixes:
    files = calc_txt_files_with_prefix(dir, prefixes)
    num_ids = NUM_IDS
    for i in range(len(files)):
        file = files[i]
        num_nodes, ID, running_duration, obj, obj_bound = read_result_comments(file)
        if running_duration not in running_durations:
            continue
        index = running_durations.index(running_duration)
        num_nodess.add(num_nodes)
        if str(num_nodes) not in res.keys():
            res[str(num_nodes)] = [[None] * len(running_durations) for _ in range(num_ids)]
        res[str(num_nodes)][ID][index] = obj
            # res[str(num_nodes)] = {**res[str(num_nodes)], **tmp_dict}
    for num_nodes_str in res.keys():
        for ID in range(num_ids):
            last_nonNone = None
            for i in range(len(running_durations)):
                if res[num_nodes_str][ID][i] is not None:
                    last_nonNone = res[num_nodes_str][ID][i]
                if res[num_nodes_str][ID][i] is None and last_nonNone is not None:
                    res[num_nodes_str][ID][i] = last_nonNone

    num_nodess = list(num_nodess)
    num_nodess.sort()
    for num_nodes in num_nodess:
        objs = []
        for i in range(len(running_durations)):
            sum_obj = 0
            for ID in range(num_ids):
                if res[str(num_nodes)][ID][i] is not None:
                    sum_obj += res[str(num_nodes)][ID][i]
            obj = sum_obj / num_ids
            objs.append(obj)
        label = f"num_nodes={num_nodes}"
        print(f"objs: {objs}, running_duration: {running_durations}, label: {label}")
        plot_fig_over_durations(objs, running_durations, label)

if __name__ == '__main__':
    # result = Tensor([0, 1, 0, 1, 0, 1, 1])
    # write_result(result)
    # result = [0, 1, 0, 1, 0, 1, 1]
    # write_result(result)
    write_result_ = False
    if write_result_:
        result = [1, 0, 1, 0, 1]
        write_result(result)


    # dir = 'syn_BA_greedy2approx'
    dir = '../result/syn_BA_gurobi'
    # prefixes = 'barabasi_albert_200'
    prefixes = 'barabasi_albert_'
    max_ID = 9
    objs, obj_bounds, running_durations, avg_objs, avg_obj_bounds, avg_running_durations, std_objs, std_obj_bounds, std_running_durations = read_result_comments_multifiles2(dir, prefixes, max_ID)

    if_plot = True
    if if_plot:
        dir = '../result/syn_PL_gurobi'
        prefixes = 'powerlaw_200_'
        running_durations = RUNNING_DURATIONS
        read_result_comments_multifiles(dir, prefixes, running_durations)

    print(avg_objs)
    print(obj_bounds)