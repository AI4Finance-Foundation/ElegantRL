import string
import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
from typing import List, Union, Tuple, Optional
from torch import Tensor
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util_evaluator import EncoderBase64
from rlsolver.methods.util_obj import obj_maxcut
from rlsolver.methods.util import (calc_result_file_name,
                                   calc_txt_files_with_prefixes
                                   )

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# if the file exists, the result file will be renamed so that there is conflict.
def write_result(obj: Union[float, int], running_duration: int,
                 alg_name: str,
                 solution: Union[Tensor, List[int], np.array], filename: str,
                 plus1=True, info_dict: dict = {}):
    write_graph_result(obj, running_duration,
                       None, alg_name,
                       solution, filename,
                       plus1, info_dict)


# if the file exists, the result file will be renamed so that there is conflict.
def write_graph_result(obj: Union[float, int], running_duration: int,
                       num_nodes: Optional[int], alg_name: str,
                       solution: Union[Tensor, List[int], np.array], filename: str,
                       plus1=True, info_dict: dict = {}):
    if type(solution[0]) == bool:
        sol = []
        for i in solution:
            assert i in [False, True]
            if i is False:
                sol.append(0)
            else:
                sol.append(1)
        solution = sol
    add_tail = '_' if running_duration is None else '_' + str(int(running_duration)) \
        if 'data' in filename else None
    new_filename = calc_result_file_name(filename, add_tail)

    # if new_filename exists, rename new_filename
    while os.path.exists(new_filename):
        assert ('.txt' in new_filename)
        parts = new_filename.split('.txt')
        assert (len(parts) == 2)
        lowercase_letters = string.ascii_lowercase
        random_int = np.random.randint(0, len(lowercase_letters))
        random_letter = lowercase_letters[random_int]
        new_filename = parts[0] + random_letter + '.txt'

    print("result filename: ", new_filename)
    with open(new_filename, 'w', encoding="UTF-8") as new_file:
        prefix = '// '
        new_file.write(f"{prefix}obj: {obj}\n")
        new_file.write(f"{prefix}running_duration: {running_duration}\n")
        if num_nodes is not None:
            new_file.write(f"// num_nodes: {num_nodes}\n")
        new_file.write(f"{prefix}alg_name: {alg_name}\n")
        for key, value in info_dict.items():
            new_file.write(f"{prefix}{key}: {value}\n")

        for i in range(len(solution)):
            if plus1:
                new_file.write(f"{i + 1} {solution[i] + 1}\n")
            else:
                new_file.write(f"{i + 1} {solution[i]}\n")


def write_result_set_cover(obj: Union[float, int], running_duration: int, num_items: int, num_sets: int, alg_name,
                           filename: str):
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
def read_graph_result_comments(filename: str):
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


# max_ID: exclusive
def read_graph_result_comments_multifiles2(dir: str, prefixes: List[str], max_ID: int):
    objs = {}
    running_durations = {}
    obj_bounds = {}
    # for prefix in prefixes:
    files = calc_txt_files_with_prefixes(dir, prefixes)
    for i in range(len(files)):
        file = files[i]
        num_nodes, ID, running_duration, obj, obj_bound = read_graph_result_comments(file)
        if ID >= max_ID:
            continue
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

    return objs, obj_bounds, running_durations, avg_objs, avg_obj_bounds, \
           avg_running_durations, std_objs, std_obj_bounds, std_running_durations


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


def calc_obj_maxcut_xstr(x_str: str, filename: str):
    graph = read_nxgraph(filename)
    num_nodes = graph.number_of_nodes()
    encoder = EncoderBase64(encode_len=num_nodes)
    x = encoder.str_to_bool(x_str).long()
    # print("x: ", x)
    obj = obj_maxcut(x, graph)
    return obj


if __name__ == '__main__':
    test_x = True
    if test_x:
        x_str = "4uBV2lGAiHqFxIenn"
        filename = "../data/syn_PL/PL_100_ID0.txt"
        obj = calc_obj_maxcut_xstr(x_str, filename)
        print("obj: ", obj)

    test_frist_10 = False
    if test_frist_10:
        dir = '../result'
        prefixes = ['BA_']
        max_ID = 10  # exclusive
        objs, obj_bounds, running_durations, avg_objs, avg_obj_bounds, \
        avg_running_durations, std_objs, std_obj_bounds, std_running_durations \
            = read_graph_result_comments_multifiles2(dir, prefixes, max_ID)

    test_frist_30 = False
    if test_frist_30:
        dir = '../result'
        prefixes = ['BA_']
        max_ID = 30  # exclusive
        objs, obj_bounds, running_durations, avg_objs, avg_obj_bounds, \
        avg_running_durations, std_objs, std_obj_bounds, std_running_durations \
            = read_graph_result_comments_multifiles2(dir, prefixes, max_ID)
