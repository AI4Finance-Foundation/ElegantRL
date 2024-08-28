import sys
sys.path.append('../')
from pyscipopt import Model, quicksum
import os
import time
from typing import List
import networkx as nx
from util_read_data import read_nxgraph
from util import calc_txt_files_with_prefix
from util import calc_result_file_name
from util import calc_avg_std_of_objs
from util import plot_fig
from util import fetch_node
from util import transfer_float_to_binary


# the file has been open
def write_statistics(model, new_file, add_slash = False):
    prefix = '// ' if add_slash else ''
    obj = model.getObjVal()
    new_file.write(f"{prefix}obj: {obj}\n")
    new_file.write(f"{prefix}running_duration: {model.getTotalTime()}\n")
    gap = model.getGap()
    new_file.write(f"{prefix}gap: {gap}\n")
    # calc obj_bound
    if model.getObjectiveSense() == "maximize":
        obj_bound = obj * (1 + gap)
    else:
        obj_bound = obj * (1 - gap)
    new_file.write(f"{prefix}obj_bound: {obj_bound}\n")
    new_file.write(f"{prefix}time_limit: {model.getParam('limits/time')}\n")

# running_duration (seconds) is included.
def write_result_of_scip(model, filename: str = './result/result', running_duration: int = None):
    if filename.split('/')[0] == 'data':
        filename = calc_result_file_name(filename)
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    if running_duration is None:
        new_filename = filename
    else:
        new_filename = filename + '_' + str(int(running_duration))

    vars = model.getVars()
    nodes: List[int] = []
    values: List[int] = []
    for var in vars:
        node = fetch_node(var.name)
        if node is None:
            break
        value = transfer_float_to_binary(model.getVal(var))
        nodes.append(node)
        values.append(value)
    with open(f"{new_filename}.txt", 'w', encoding="UTF-8") as new_file:
        write_statistics(model, new_file, True)
        for i in range(len(nodes)):
            new_file.write(f"{nodes[i] + 1} {values[i] + 1}\n")
    with open(f"{new_filename}.sta", 'w', encoding="UTF-8") as new_file:
        write_statistics(model, new_file, False)
    with open(f"{new_filename}.sov", 'w', encoding="UTF-8") as new_file:
        new_file.write('values of vars: \n')
        for var in vars:
            new_file.write(f'{var.name}: {model.getVal(var)}\n')
    model.writeLP(f"{new_filename}.lp")
    model.writeStatistics(f"{new_filename}.sts")
    model.writeBestSol(f"{new_filename}.sol")
    # model.writeSol(f"{filename}.sol")
    print()

def run_using_scip(filename: str, time_limit: int = None, plot_fig_: bool = False):
    start_time = time.time()
    model = Model("maxcut")

    graph = read_nxgraph(filename)

    adjacency_matrix = nx.to_numpy_array(graph)
    num_nodes = nx.number_of_nodes(graph)
    nodes = list(range(num_nodes))

    x = {}
    y = {}
    for i in range(num_nodes):
        x[i] = model.addVar(vtype='B', name=f"x[{i}]")
    for i in range(num_nodes):
        for j in range(num_nodes):
            y[(i, j)] = model.addVar(vtype='B', name=f"y[{i}][{j}]")
    model.setObjective(quicksum(quicksum(adjacency_matrix[(i, j)] * y[(i, j)] for i in range(0, j)) for j in nodes),
                    'maximize')

    # constrs
    for j in nodes:
        for i in range(0, j):
            model.addCons(y[(i, j)] - x[i] - x[j] <= 0, name='C0a_' + str(i) + '_' + str(j))
            model.addCons(y[(i, j)] + x[i] + x[j] <= 2, name='C0b_' + str(i) + '_' + str(j))
    if time_limit is not None:
        model.setRealParam("limits/time", time_limit)
    model.optimize()


    # if model.getStatus() == "optimal":
    running_duration = time.time() - start_time
    write_result_of_scip(model, filename, time_limit)


    print('obj:', model.getObjVal())


    scores = [model.getObjVal()]
    alg_name = 'SCIP'
    if plot_fig_:
        plot_fig(scores, alg_name)
    print()

def run_scip_over_multiple_files(prefixes: List[str], time_limits: List[int], directory_data: str = 'data', directory_result: str = './result'):
    for prefix in prefixes:
        files = calc_txt_files_with_prefix(directory_data, prefix)
        for i in range(len(files)):
            print(f'The {i}-th file: {files[i]}')
            for j in range(len(time_limits)):
                run_using_scip(files[i], time_limits[j])
    directory = '../result'
    calc_avg_std_of_objs(directory, prefixes, time_limits)
if __name__ == '__main__':
    select_single_file = True
    if select_single_file:
        filename = '../data/syn/syn_50_176.txt'
        time_limits = [0.5 * 3600]
        run_using_scip(filename, time_limit=time_limits[0], plot_fig_=True)
        directory = '../result'
        prefixes = ['syn_50_']
        avg_std = calc_avg_std_of_objs(directory, prefixes, time_limits)

    else:
        directory_data = '../data/syn'
        prefixes = ['syn_10_', 'syn_50_', 'syn_100_', 'syn_300_', 'syn_500_', 'syn_700_', 'syn_900_', 'syn_1000_', 'syn_3000_', 'syn_5000_', 'syn_7000_', 'syn_9000_', 'syn_10000_']
        # prefixes = ['syn_10_']
        # time_limits = [0.5 * 3600, 1 * 3600]
        time_limits = [0.5 * 3600]
        directory_result = '../result'
        run_scip_over_multiple_files(prefixes, time_limits, directory_data, directory_result)
        avg_std = calc_avg_std_of_objs(directory_result, prefixes, time_limits)
    pass

