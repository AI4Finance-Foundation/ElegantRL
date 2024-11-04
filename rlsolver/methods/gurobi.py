import sys
import os
import numpy as np

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

from gurobipy import *
import copy
import networkx as nx
import sys
import matplotlib.pyplot as plt
import time
from rlsolver.methods.util_read_data import (read_nxgraph,
                                             read_knapsack_data,
                                             read_set_cover_data)
from rlsolver.methods.util import (transfer_float_to_binary,
                                   calc_txt_files_with_prefixes,
                                   calc_result_file_name,
                                   calc_avg_std_of_objs,
                                   plot_fig,
                                   fetch_node,
                                   transfer_nxgraph_to_adjacencymatrix)
# from util import fetch_indices
from rlsolver.methods.config import *
from rlsolver.methods.tsp_algs.util import read_tsp_file
from itertools import combinations


# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        num_nodes = model._attribute['num_nodes']
        # varlist = [v for v in model.getVars() if 'x' in v.VarName]
        # vals = model.cbGetSolution(varlist)
        # vals = np.array(vals)
        # vals = vals.reshape(num_nodes, num_nodes)
        vals = model.cbGetSolution(model._vars)
        # find the shortest cycle in the selected edge list
        tour = subtour(model, vals)
        if len(tour) < num_nodes:
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(quicksum(model._vars[i, j]
                                     for i, j in combinations(tour, 2))
                         <= len(tour) - 1)


# Given a tuplelist of edges, find the shortest subtour

def subtour(model, vals):
    n = model._attribute['num_nodes']
    # make a list of edges selected in the solution
    # edges = tuplelist((i, j) for i, j in vals.keys()
    #                      if vals[i, j] > 0.5)
    print("vals shape: ", np.shape(vals))
    edges = tuplelist()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if type(vals) == list:
                val = vals[i][j]
            else:
                val = vals[(i, j)]
            if val > 0.5:
                edges.append((i, j))
    print("edges: ", edges)
    unvisited = list(range(n))
    cycle = range(n + 1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle


# 定义回调函数，每隔一段时间将当前找到的最佳可行解输出到当前目录下以 solution 开头
# 的文件中。同时，将当前进展输出到 report.txt 报告中。
def mycallback_write(model, where):
    if where == GRB.Callback.MIPSOL:
        # MIP solution callback
        currentTime = time.time()
        running_duation = int((currentTime - model._startTime) / model._interval) * model._interval

        # Statistics
        objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        # gap = abs(obj - objbnd) / (obj + 1e-6)

        # Export solution to a file
        # solutionfile = open("solution_" + str(running_duation) + ".txt", 'w')

        # filename = copy.deepcopy(model._reportFile.name) # ok, successful
        filename = copy.deepcopy(model._attribute['result_filename'])
        filename = filename.replace('.txt', '')
        filename = filename + '_' + str(running_duation) + '.txt'

        # vars = model.getVars()
        # nodes: List[int] = []
        # values: List[int] = []
        # for var in vars:
        #     node = fetch_node(var.VarName)
        #     if node is None:
        #         break
        #     value = transfer_float_to_binary(var.x)
        #     nodes.append(node)
        #     values.append(value)

        if GUROBI_INTERVAL is not None and running_duation < GUROBI_INTERVAL:
            return
        with open(filename, 'w', encoding="UTF-8") as new_file:
            write_statistics_in_mycallback(model, new_file, add_slash=True)
            new_file.write(f"// num_nodes: {model._attribute['num_nodes']}\n")
            # for i in range(len(nodes)):
            #     new_file.write(f"{nodes[i] + 1} {values[i] + 1}\n")
            varlist = [v for v in model.getVars() if 'x' in v.VarName]
            soln = model.cbGetSolution(varlist)

            # for var, soln in zip(varlist, soln):
            #     solutionfile.write('%s %d\n' % (var.VarName, soln))
            for i in range(len(varlist)):
                value = int(round(soln[i]) + 1) if not GUROBI_VAR_CONTINUOUS else soln[i]
                new_file.write(f"{i + 1} {value}\n")


# the file has been open
def write_statistics(model, new_file, add_slash=False):
    prefix = '// ' if add_slash else ''
    if PROBLEM == Problem.maximum_independent_set:
        from util import obj_maximum_independent_set
        solution = model._attribute['solution']
        graph = model._attribute['graph']
        obj = obj_maximum_independent_set(solution, graph)
        new_file.write(f"{prefix}obj: {obj}\n")
    else:
        new_file.write(f"{prefix}obj: {model.objVal}\n")
    new_file.write(f"{prefix}running_duration: {model.Runtime}\n")
    if not GUROBI_VAR_CONTINUOUS:
        new_file.write(f"{prefix}gap: {model.MIPGap}\n")
    new_file.write(f"{prefix}obj_bound: {model.ObjBound}\n")
    # new_file.write(f"time_limit: {time_limit}\n")
    time_limit = model.getParamInfo("TIME_LIMIT")
    new_file.write(f"{prefix}time_limit: {time_limit}\n")


def write_statistics_in_mycallback(model, new_file, add_slash=False):
    if model.getAttr('SolCount') == 0:
        return

    currentTime = time.time()
    running_duation = int((currentTime - model._startTime) / model._interval) * model._interval

    # Statistics
    objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
    gap = abs(obj - objbnd) / (obj + 1e-6)

    # varlist = model.getVars()
    # soln = model.cbGetSolution(varlist)
    # new_file.write('Objective %e\n' % obj)
    # for var, soln in zip(varlist, soln):
    #     new_file.write('%s %.16e\n' % (var.VarName, soln))
    # new_file.close()

    # # Export statistics
    # msg = str(currentTime - model._startTime) + " : " + "Solution Obj: " + str(obj) + " Solution Gap: " + str(
    #     gap) + "\n"
    # model._reportFile.write(msg)
    # model._reportFile.flush()

    prefix = '// ' if add_slash else ''
    new_file.write(f"{prefix}obj: {obj}\n")
    new_file.write(f"{prefix}running_duration: {running_duation}\n")
    new_file.write(f"{prefix}gap: {gap}\n")
    new_file.write(f"{prefix}obj_bound: {objbnd}\n")
    # new_file.write(f"time_limit: {time_limit}\n")
    time_limit = model.getParamInfo("TIME_LIMIT")
    # time_limit2 = model.params['TimeLimit']
    new_file.write(f"{prefix}time_limit: {time_limit}\n")


# if filename = '../result/BA_100_ID0.txt', running_duration = 100,
def write_result_gurobi(model, filename: str = './result/result', running_duration: int = None):
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    add_tail = '_' + str(int(running_duration)) if running_duration is not None else None
    new_filename = calc_result_file_name(filename, add_tail)

    vars = model.getVars()
    nodes: List[int] = []
    values: List[int] = []
    tuples_and_values = {}
    if PROBLEM == Problem.tsp:
        tour = model._attribute['tour']
        with open(new_filename, 'w', encoding="UTF-8") as new_file:
            write_statistics(model, new_file, True)
            if 'num_nodes' in model._attribute:
                new_file.write(f"// num_nodes: {model._attribute['num_nodes']}\n")
            for i in range(len(tour)):
                new_file.write(f"{i + 1} {tour[i] + 1}\n")
            new_file.write(f"{len(tour) + 1} {tour[0] + 1}\n")
        return

    for var in vars:
        if "x" not in var.VarName:
            continue
        try:
            node = fetch_node(var.VarName)
            if GUROBI_VAR_CONTINUOUS:
                value = var.x
            else:
                value = transfer_float_to_binary(var.x)
            nodes.append(node)
            values.append(value)
        except ValueError:
            pass


    with open(new_filename, 'w', encoding="UTF-8") as new_file:
        model._attribute['solution'] = values
        write_statistics(model, new_file, True)
        if 'num_nodes' in model._attribute:
            new_file.write(f"// num_nodes: {model._attribute['num_nodes']}\n")
        for i in range(len(nodes)):
            if GUROBI_VAR_CONTINUOUS or PROBLEM == Problem.minimum_vertex_cover:
                new_file.write(f"{nodes[i] + 1} {values[i]}\n")
            else:
                new_file.write(f"{nodes[i] + 1} {values[i] + 1}\n")

        new_file.write("// Tuples Format: \n")
        for (i, j), value in tuples_and_values.items():
            new_file.write(f"{i + 1}, {j + 1}: {value}\n")

    if_write_others = False
    if if_write_others:
        with open(f"{new_filename}.sta", 'w', encoding="UTF-8") as new_file:
            write_statistics(model, new_file, False)
        with open(f"{new_filename}.sov", 'w', encoding="UTF-8") as new_file:
            new_file.write('values of vars: \n')
            vars = model.getVars()
            for var in vars:
                new_file.write(f'{var.VarName}: {var.solution}\n')
        model.write(f"{new_filename}.mst")
        model.write(f"{new_filename}.lp")
        model.write(f"{new_filename}.mps")
        model.write(f"{new_filename}.sol")


def run_using_gurobi(filename: str, init_x=None, time_limit: int = None, plot_fig_: bool = False):
    model = Model("maxcut")

    if PROBLEM == Problem.tsp:
        assert GUROBI_MILP_QUBO == 0  # QUBO is not tested
        graph, points = read_tsp_file(filename)
        n = graph.number_of_nodes()
        dist = {(i, j):
                    math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
                for i in range(n) for j in range(i)}
    elif PROBLEM == Problem.knapsack:
        num, weight, items = read_knapsack_data(filename)
    elif PROBLEM == Problem.set_cover:
        total_elements, total_subsets, subsets = read_set_cover_data(filename)
    else:
        graph = read_nxgraph(filename)

    if PROBLEM not in [Problem.knapsack, Problem.set_cover]:
        edges = list(graph.edges)
        subax1 = plt.subplot(111)
        nx.draw_networkx(graph, with_labels=True)
        if plot_fig_:
            plt.show()
        adjacency_matrix = transfer_nxgraph_to_adjacencymatrix(graph)
        num_nodes = nx.number_of_nodes(graph)
        nodes = list(range(num_nodes))

    if PROBLEM == Problem.maxcut:
        y_lb = adjacency_matrix.min()
        y_ub = adjacency_matrix.max()
        x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
        mode = None
        if init_x is None:
            pass
        elif init_x[0] is True or init_x[0] is False:
            mode = 0
        elif max(init_x) == 1:
            mode = 1
        elif max(init_x) == 2:
            mode = 2
        else:
            raise ValueError("wrong mode")
        if init_x is not None:
            for i in range(len(init_x)):
                init = init_x[i]
                if mode == 0:
                    x[i].start = 1 if init is True else 0
                elif mode == 1:
                    x[i].start = 1 if init == 1 else 0
                else:
                    x[i].start = 1 if init == 2 else 0

        if GUROBI_MILP_QUBO == 0:
            y = model.addVars(num_nodes, num_nodes, vtype=GRB.CONTINUOUS, lb=y_lb, ub=y_ub, name="y")
            model.setObjective(
                quicksum(quicksum(adjacency_matrix[(i, j)] * y[(i, j)] for i in range(0, j)) for j in nodes),
                GRB.MAXIMIZE)
        else:
            model.setObjective(
                quicksum(
                    quicksum(adjacency_matrix[(i, j)] * (0.5 - 2 * (x[i] - 0.5) * (x[j] - 0.5)) for i in range(0, j))
                    for j in nodes),
                GRB.MAXIMIZE)
    elif PROBLEM == Problem.graph_partitioning:
        if GUROBI_MILP_QUBO == 0:
            y_lb = adjacency_matrix.min()
            y_ub = adjacency_matrix.max()
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            y = model.addVars(num_nodes, num_nodes, vtype=GRB.CONTINUOUS, lb=y_lb, ub=y_ub, name="y")
            model.setObjective(
                quicksum(quicksum(adjacency_matrix[(i, j)] * y[(i, j)] for i in range(0, j)) for j in nodes),
                GRB.MINIMIZE)
        else:
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            coef_A = len(edges) + 10
            model.setObjective(
                coef_A * quicksum(2 * (x[k] - 0.5) for k in nodes) * quicksum(2 * (x[k] - 0.5) for k in nodes)
                + quicksum(
                    quicksum(adjacency_matrix[(i, j)] * (0.5 - 2 * (x[i] - 0.5) * (x[j] - 0.5)) for i in range(0, j))
                    for j in nodes),
                GRB.MINIMIZE)
    elif PROBLEM == Problem.minimum_vertex_cover:
        if GUROBI_MILP_QUBO == 0:
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            model.setObjective(quicksum(x[j] for j in nodes),
                               GRB.MINIMIZE)
        else:
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            coef_A = len(nodes) + 10
            model.setObjective(coef_A * quicksum(
                quicksum(adjacency_matrix[(i, j)] * (1 - x[i]) * (1 - x[j]) for i in range(0, j)) for j in nodes)
                               + quicksum(x[j] for j in nodes),
                               GRB.MINIMIZE)
    elif PROBLEM == Problem.maximum_independent_set:
        if GUROBI_MILP_QUBO == 0:
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            model.setObjective(quicksum(x[j] for j in nodes),
                               GRB.MAXIMIZE)
        else:
            coef_B1 = -1
            coef_B2 = 3
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            model.setObjective(-quicksum(x[j] for j in nodes) + coef_B1 * quicksum(
                (2 - x[i] - x[j]) * (2 - x[i] - x[j]) for (i, j) in edges)
                               + coef_B2 * quicksum((1 - x[i] - x[j]) * (1 - x[i] - x[j]) for (i, j) in edges),
                               GRB.MINIMIZE)
    elif PROBLEM == Problem.tsp:
        if GUROBI_MILP_QUBO == 0:
            use_gurobiexample = True
            if use_gurobiexample:
                x = model.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')
                for i, j in x.keys():
                    x[j, i] = x[i, j]
                model.setObjective(quicksum(adjacency_matrix[i][j] * x[i, j]
                                            for i in range(num_nodes)
                                            for j in range(num_nodes)
                                            if i != j), GRB.MINIMIZE)
            else:
                x = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name='x')
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            x[i, j] = x[j, i]
                model.addConstrs(x.sum(i, '*') == 2 for i in range(n))
            # u = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, lb=1, ub=1000000, name='u_tsp')

            model._vars = x
            model.Params.LazyConstraints = 1
        else:
            coef_A = 1.0
            x = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name='x')
            HA = quicksum((1 - quicksum(x[i, j] for j in range(num_nodes))) ** 2 for i in range(num_nodes)) + \
                 quicksum((1 - quicksum(x[i, j] for i in range(num_nodes))) ** 2 for j in range(num_nodes))
            # edges = list(graph.edges)
            HB = quicksum(
                adjacency_matrix[u][v] * x[u, j] * x[v, (j + 1) % num_nodes] for u in range(num_nodes) for v in
                range(num_nodes) if u != v for j
                in range(num_nodes))

            model.setObjective(HA + coef_A * HB, GRB.MINIMIZE)
    elif PROBLEM == Problem.knapsack:
        if GUROBI_MILP_QUBO == 0:
            x = model.addVars(num, vtype=GRB.BINARY, name="x")
            model.setObjective(quicksum(items[i][1] * x[i] for i in range(num)), GRB.MAXIMIZE)
        else:
            x = model.addVars(num, vtype=GRB.BINARY, name='x')
            y = model.addVars(weight + 1, vtype=GRB.BINARY, name='y')
            alpha = min(1.0 / max(value for weight, value in items), 1.0) / 2
            model.setObjective(
                (quicksum(y[n] for n in range(1, weight + 1))) ** 2 +
                (quicksum(n * y[n] for n in range(1, weight + 1)) - quicksum(
                    items[i][0] * x[i] for i in range(num))) ** 2 -
                alpha * quicksum(items[i][1] * x[i] for i in range(num)),
                GRB.MINIMIZE
            )

    elif PROBLEM == Problem.set_cover:
        if GUROBI_MILP_QUBO == 0:
            x = model.addVars(total_subsets, vtype=GRB.BINARY, name="x")
            model.setObjective(quicksum(x[i] for i in range(total_subsets)), GRB.MINIMIZE)
        else:
            x = model.addVars(total_subsets, vtype=GRB.BINARY, name="x")
            y = {}
            for u in range(total_elements):
                for m in range(total_subsets):
                    y[u, m] = model.addVar(vtype=GRB.BINARY, name=f"y_{u}_{m}")
            coef_A = 1.0
            H1 = quicksum(x[i] for i in range(total_subsets))
            H2 = quicksum((1 - quicksum(y[u, m] for m in range(total_subsets))) ** 2 for u in range(total_elements))
            H3 = quicksum((quicksum(m * y[u, m] for m in range(total_subsets)) - quicksum(
                x[i] for i in range(total_subsets) if u in subsets[i])) ** 2 for u in range(total_elements))
            model.setObjective(H1 + coef_A * (H2 + H3), GRB.MINIMIZE)

    # constrs if using MILP
    if GUROBI_MILP_QUBO == 0:
        if PROBLEM == Problem.maxcut:
            # y_{i, j} = x_i XOR x_j
            for j in nodes:
                for i in range(0, j):
                    model.addConstr(y[(i, j)] <= x[i] + x[j], name='C0b_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] <= 2 - x[i] - x[j], name='C0a_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] >= x[i] - x[j], name='C0c_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] >= -x[i] + x[j], name='C0d_' + str(i) + '_' + str(j))
        elif PROBLEM == Problem.graph_partitioning:
            # y_{i, j} = x_i XOR x_j
            for j in nodes:
                for i in range(0, j):
                    model.addConstr(y[(i, j)] <= x[i] + x[j], name='C0b_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] <= 2 - x[i] - x[j], name='C0a_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] >= x[i] - x[j], name='C0c_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] >= -x[i] + x[j], name='C0d_' + str(i) + '_' + str(j))
            model.addConstr(quicksum(x[j] for j in nodes) == num_nodes / 2, name='C1')
        elif PROBLEM == Problem.minimum_vertex_cover:
            for i in range(len(edges)):
                node1, node2 = edges[i]
                model.addConstr(x[node1] + x[node2] >= 1, name=f'C0_{node1}_{node2}')
        elif PROBLEM == Problem.maximum_independent_set:
            for i in range(len(edges)):
                node1, node2 = edges[i]
                model.addConstr(x[node1] + x[node2] <= 1, name=f'C0_{node1}_{node2}')
        elif PROBLEM == Problem.tsp:
            for i in range(num_nodes):
                use_gurobiexample = True
                if use_gurobiexample:
                    model.addConstrs(x.sum(i, '*') == 2 for i in range(num_nodes))
                else:
                    model.addConstr(quicksum(x[i, j] for j in range(num_nodes) if i != j) == 1, name=f"leave_{i}")
                    model.addConstr(quicksum(x[j, i] for j in range(num_nodes) if i != j) == 1, name=f"enter_{i}")
            # for i in range(num_nodes):
            #     for j in range(num_nodes):
            #         if i != j:
            #             model.addConstr(u[i] + adjacency_matrix[(i, j)] <= u[j] + 1000000 * (1 - x[i, j]), name=f'subtour_{i}_{j}')
        elif PROBLEM == Problem.knapsack:
            model.addConstr(quicksum(items[i][0] * x[i] for i in range(num)) <= weight, "knapsack")
        elif PROBLEM == Problem.set_cover:
            for u in range(total_elements):
                model.addConstr(quicksum(x[i] for i, subset in enumerate(subsets) if u in subset) >= 1,
                                name=f"cover_{u}")

    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)

    # reportFile = open('../result/', 'w')

    result_filename = calc_result_file_name(filename)
    model._startTime = time.time()
    # model._reportFile = open(result_filename, 'w')
    model._interval = GUROBI_INTERVAL  # 每隔一段时间输出当前可行解，单位秒
    if PROBLEM == Problem.knapsack:
        model._attribute = {'data_filename': filename, 'result_filename': result_filename}
    elif PROBLEM == Problem.set_cover:
        model._attribute = {'data_filename': filename, 'result_filename': result_filename}
    else:
        model._attribute = {'data_filename': filename, 'result_filename': result_filename, 'num_nodes': num_nodes}
    if GUROBI_VAR_CONTINUOUS:
        # for v in model.getVars():
        #     v.setAttr('vtype', GRB.CONTINUOUS)
        model.update()
        r = model.relax()
        r.update()
        if GUROBI_INTERVAL is None:
            r.optimize()
        else:
            r.optimize(mycallback_write)

        if_write_others = False
        if if_write_others:
            r.write("../result/result.lp")
            r.write("../result/result.mps")
            r.write("../result/result.sol")
        x_values = []
        # for i in range(num_nodes):
        #     var = r.getVarByName(x[i].VarName)
        #     x_values.append(var.x)
        # var = r.getVarByName(x.VarName)
        vars_in_model = [var for var in model.getVars() if "x" in var.VarName]
        name = "x"
        names_to_retrieve = [f"{name}[{i}]" for i in range(num_nodes)]

        for i in range(num_nodes):
            var = r.getVarByName(names_to_retrieve[i])
            x_values.append(var.solution)
        print(f'values of x: {x_values}')
        return x_values

    if GUROBI_INTERVAL is None:
        if PROBLEM == Problem.tsp:
            model.optimize(subtourelim)
        else:
            model.optimize()
    else:
        model.optimize(mycallback_write)

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        infeasibleConstrName = [c.getAttr('ConstrName') for c in model.getConstrs() if
                                c.getAttr(GRB.Attr.IISConstr) > 0]
        print('infeasibleConstrName: {}'.format(infeasibleConstrName))
        model.write('../result/model.ilp')
        sys.exit()

    elif model.getAttr('SolCount') >= 1:  # get the SolCount:
        # result_filename = '../result/result'
        if PROBLEM in [Problem.maxcut, Problem.minimum_vertex_cover, Problem.maximum_independent_set,
                       Problem.graph_partitioning]:
            x_values = [x[i].x for i in range(num_nodes) if i in x]
        elif PROBLEM == Problem.tsp:
            x_values = [[x[i, j].x if (i, j) in x else 0 for j in range(num_nodes)] for i in range(num_nodes)]
            # varlist = [v for v in model.getVars() if 'x' in v.VarName]
            tour = subtour(model, x_values)
            model._attribute['tour'] = tour
            print("tour: ", tour)
        elif PROBLEM == Problem.set_cover:
            x_values = [x[i].x for i in range(total_subsets)]
        elif PROBLEM == Problem.knapsack:
            x_values = [x[i].x for i in range(num)]
        else:
            x_values = []
        if PROBLEM not in [Problem.knapsack, Problem.set_cover]:
            model._attribute['graph'] = graph
        write_result_gurobi(model, result_filename, time_limit)
        return x_values

    num_vars = model.getAttr(GRB.Attr.NumVars)
    num_constrs = model.getAttr(GRB.Attr.NumConstrs)
    print(f'num_vars: {num_vars}, num_constrs: {num_constrs}')
    print('obj:', model.getObjective().getValue())
    vars = model.getVars()

    if model.getAttr('SolCount') == 0:  # model.getAttr(GRB.Attr.SolCount)
        print("No solution.")
    print("SolCount: ", model.getAttr('SolCount'))
    # except Exception as e:
    #     print("Exception!")

    scores = [model.getObjective().getValue()]
    alg_name = 'Gurobi'
    if plot_fig_:
        plot_fig(scores, alg_name)

    print(f"model.Runtime: {model.Runtime}")
    print()

    x_values = []
    for i in range(num_nodes):
        x_values.append(x[i].x)
    print(f'values of x: {x_values}')
    return x_values


def run_gurobi_over_multiple_files(prefixes: List[str], time_limits: List[int], directory_data: str):
    files = calc_txt_files_with_prefixes(directory_data, prefixes)
    files.sort()
    for i in range(len(files)):
        filename = files[i]
        print(f'Start the {i}-th file: {filename}')
        for j in range(len(time_limits)):
            run_using_gurobi(files[i], None, time_limits[j])
    assert 'data' in directory_data
    directory_data2 = copy.deepcopy(directory_data)
    directory_result3 = directory_data2.replace('data', 'result')
    index = directory_result3.find('result')
    directory_result4 = directory_result3[:index] + 'result'
    # directory_result4 = directory_result3.split('/')
    # directory_result = directory_result3.replace('/' + directory_result4[-1], '')
    avg_std = calc_avg_std_of_objs(directory_result4, prefixes, time_limits)


if __name__ == '__main__':
    run_one_file = False
    if run_one_file:
        filename = '../data/syn_BA/BA_100_ID0.txt'
        time_limits = GUROBI_TIME_LIMITS

        from L2A.maxcut_simulator import SimulatorMaxcut, load_graph_list
        from L2A.evaluator import *

        graph_name = 'gset_14'

        graph = load_graph_list(graph_name=graph_name)
        simulator = SimulatorMaxcut(sim_name=graph_name, graph_list=graph)

        x_str = X_G14
        num_nodes = simulator.num_nodes
        encoder = EncoderBase64(encode_len=num_nodes)

        x = encoder.str_to_bool(x_str)
        vs = simulator.obj(xs=x[None, :])
        print(f"objective value  {vs[0].item():8.2f}  solution {x_str}")

        run_using_gurobi(filename, x, time_limit=time_limits[0], plot_fig_=True)
        directory = '../result'
        prefixes = ['syn_BA_']
        avg_std = calc_avg_std_of_objs(directory, prefixes, time_limits)

    run_multi_files = True
    if run_multi_files:
        run_syndistr = True
        # time_limits = GUROBI_TIME_LIMITS
        # time_limits = [10 * 60, 20 * 60, 30 * 60, 40 * 60, 50 * 60, 60 * 60]
        if run_syndistr:
            directory_data = '../data/syn_BA'
            prefixes = ['BA_100_']

        if PROBLEM == Problem.tsp:
            directory_data = '../data/tsplib'
            # prefixes = ['g', 'k', 'l', 'p', 'r', 's', 't', 'u']
            # prefixes = ['a', 'b', 'c', 'd', 'e', 'f']

            # prefixes = ['a', 'b', 'c']
            # prefixes = ['d']
            # prefixes = ['e']

            # prefixes = ['g', 'k']
            # prefixes = ['l']
            # prefixes = ['k']
            # prefixes = ['l']
            # prefixes = ['p']
            # prefixes = ['r']
            # prefixes = ['s', 't', 'u']
            prefixes = ['a5']

        run_gurobi_over_multiple_files(prefixes, GUROBI_TIME_LIMITS, directory_data)

        directory_result = '../result'
        avg_std = calc_avg_std_of_objs(directory_result, prefixes, GUROBI_TIME_LIMITS)

        run_knapsack = False
        if run_knapsack:
            directory_data = '../data/knapsack'
            prefixes = ['kp_']

        run_set_cover = False
        if run_set_cover:
            directory_data = '../data/set_cover'
            prefixes = ['set_cover_']

    pass
