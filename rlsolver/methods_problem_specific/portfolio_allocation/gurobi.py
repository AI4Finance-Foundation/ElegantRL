from gurobipy import *
import os
from typing import List
import networkx as nx
import sys
from util import read_csv
from util import read_npy
class Config():
    lb = -10 ** (-3)
    ub = 10 ** (-3)
    scale = 1000




# the file has been open
def write_statistics(model, new_file, add_slash=False):
    prefix = '// ' if add_slash else ''
    new_file.write(f"{prefix}obj: {model.objVal}\n")
    new_file.write(f"{prefix}running_duration: {model.Runtime}\n")
    new_file.write(f"{prefix}gap: {model.MIPGap}\n")
    new_file.write(f"{prefix}obj_bound: {model.ObjBound}\n")
    # new_file.write(f"time_limit: {time_limit}\n")
    time_limit = model.getParamInfo("TIME_LIMIT")
    new_file.write(f"{prefix}time_limit: {time_limit}\n")


# running_duration (seconds) is included.
def write_result_gurobi(model, filename: str = 'result/result'):
    with open(f"{filename}.sta", 'w', encoding="UTF-8") as new_file:
        write_statistics(model, new_file, False)
    with open(f"{filename}.sov", 'w', encoding="UTF-8") as new_file:
        new_file.write('values of vars: \n')
        vars = model.getVars()
        for var in vars:
            new_file.write(f'{var.VarName}: {var.x}\n')
    model.write(f"{filename}.mst")
    model.write(f"{filename}.lp")
    model.write(f"{filename}.mps")
    model.write(f"{filename}.sol")


def run_using_gurobi(filename: str, time_limit: int = None):
    model = Model("portfolio")
    if 'csv' in filename:
        AMOUNT = read_csv(filename)
    else:
        AMOUNT = read_npy(filename)
    num_vars = len(AMOUNT)

    x = model.addVars(num_vars, vtype=GRB.BINARY, name="x")
    model.setObjective(quicksum(x[i] for i in range(num_vars)),
                       GRB.MAXIMIZE)

    # constrs
    model.addConstr(quicksum(AMOUNT[i] * Config.scale * x[i] for i in range(num_vars)) <= Config.ub * Config.scale, name='C0')
    model.addConstr(quicksum(AMOUNT[i] * Config.scale * x[i] for i in range(num_vars)) >= Config.lb * Config.scale, name='C1')
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        infeasibleConstrName = [c.getAttr('ConstrName') for c in model.getConstrs() if
                                c.getAttr(GRB.Attr.IISConstr) > 0]
        print('infeasibleConstrName: {}'.format(infeasibleConstrName))
        model.write('result/model.ilp')
        sys.exit()

    elif model.getAttr('SolCount') >= 1:  # get the SolCount:
        write_result_gurobi(model)

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


if __name__ == '__main__':
    file = 'data_subset_sum/xxx.csv'
    run_using_gurobi(file)

    pass
