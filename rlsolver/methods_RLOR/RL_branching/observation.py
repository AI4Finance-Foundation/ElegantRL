import numpy as np
import pyscipopt as scip


def variable_features(model, buffer=None):
    ncols = model.getNLPCols()
    cols = model.getLPCols()
    vars = [col.getVar() for col in cols]
    if buffer is None:
        coefs = np.array([col.getObj() for col in cols], dtype=np.float32)
        types = np.array([var.getType() for var in vars], dtype=np.int32)
        type_binary = np.asarray(types == 0)
        type_integer = np.asarray(types == 1)
        type_implint = np.asarray(types == 2)
        type_continu = np.asarray(types == 3)
    else:
        coefs = buffer['coefs']
        type_binary = buffer['type_binary']
        type_integer = buffer['type_integer']
        type_implint = buffer['type_implint']
        type_continu = buffer['type_continu']

    basis_status = np.array([col.getBasisStatus() for col in cols], dtype=np.object_)
    b_stat_lower = np.asarray(basis_status == "lower")
    b_stat_basic = np.asarray(basis_status == "basic")
    b_stat_upper = np.asarray(basis_status == "upper")
    b_stat_zero = np.asarray(basis_status == "zero")

    lbs = np.array([np.nan if model.isInfinity(abs(lb)) else lb
                    for lb in [col.getLb() for col in cols]], dtype=np.float32)
    ubs = np.array([np.nan if model.isInfinity(abs(ub)) else ub
                    for ub in [col.getUb() for col in cols]], dtype=np.float32)
    ages = np.array([col.getAge() for col in cols], dtype=np.int32)
    reduced_cost = np.array([model.getColRedcost(col) for col in cols], dtype=np.float32)
    sol_vals = np.array([col.getPrimsol() for col in cols], dtype=np.float32)
    sol_fracs = np.array([model.feasFrac(sol_val) for sol_val in sol_vals], dtype=np.float32)
    sol_is_at_lb = np.array([model.isEQ(sol_val, lb) for sol_val, lb in zip(sol_vals, lbs)], dtype=np.int32)
    sol_is_at_ub = np.array([model.isEQ(sol_val, ub) for sol_val, ub in zip(sol_vals, ubs)], dtype=np.int32)

    inc_vals = np.empty(shape=(ncols,), dtype=np.float32)
    inc_vals_avg = np.empty(shape=(ncols,), dtype=np.float32)
    sol = model.getBestSol()
    if sol is not None:
        inc_vals = np.array([model.getSolVal(sol, var) for var in vars], dtype=np.float32)
        inc_vals_avg = np.array([var.getAvgSol() for var in vars], dtype=np.float32)
    sol_fracs[type_continu == 1] = 0  # continuous variables have no fractionality

    obj_norm = np.linalg.norm(coefs)
    obj_norm = 1 if obj_norm <= 0 else obj_norm

    return {
        'coefs': coefs / obj_norm,
        'type_binary': type_binary,
        'type_integer': type_integer,
        'type_implint': type_implint,
        'type_continu': type_continu,
        'b_stat_lower': b_stat_lower,
        'b_stat_basic': b_stat_basic,
        'b_stat_upper': b_stat_upper,
        'b_stat_zero': b_stat_zero,
        'has_lb': ~np.isnan(lbs),
        'has_ub': ~np.isnan(ubs),
        'age': ages / (model.getNLPs() + 5),
        'reduced_cost': reduced_cost / obj_norm,
        'sol_val': sol_vals,
        'sol_frac': sol_fracs,
        'sol_is_at_lb': sol_is_at_lb,
        'sol_is_at_ub': sol_is_at_ub,
        'inc_val': inc_vals,
        'inc_vals_avg': inc_vals_avg,
    }


def constraint_features(model, buffer):
    nrows = model.getNLPRows()
    rows = model.getLPRows()

    constants = [row.getConstant() for row in rows]
    if buffer is None:
        lhss = np.array([np.nan if model.isInfinity(abs(row.getLhs())) else row.getLhs() - cst
                         for row, cst in zip(rows, constants)], dtype=np.float32)
        rhss = np.array([np.nan if model.isInfinity(abs(row.getRhs())) else row.getRhs() - cst
                         for row, cst in zip(rows, constants)], dtype=np.float32)
        nnon_zeros = np.array([row.getNLPNonz() for row in rows], dtype=np.int32)
        norms = np.array([row.getNorm() for row in rows], dtype=np.float32)
        objcossims = np.empty(shape=(nrows,), dtype=np.float32)
    else:
        lhss = buffer['lhss']
        rhss = buffer['rhss']
        nnon_zeros = buffer['nnzrs']
        objcossims = buffer['objcossims']
        norms = buffer['norms']

    dual_sols = np.array([row.getDualsol() for row in rows], dtype=np.float32)
    ages = np.array([row.getAge() for row in rows], dtype=np.int32)
    activities = np.array([model.getRowLPActivity(row) - cst for row, cst in zip(rows, constants)], dtype=np.float32)
    is_at_lhs = np.array([model.isEQ(activity, lhs) for activity, lhs in zip(activities, lhss)], dtype=np.int32)
    is_at_rhs = np.array([model.isEQ(activity, rhs) for activity, rhs in zip(activities, rhss)], dtype=np.int32)

    if buffer is None:
        # --- This functionality might have to be added to PySCIPOpt directly ---
        # [[ From ds4dm ]]
        # # Inspired from SCIProwGetObjParallelism()
        # SCIPlpRecalculateObjSqrNorm(scip.set, scip.lp)
        # prod = rows[i].sqrnorm * scip.lp.objsqrnorm
        # row_objcossims[i] = rows[i].objprod / SQRT(prod) if SCIPisPositive(scip, prod) else 0.0
        # [[ From ecole ]]
        # norm_prod = SCIProwGetNorm(row) * SCIPgetObjNorm(scip);
        # return row->objprod / norm_prod if (SCIPisPositive(scip, norm_prod)) else 0.0
        # --- Unless we can get objprod from row ---

        # Objective cosine similarity
        objcossims = np.array([0.0 for _ in range(nrows)], dtype=np.float32)

    # Row coefficients
    non_zeros = np.sum(nnon_zeros)
    if buffer is None:
        col_ids = np.empty(shape=(non_zeros,), dtype=np.int32)
        row_ids = np.empty(shape=(non_zeros,), dtype=np.int32)
        values = np.empty(shape=(non_zeros,), dtype=np.float32)

        j = 0
        for i, row in enumerate(rows):
            # coefficient indexes and values
            row_cols = row.getCols()
            row_vals = row.getVals()
            for k in range(nnon_zeros[i]):
                col_ids[j + k] = row_cols[k].getLPPos()
                row_ids[j + k] = i
                values[j + k] = row_vals[k]

            j += nnon_zeros[i]
    else:
        col_ids = buffer['nzrcoef']['col_ids']
        row_ids = buffer['nzrcoef']['row_ids']
        values = buffer['nzrcoef']['values']

    return {
        'lhss': lhss,
        'rhss': rhss,
        'nnon_zeros': nnon_zeros,
        'dual_sols': dual_sols,
        'ages': ages,
        'activities': activities,
        'objcossims': objcossims,
        'norms': norms,
        'is_at_lhs': is_at_lhs,
        'is_at_rhs': is_at_rhs,
        'coefs': {
            'col_ids': col_ids,
            'row_ids': row_ids,
            'values': values,
        },
    }


def branching_features(model, node, buffer=None):
    bound_changes = node.getDomchg().getBoundchgs()
    bound_change = bound_changes[0]
    branch_var = bound_change.getVar()
    # preferred branch direction is always AUTO
    branch_dir = branch_var.getBranchDirection()

    branchbound = bound_change.getNewBound()
    var_rootsol = branch_var.getRootSol()
    var_sol = branch_var.getLPSol()
    if model.isInfinity(var_sol):
        var_sol = var_rootsol

    bound_lp_diff = branchbound - var_sol
    root_lp_diff = var_rootsol - var_sol

    pseudo_cost = model.getVarPseudocost(branch_var, branch_dir)
    n_inferences = (model.getVarAvgInferences(branch_var, 1)
                    if bound_change.getBoundchgtype() == 0
                    else model.getVarAvgInferences(branch_var, 0))

    return {
        # 'prio_down': branch_dir == 0,
        # 'prio_up': branch_dir == 1,  # these never activate
        'bound_lp_diff': bound_lp_diff,
        'root_lp_diff': root_lp_diff,
        'pseudo_cost': pseudo_cost,
        'n_inferences': n_inferences,
    }


def node_features(model, node, buffer=None):
    assert sum(node.getNDomchg()) <= 1
    return {
        # 'type_child': node.getType() == 3,  # always true for smartDFS
        # 'type_sibling': node.getType() == 2,
        # 'type_leaf': node.getType() == 4,
        'estimate': node.getEstimate(),
        'node_lb': node.getLowerbound(),
        'relative_bound': None,
        'is_prio_child': node == model.getPrioChild(),
        'node_depth': node.getDepth() / (model.getMaxDepth() + 1),
    }


def global_features(model, buffer=None):
    global_lb = model.getLowerbound()
    global_ub = model.getUpperbound()
    ub_is_infinite = model.isInfinity(global_ub) or model.isInfinity(-global_ub)
    if ub_is_infinite: global_ub = global_lb + 0.2 * (global_ub - global_lb)
    gap_is_infinite = model.isZero(global_lb) or ub_is_infinite
    bound_gap = 0 if gap_is_infinite else (global_ub - global_lb) / abs(global_lb)
    # if these activate, we need to evaluate them
    # if ub_is_infinite or gap_is_infinite:
    #   print("ub_is_infinite or gap_is_infinite")

    max_plunge_depth = max(int(model.getMaxDepth() / 2), 1)
    plunge_depth = model.getPlungeDepth() / max_plunge_depth

    return {
        'plunge_depth': plunge_depth,
        'global_ub': global_ub,
        'bound_gap': bound_gap,
        # 'ub_is_infinite': ub_is_infinite,  # always False
        # 'gap_is_infinite': gap_is_infinite,
    }
