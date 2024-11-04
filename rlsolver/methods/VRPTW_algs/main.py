import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

from rlsolver.methods.VRPTW_algs.config import (Config, Alg)
from rlsolver.methods.VRPTW_algs.impact_heuristic import run_impact_heuristic
from rlsolver.methods.VRPTW_algs.column_generation import run_column_generation
def main():
    if Config.ALG == Alg.impact_heuristic:
        run_impact_heuristic()
    elif Config.ALG == Alg.column_generation:
        run_column_generation()

    pass


if __name__ == '__main__':
    main()