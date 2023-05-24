#! /usr/bin/env python

from ML_dmft.main.solve_db import solve_AIM
from ML_dmft.utility.read_yml import solve_db_args
from ML_dmft.utility.tools import read_file_name_solver


if __name__ == "__main__":
    CMD_args = read_file_name_solver()
    args = solve_db_args(CMD_args.file)
    solve_AIM(args,CMD_args.solver)