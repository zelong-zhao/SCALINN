#! /usr/bin/env python

from ML_dmft.main.solve_PT import solve_AIM
from ML_dmft.utility.read_yml import solve_db_args
from ML_dmft.utility.tools import read_file_name


if __name__ == "__main__":
    file_name = read_file_name().file
    args = solve_db_args(file_name)
    solve_AIM(args)