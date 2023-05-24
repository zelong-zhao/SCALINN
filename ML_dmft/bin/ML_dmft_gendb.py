#! /usr/bin/env python
from ML_dmft.utility.read_yml import gen_db_args
from ML_dmft.main.gen_db import gen_db
from ML_dmft.utility.tools import read_file_name

if __name__ == "__main__":
    file_name = read_file_name().file
    args = gen_db_args(file_name)
    gen_db(args)