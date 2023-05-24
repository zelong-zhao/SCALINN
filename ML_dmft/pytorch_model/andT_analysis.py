# -*- coding: utf-8 -*-
"""
Anderson transformer analysis tool page.
===============================


# Author: Zelong Zhao 2022
"""
from .utility.args_parser_inputs import analysis_args
from .benchmark_tasks import worst_and_best

def _plot_one_file(args):
    worst_and_best(args)

def main():
    args = analysis_args()
    
    if args.plot_one_file:
        _plot_one_file(args)

if __name__ == '__main__':
    main()