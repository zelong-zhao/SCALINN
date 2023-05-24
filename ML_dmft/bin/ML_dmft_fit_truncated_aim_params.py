#! /usr/bin/env python
import argparse
import numpy as np
from ML_dmft.utility.tools import read_params,flat_aim
from ML_dmft.torch_model.benchmark_tasks.truncated_bath_picker import Choose_Value_From_Input

def read_file_name():
    parser = argparse.ArgumentParser(description='ML dmft inputs')
    parser.add_argument('--num_bath_out','--bath','-bath','-b',type=int, metavar='n',
                    help='file name must present',required=True) 
    parser.add_argument('--out-file','--out','-out','-o',type=str, metavar='f',
                    help='file name must present',required=True)
    parser.add_argument('--index-in-permutation','-i',type=int,required=True,help='index in permutation')
    args = parser.parse_args()
    return args


def main():
    read_args = read_file_name()

    file_name_out = read_args.out_file
    num_bath_output = read_args.num_bath_out
    idx_permutation = read_args.index_in_permutation

    assert file_name_out != 'aim_params.csv'
    in_file = './db/aim_params.csv'
    outfile = f'./db/{file_name_out}'

    print(f"{50*'#'}")
    print(f"In fitting hyb {num_bath_output=}, \nwill read {in_file=} and write to \n{outfile=}")
    params = read_params(in_file)

    NUM_BATH_INPUT = params[0]['N']

    print(f"This code will ramdonly use {num_bath_output} of {params[0]['N']}.")
    assert NUM_BATH_INPUT > num_bath_output, 'readuce bath sites'

    #FOR SYMMETRY CASE:
    E_p,V_p = np.array(params[0]['E_p']),np.array(params[0]['V_p'])
    #INPUT
    NUM_BATH_HALF_INPUT = int(NUM_BATH_INPUT/2)
    ODD_BATH_NUM = False
    if NUM_BATH_INPUT%2 != 0: 
        ODD_BATH_NUM = True
        assert E_p[-1] == 0
    print(f"{ODD_BATH_NUM=} {NUM_BATH_HALF_INPUT=}")
    #OUTPUT
    num_bath_half_output = int(num_bath_output/2)
    odd_bath_num_output = False
    if num_bath_output%2 != 0: 
        odd_bath_num_output = True
    print(f"{odd_bath_num_output=} {num_bath_half_output=}")

    print(f"{V_p=}\n{E_p=}")
    value_picker = Choose_Value_From_Input(idx_permutation=idx_permutation,
                                            NUM_BATH_HALF_INPUT=NUM_BATH_HALF_INPUT,
                                            num_bath_half_output=num_bath_half_output,
                                           ODD_BATH_NUM=ODD_BATH_NUM,
                                           odd_bath_num_output=odd_bath_num_output)
    E_p_out,V_p_out=value_picker(E_p,V_p)
    print(f"{E_p_out=}\n{V_p_out=}")
    print(f"{50*'#'}")


    for idx,param in enumerate(params):
        E_p_out,V_p_out=value_picker(param['E_p'],param['V_p'])
        params[idx]['N'],params[idx]['E_p'],params[idx]['V_p']=num_bath_output, E_p_out,V_p_out

    params = [flat_aim(item)[0] for item in params]
    np.savetxt(outfile, params, delimiter=",", fmt="%1.6f")

if __name__=='__main__':
    main()