#!/usr/bin/bash

serial_run_CMD() {
    ML_dmft_gendb.py -inp $gen_db_inputs

	ML_dmft_fit_truncated_aim_params.py -bath 5 -o 'bath5_0_truncated_aim_params.csv' -i 0
	ML_dmft_fit_truncated_aim_params.py -bath 5 -o 'bath5_1_truncated_aim_params.csv' -i 1
	ML_dmft_fit_truncated_aim_params.py -bath 5 -o 'bath5_2_truncated_aim_params.csv' -i 2

}


spawn_run_CMD() {
for i in $(seq 1 $num_subprocess)
do
	mkdir db_$i
	pushd db_$i
	serial_run_CMD &
	popd
done
wait

}

set -e

prefix=$(dirname "$(realpath "$0")")"/"

source $prefix'example_lib.sh'
num_subprocess=$1
check_integer $num_subprocess

gen_db_inputs=$prefix'dbgen_inputs.yml'

spawn_run_CMD