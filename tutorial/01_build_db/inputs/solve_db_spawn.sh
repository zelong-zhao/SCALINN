
#!/usr/bin/bash

check_integer(){
   re='^[0-9]+$'
   if ! [[ $1 =~ $re ]] ; then
   echo "error: Not a number" >&2; exit 1
fi
}

serial_run_CMD() {
	ML_dmft_solve_db.py -i $solv_db_input -s IPT

	ML_dmft_solve_db.py -i $solv_db_input -s HubbardI

	for inputs_file in "${ed_input_list[@]}"; do
		ML_dmft_solve_ED.py -inp $inputs_file
	done

	ML_dmft_solve_ED.py -inp $solv_db_input

}


spawn_run_CMD() {
check_integer $num_subprocess
for i in $(seq 1 $num_subprocess)
do
	pushd db_$i
	serial_run_CMD &
	popd
done
wait
}


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
set -e

num_subprocess=$1


prefix=$(dirname "$(realpath "$0")")"/"
source $prefix'example_lib.sh'


beta=100

solv_db_input=$prefix'dbsolv_main.yml'
# solv_db_bath4_input=$prefix'dbsolv_bath4.yml'

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#creating inputs
example_truncate_ed_tmp 5 $beta 'bath5_0_truncated_aim_params.csv' 'ED_KCL_5_truncated_0' $prefix'dbsolv_bath5_truncated_idx_0.yml'
example_truncate_ed_tmp 5 $beta 'bath5_1_truncated_aim_params.csv' 'ED_KCL_5_truncated_1' $prefix'dbsolv_bath5_truncated_idx_1.yml'
example_truncate_ed_tmp 5 $beta 'bath5_2_truncated_aim_params.csv' 'ED_KCL_5_truncated_2' $prefix'dbsolv_bath5_truncated_idx_2.yml'
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

ed_input_list=(
			$prefix'dbsolv_bath5_truncated_idx_0.yml' $prefix'dbsolv_bath5_truncated_idx_1.yml' $prefix'dbsolv_bath5_truncated_idx_2.yml'
			)


spawn_run_CMD
