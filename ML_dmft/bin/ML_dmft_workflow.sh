#!/usr/bin/bash
cat << "EOF" > example_ML_dmft_spawn.sh

#!/usr/bin/bash
set -xe

serial_run_CMD() {
    ML_dmft_gendb.py -inp $gendb_input

    ML_dmft_solve_PT.py -inp $solver_input

	ML_dmft_solve_ED.py -inp $solver_input
}


spawn_run_CMD() {

for i in {1..32..1}
do
	mkdir db_$i
	pushd db_$i
	serial_run_CMD &
	popd
done
wait

ML_dmft_merge.py --merge-small-db
rm -rf ./db_*

mv ./merged_db ./db
}

spawn_run_CMD
EOF
solver_input='solver_inputs.yml'
gendb_input='gendb_inputs.yml'

chmod +x ./example_ML_dmft_spawn.sh