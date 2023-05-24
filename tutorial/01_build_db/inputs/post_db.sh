#!/usr/bin/bash

Analysis_run_CMD() {

if [ -d './db' ] ; then
	echo "db exist!"
	exit 1

else
	ML_dmft_merge.py --merge-small-db &&
	rm -rf ./db_* &&
	mv ./merged_db ./db 
fi

for solver in "${solver_list[@]}"; do
	ML_dmft_database_post.py --reshape --searched_key G_iw --target_shape 32 --solver_list $solver &&
	ML_dmft_database_post.py --merge-to-csv --searched_key G_iw_32 --solver_list $solver &&
	ML_dmft_database_analysis.py --info --solver $solver --basis G_iw_32 &
done

wait

### database summary
for solver in "${solver_list[@]}"; do
    ML_dmft_database_analysis.py --summary --solver $solver &&
    	    mv database_viz.png $solver'_database_viz.png'
done

wait
}



solver_list=(
			'ED_KCL_5_truncated_0' 'ED_KCL_5_truncated_1' 'ED_KCL_5_truncated_2'
			'ED_KCL_7' 'IPT' 'HubbardI'
			)


set -e

Analysis_run_CMD
