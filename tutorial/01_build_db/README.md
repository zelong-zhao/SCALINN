# Generate database

## Generate SIAM parameters 
In this section we how to generate single impurity Anderson impurity model database.  All the inputs file located in INPUTS directory. The main input is `./inputs/dbgen_inputs.yml`

```yaml
main-settings:
    aim_samples : 10 #int

gf-settings:
    beta : 100
```

aim_samples controls number sample generate for each parallel process.


```bash
bash ./inputs/gen_db_spawn.sh  $NP

bash ./inputs/gen_db_spawn.sh  18
```

$NP stands for number of parallel process. While inside of each subprocess, there are two parts. 1) Randomly generate SIAM samples by ``ML_dmft_gendb.py`` which will generate `./db/aim_param.csv`. 2) truncation of Hamitonian by ``ML_dmft_fit_truncated_aim_params.py``. 

```bash	
ML_dmft_fit_truncated_aim_params.py 
-bath $bath_size [Truncated bath size. e.g. 5]
-o $outfile [outfile 'example.csv']
-i $truncated_index [index 0]
``` 

This code will generate `./db/$outfile`. `$truncated_index` is there because there may many different truncated inputs. 

---
## Solve SIAM samples

After SIAM samples are generated, 'solve_db_spawn.sh' will solve items in `./db/aim_param.csv` line by line. 

```
bash ./inputs/solve_db_spawn.sh  $NP

bash ./inputs/solve_db_spawn.sh  18
```

The outputs will be in `./db/{solver used}/{which SIAM params or row number solved in aim_param.csv}`

```bash 
ls ./db/ED_KCL_7/0/

aim_params.dat  beta.dat  G0_iw.dat  G_iw.dat  G_l.dat  G_tau.dat  logfile-p1_DC.dat  n_imp.dat  Sigma_iw.dat  Z.dat
```

---
## Post processing data 

Now database is still scattered, they will be merged in together in this step 

There are more different type of data we did not generated in `./db/ED_KCL_7/0/` at first place. They will also be solved by in this step. 


```
bash inputs/post_db.sh 
```
FYI: left more number of cores in this step.