import numpy as np
import time
from ML_dmft.utility.db_gen import db_AIM
from ML_dmft.utility.tools import create_data_dir

def gen_db(args):
    """
    Main routine for the database generation
    """
    # seed = int((time.time()*10e6))
    # np.random.seed(seed)
    # random.seed(seed)
    
    time_start=time.time()
    print("\n#################  Generating database  #####################\n")

    create_data_dir()
    
    db_random = db_AIM(args['db_param'])

    if args['db_param']["mott_num"] == 0:
        print(20*'#',' True Random ',20*'#')
        db_random.create_db()
        local_params = db_random.data_entries
    else:
        print(20*'#','Create Mott Database',20*'#')
        db_random.create_mott_db()
        local_params=db_random.data_entries

    fname="./db/aim_params.csv"
    np.savetxt(fname, local_params, delimiter=",", fmt="%1.6f")
    print(fname,' is created')
    time_finished=time.time()-time_start
    print('*'*20,time_finished,'*'*20)