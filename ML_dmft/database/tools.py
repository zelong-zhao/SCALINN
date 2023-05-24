import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use(plt.style.available[-2])
import pandas as pd
import numpy as np
from ML_dmft.database.constant import default_solver
from ML_dmft.utility.mpi_tools import mpi_rank

def verify_database_match(data_dict):
    total_number_data=[]
    for solver in data_dict:
        if mpi_rank()==0: print(solver)
        total_number_data.append(len(data_dict[solver]))
    assert len(set(total_number_data)) == 1,'number of data is not identical for different solver'
    number_of_data=list(set(total_number_data))[0]

    return number_of_data

def get_single_distribution(data_dict,variable,solver=default_solver):
    variable_list=[]
    for item in data_dict[solver]:
        variable_list.append(item['data info'][variable])
    return (list(set(variable_list)))

def get_all_distribution(data_dict,solver=default_solver):
    data_base_info=data_dict[solver]

    data_distribution_list=[]
    for item in data_base_info:
        data_distribution={}
        
        data_distribution['beta']=item['data info']['beta']
        data_distribution['U']=item['AIM params']['U']
        if 'W' in item['AIM params']:
            data_distribution['W']=item['AIM params']['W']

        data_distribution['eps']=item['AIM params']['eps']

        for item2 in item['AIM params']:
            aim_params= item['AIM params'][item2]
            if type(aim_params) != float and type(aim_params) != int:
                for idx,item3 in enumerate(aim_params):
                    label=' '.join([item2,str(idx+1)])
                    data_distribution[label]=item3
        data_distribution_list.append(data_distribution)

    aim_pd_data=pd.DataFrame(data_distribution_list)

    ax=aim_pd_data.hist(bins=50,
        xlabelsize=10, ylabelsize=10,
        figsize=(12,12))

    fig = ax[0][0].get_figure()
    
    fig.savefig("database_viz.png", format = "png")

    return

def cap_few_data(input_y,targe_n_points):
    idxs = np.linspace(0, len(input_y[0]) - 1, targe_n_points).astype(int)
    out=np.zeros((1,len(idxs)))
    for i,idx in enumerate(idxs):
        out[0][i]=input_y[1][idx]
    return out
