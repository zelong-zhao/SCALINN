import numpy as np
from h5 import HDFArchive
import pickle as pkl
import os,csv,yaml,warnings,argparse
import math

def read_file_name():
    parser = argparse.ArgumentParser(description='ML dmft inputs')
    parser.add_argument('--file','-f','-inp','-i',type=str, metavar='f',
                    help='file name must present',required=True) 
    args = parser.parse_args()
    return args

def read_file_name_solver():
    parser = argparse.ArgumentParser(description='ML dmft inputs')
    parser.add_argument('--file','-f','-inp','-i',type=str, metavar='f',
                    help='file name must present',required=True) 
    parser.add_argument('--solver','-solver','-s','-S',type=str,default='HubbardI',
                    help='solver to solve db',choices=['IPT','HubbardI']) 
    args = parser.parse_args()
    return args


def enforce_types(func):
    def wrapper(*args, **kwargs):
        for arg, arg_type in zip(args, func.__annotations__.values()):
            if not isinstance(arg, arg_type):
                raise TypeError(f"Argument {arg} is not of type {arg_type}")
        return func(*args, **kwargs)
    return wrapper

def get_bool_env_var(name:str)->bool:
    """Reads a boolean value from an environment variable.
    
    Returns True if the variable is set and has a value of "true", "1", or "yes"
    (case-insensitive). Returns False otherwise.
    """
    str_value = os.getenv(name)
    
    if str_value is not None and str_value.lower() in ["true", "1", "yes"]:
        return True
    else:
        if str_value is None: warnings.warn(f"{name} not found",DeprecationWarning)
        return False

class EarlyStopper_DMFT:
    def __init__(self, patience:int=1, min_delta:float=0.):
        r"""
        EarlyStopper_DMFT
        -----------------
        args:
        -----
            patience: number of tolerence iteration
            min_delta: tolerence fluaction Bool(err > self.min_err + self.min_delta)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_err = np.inf

    def early_stop(self, err):
        if err < self.min_err:
            self.min_err = err
            self.counter = 0
        elif err > (self.min_err + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def iw_obj_flip(iw_obj_in):
    """
    iw_obj_flip
    re(iw_obj)[1024:2048]=re(iw_obj_in)[0:]
    """
    dim=len(iw_obj_in)
    dtype=iw_obj_in.dtype
    iwb_obj_out=np.zeros((int(2*dim),1),dtype=dtype)
    iwb_obj_out[dim:2*dim]=iw_obj_in
    iwb_obj_out[0:dim].real=np.flip(iw_obj_in.real)
    iwb_obj_out[0:dim].imag=np.flip(iw_obj_in.imag)*(-1)
    return iwb_obj_out

def save2csv(input_list,fname):
    input_list=np.array(input_list)
    dim=input_list.shape
    input_list=input_list.reshape(dim[0],dim[-1])
    print(fname,'is created')
    np.savetxt(fname, input_list, delimiter="\t", fmt="%1.6f")

def save2h5_from_saveGF2list(input_list,fname):
    with HDFArchive(fname,'w') as ar:
        for idx,dictionary_ in enumerate(input_list):
            idx=str(idx)
            if not idx in ar:
                ar.create_group(idx)
            for name in dictionary_:
                ar[idx][name] = dictionary_[name]
    return

def save2h5_from_saveGF2list_MPI(input_list,fname):
    input_list=input_list[0]
    with HDFArchive(fname,'w') as ar:
        for idx,dictionary_ in enumerate(input_list):
            idx=str(idx)
            if not idx in ar:
                ar.create_group(idx)
            for name in dictionary_:
                ar[idx][name] = dictionary_[name]
    return

def save2pkl(input_list,fname):
    print(fname,'is created')
    with open(fname,'wb') as f:
        pkl.dump(input_list,f)
    return 

def label_zz(pfix, basis,prefix=os.getcwd()):
    """
    Naming function for the database files 
    """
    directory = "/db/"
    cwd = prefix
    parent = cwd + directory
    if basis=='legendre': basis='leg'

    label = parent + pfix + "_" + str(basis) + ".csv"
    return label

def label_params(pfix, basis):
    """
    Naming function for the database files 
    """
    directory = "/db/"
    cwd = os.getcwd()
    parent = cwd + directory
    whoami = mpi_rank()
    if whoami < 10:
        str_who = "0"+str(whoami)
    else:
        str_who = str(whoami)        
    label = parent + pfix + "_" + str_who + ".csv"        
    return label

def create_data_dir():
    """
    Creation/deletion of the ./db/ directory
    """
    
    print("Checking if ./db/ exists on root node")
    data_dir = os.getcwd()+"/db/"
    isdir = os.path.isdir(data_dir)
    if isdir: 
        raise SyntaxError('File exists, delete to')
        # print("Database already exists, we delete")
        # shutil.rmtree(data_dir)
        # print("Database is here: ", data_dir)
        # os.mkdir(data_dir)    
    else:
        print("Database is here: ", data_dir)
        os.mkdir(data_dir)    

def read_params(filename):
    """
    Reads params from /db/*.csv files 
    """    
    params_list = []
    with open(filename) as f:
        reader = csv.reader(f)
        for line in reader:
            params = {}
            params["U"] = float(line[0])
            params["eps"] = float(line[1])
            params['W'] = float(line[2])
            N = int(float(line[3]))
            params['N'] = N
            params["E_p"] = [float(e) for e in line[4:4+N]]
            params["V_p"] = [float(v) for v in line[4+N:4+2*N]]
            params_list.append(params)
    return params_list

def params_to_dict(line):
    """
    Reads params from /db/*.csv files 
    """    
    
    params = {}
    params["U"] = float(line[0])
    params["eps"] = float(line[1])
    params['W'] = float(line[2])
    N = int(float(line[3]))
    params['N'] = N
    params["E_p"] = [float(e) for e in line[4:4+N]]
    params["V_p"] = [float(v) for v in line[4+N:4+2*N]]
    return params

def flat_aim(params):
    """
    aim params in, generate flat aim output.
    """
    out_list = []
    for item2 in params:
        out_list.append(params[item2]) 

    out_list=np.hstack(out_list)
    out_list=out_list.reshape(len(out_list),1).T

    return out_list

def flat_aim_advanced(params,kept_keys=['U','eps','E_p','V_p']):
    """
    aim params in, generate flat aim output.
    """
    out_list = []
    for item2 in kept_keys:
        out_list.append(params[item2]) 

    out_list=np.hstack(out_list)
    out_list=out_list.reshape(1,len(out_list))
    return out_list
    
def del_file(filename):     
    if os.path.isfile(filename):
        os.remove(filename)

def extract_from_csv(filename, index):
    y_axis = []
    with open(filename) as f:
        reader = csv.reader(f)
        interestingrows=[row for idx, row in enumerate(reader) if idx == index]
    [y_axis.append(float(i)) for i in interestingrows[0]]
    y_axis = np.asarray(y_axis)
    return y_axis
        
def plot_from_csv(filename, x_axis, index, descrip, axes, hyb_param, cin, ms, mt, skip, dots):

    y_axis = extract_from_csv(filename, index)
    if hyb_param["basis"] == "legendre" and hyb_param["poly_semilog"]:        
        axes.semilogy(x_axis, np.abs(y_axis),
                      '-o', label = descrip)
        axes.set_ylim([1e-6,1e+1])
    else:
        axes.plot(x_axis[::skip], y_axis[::skip],
                  markersize=ms,
                  linestyle=mt,
                  marker=dots,
                  color=cin,
                  label = descrip)        
    return axes
        
def get_obj_prop(obj):        
    for property, value in vars(obj).items():
        print(property, ":", value)


def dump_to_yaml(file,dict):
    print(f'writting to {file}')
    with open(file, 'w') as outfile:
        yaml.dump(dict, outfile)

def read_yaml(file):
    with open(file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return config