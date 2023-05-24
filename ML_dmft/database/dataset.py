from typing import Tuple,Callable
import numpy as np
import os,glob
from ML_dmft.utility.tools import read_params,flat_aim_advanced,params_to_dict,iw_obj_flip
from ML_dmft.numpy_GF.GF import complex_to_real,real_to_complex,matsubara_freq
from ML_dmft.utility.tools import dump_to_yaml,read_yaml
from functools import partial
import warnings,copy

class AIM_Dataset():
    """
    load AIM dataset by access to individual elements
    """
    def __init__(self,root,db,solver,load_aim_csv_cpu=False):
        super().__init__()
        self.root=root
        self.db=db
        self.solver=solver

        self.__files_rules__()
        
        if load_aim_csv_cpu :
            self.aim_params_csv=self.__load_aim_csv_to_cpu__()
            n_samples=len(self.aim_params_csv)
            if self.num_sampels() != n_samples:
                print(f'number of files {self.num_sampels()} and number csv items {n_samples}')
                # warnings.warn('num samples in aim csv and directory different')
                raise ValueError('num samples in aim csv and directory different')

        self.n_samples=self.num_sampels()
        
        self.possible_keys=self.__keys_in_dataset__()

        self.single_float_key=['beta','logfile-p1_DC','Z','n_imp']
        self.oneline_array=['aim_params','G_tau','G_l','G0_tau','G0_l']
        self.iw_obj=['Sigma_iw','G0_iw','G_iw']
        self.w_obj=['A_w']

        self.__key_add_shape('G_tau',[64,128,256])
        self.__key_add_shape('G0_tau',[64,128,256])
        self.__key_add_shape('G_l',[8,16,32])
        self.__key_add_shape('G0_l',[8,16,32])
        self.__key_add_shape('Sigma_iw',[8,16,32,64,128,256,512])
        self.__key_add_shape('G0_iw',[8,16,32,64,128,256,512])
        self.__key_add_shape('G_iw',[8,16,32,64,128,256,512])
    
    def num_sampels(self):
        return len(glob.glob(os.path.join(self.root,self.db,'db',self.solver,'*')))

    def __key_add_shape(self,key,added_shape):
        for shape in added_shape:
            if key in self.oneline_array:
                self.oneline_array.append(f'{key}_{shape}')
            elif key in self.iw_obj:
                self.iw_obj.append(f'{key}_{shape}')
        
    def __files_rules__(self):
        self.aim_file_csv=os.path.join(self.root,self.db,'db','aim_params.csv')
        self.item_path_rule=lambda index,key : os.path.join(self.root,self.db,'db',self.solver,str(index),f'{key}.dat')
    
    def __load_aim_csv_to_cpu__(self):
        print('loading aim_params.csv to RAM')
        params_list_dictionary=read_params(self.aim_file_csv) # reads aim_params
        return params_list_dictionary

    def load_merged_G_to_cpu(self,basis):
        r"""
        Loading merged G to cpu
        """
        G_path=os.path.join(self.root,self.db,'db',f'{self.solver}_{basis}.csv')
        G=np.loadtxt(G_path)
        return G
    
    def __keys_in_dataset__(self):
        possible_keys=[]
        path=os.path.join(self.root,self.db,'db',self.solver,str(0))
        if os.path.isdir(path):
            for item in os.listdir(path):
                possible_keys.append(item.split('.')[0])
        else:
            print(f"{path} does exist in searching possible keys")
        return possible_keys

    def numpy_flatten_aim_params(self)->np.ndarray:
        r"""
        return
        ------
        aim_parameters_array: shape=(num_samples,dim_item)
        """
        params_array=[]
        try:
            params_list_dictionary = self.aim_params_csv
            warnings.warn('Warning: RAM has two copies of full aim params!',ResourceWarning)
        except AttributeError:
            params_list_dictionary=self.__load_aim_csv_to_cpu__()

        for param in params_list_dictionary:
            params_array.append(flat_aim_advanced(param,kept_keys=['U','eps','E_p','V_p'])[0]) # flat_aim: 'U', 'eps', 'E_p', 'V_p'
        params_array=np.array(params_array) # array of 'U', 'eps', 'E_p', 'V_p'
        return params_array

    def data_transform(self,key:str,method:str,forward:bool)-> Callable:
        r"""
        data transform
        
        args
        ---- 
            key: basis G_iw_32
                 U","eps","E_k","V_k","E_k_V_k_tot"

            method:'normalise','standardise'
        
        return
        ------
            transform function
        """
        method_list=['normalise','standardise']
        if method not in method_list:
            raise ValueError('not such method in data_transform. method has to be normalise or standardise')

        def normalise(value,key,stats_dict,forward):
            min_ = stats_dict[f"{key}_min"]
            max_ = stats_dict[f"{key}_max"]
            if forward:
                return  (value-min_)/(max_-min_)
            else:
                return  value*(max_-min_) + min_


        def standardise(value,key,stats_dict,forward):
            mean_ = stats_dict[f"{key}_mean"]
            std_ = stats_dict[f"{key}_std"]
            if forward:
                return  (value-mean_)/std_
            else:
                return  (value*std_)+mean_
        
        if key in self.iw_obj:
            fname=os.path.join(self.root,self.db,'db',f"{self.solver}_{key}_stas.ymal")
        elif key in ["U","eps","U_eps_tot","E_k","V_k","E_k_V_k_tot"]:
            fname=os.path.join(self.root,self.db,'db',f"{self.solver}_aim_params_stas.ymal")
        else:
            raise ValueError('not such key in data transform')
        
        stats_dict = read_yaml(fname)
        for item in stats_dict: stats_dict[item]=float(stats_dict[item])
        
        if method == 'normalise':
            normalise_transform = lambda value:normalise(value,key,stats_dict,forward)
            return normalise_transform

        elif method == 'standardise':
            standardise_transform = lambda value:standardise(value,key,stats_dict,forward)
            return standardise_transform


    def __call__(self,index,key):
        item=np.loadtxt(self.item_path_rule(index,key))
        if key in self.single_float_key:
            item=float(item)
        elif key in self.oneline_array:
            if key == 'aim_params':
                item = params_to_dict(item)    
            else:          
                item=item.reshape((len(item),1))
        elif key in self.iw_obj:
            item=item
            item=item.view(dtype=np.complex128)
        elif key in self.w_obj:
            item=item
        else:
            if not any(search_key in key for search_key in ['G_tau','G_iw','G_l','G0']):
                warnings.warn(f"{key} is not classified in database")

        return item
    def __item__(self,index,key):
        return self.__call__(index,key)
        
    def __len__(self):
        return self.n_samples

class AimG_Dataset_RAM():
    def __init__(self,root,db,solver,basis,imag_only=False):
        dataset=AIM_Dataset(root,db,solver,load_aim_csv_cpu=True)
        self.aim_params=dataset.aim_params_csv
        self.G=dataset.load_merged_G_to_cpu(basis)

        if basis in dataset.iw_obj:
            if imag_only:
                self.G=np.array([real_to_complex(item) for item in self.G]).imag
                 

    def __getitem__(self,index):
        return  self.aim_params[index],self.G[index]
    
    def __len__(self):
        return len(self.aim_params)

class Aim_Dataset_Numpy(AIM_Dataset):
    def __init__(self, root, db, solver, load_aim_csv_cpu=False,imag_only=False):
        super().__init__(root, db, solver, load_aim_csv_cpu)
        self.imag_only=imag_only
        
    def __call__(self, index, key):
        r"""
        return:
        -------
        individual items returned by file
        """
        if key in self.iw_obj:
            if not self.imag_only:
                return complex_to_real(super().__call__(index, key))
            else:
                return super().__call__(index, key).imag

        elif key == 'aim_params':
            param=super().__call__(index, key) 
            if param is None:
                print(f"{self.possible_keys=}")
            flat_aim=flat_aim_advanced(param,kept_keys=['U','eps','E_p','V_p']).T
            return flat_aim
        else:
            return super().__call__(index, key) 

    def __item__(self,index,key):
        return self.__call__(index,key)


class Aim_Dataset_special(AIM_Dataset):
    def __init__(self, root, db, solver, load_aim_csv_cpu=False):
        super().__init__(root, db, solver, False)
        
    def __call__(self, index, key):
        r"""
        return:
        -------
        individual items returned by file
        """
        if key in self.iw_obj:
            return super().__call__(index, key).imag

        elif key == 'aim_params':
            param=super().__call__(index, key) 
            if param is None:
                print(f"{self.possible_keys=}")
            flat_aim=flat_aim_advanced(param,kept_keys=['U','eps','E_p','V_p']).T
            return flat_aim
        
        elif key == 'imp_params':
            param=super().__call__(index,'aim_params') 
            imp_param=flat_aim_advanced(param,kept_keys=['U','eps']).T
            return imp_param
        
        elif key == 'hyb_params':
            param=super().__call__(index,'aim_params') 
            onsite = flat_aim_advanced(param,kept_keys=['E_p']).T
            hopping = flat_aim_advanced(param,kept_keys=['V_p']).T
            return onsite,hopping

        else:
            return super().__call__(index, key) 

    def __item__(self,index,key):
        return self.__call__(index,key)



class AimG_Database_oneline(Aim_Dataset_Numpy):
    def __init__(self, root, db, solver, load_aim_csv_cpu=False):
        r"""
        old name of this class
        """
        super().__init__(root, db, solver, load_aim_csv_cpu)
    

class AIM_dataset_meshG(AIM_Dataset):
    def __init__(self, root, db, solver, load_aim_csv_cpu=False):
        super().__init__(root, db, solver, load_aim_csv_cpu)
        pass

    def __call__(self,index:int,key:str,Positive_Semi_defined=True)-> Tuple[Tuple[np.ndarray,np.ndarray],Tuple[str,str]]:
        """
        input:
        ------
        index: index in database db/solver/index
        key  : key in index db/solver/index/{key}.dat

        output:
        -------
        data,info
        data = (mesh:np.ndarray,G:np.ndarray)
        info = (title:str,mesh_type:str)
        """
        beta=super().__call__(0,'beta')
        G=super().__call__(index,key)
        aim_params=super().__call__(index,'aim_params')
        title=AIM_Dataset_Rich.gen_title(aim_params,beta)
        if 'G_tau' in key:
            mesh=np.linspace(0,beta,len(G))
            mesh_type='tau'
        elif 'G_l' in key:
            mesh=np.arange(len(G))
            mesh_type='l'
        elif key in self.iw_obj:
            mesh=matsubara_freq(beta,len(G),Positive_Semi_defined)
            mesh_type='iw'
        elif key in self.w_obj:
            loaded_data=copy.deepcopy(G)
            mesh=loaded_data[:,0]
            del G
            G=loaded_data[:,1]
            mesh_type='w'
        else:
            raise TypeError(f"{key} not supported yet")
        data=(mesh,G)
        info=(title,mesh_type)
        return data,info

    def __item__(self,index,key):
        return self.__call__(index,key)


class AIM_Dataset_Rich():
    def __init__(self,root,db,solver,index):
        """
        includes self.n_iw,self.n_l...
        """

        dataset=AIM_Dataset(root,db,solver)
        self.__procceed__(dataset,index)
        self.__index_in_solver_diretory__=index
        self.title=self.gen_title(self.aim_params,self.beta)

    def __procceed__(self,dataset,index):
        self.solver=dataset.solver
        self.aim_params=dataset(index,'aim_params')
        self.G_l=dataset(index,'G_l')
        self.n_l=len(self.G_l)

        self.G_tau=dataset(index,'G_tau') # better to include G_tau_64.dat，G_tau_128.dat,G_tau_256.dat ...
        self.n_tau=len(self.G_tau)

        self.n_iw=len(dataset(index,'G_iw'))

        self.G_iw=iw_obj_flip(dataset(index,'G_iw'))
        self.Sigma_iw=iw_obj_flip(dataset(index,'Sigma_iw'))
        self.G0_iw=iw_obj_flip(dataset(index,'G0_iw'))

        self.Z=dataset(index,"Z")
        self.n=dataset(index,"n_imp")
        self.beta=dataset(index,'beta')

        self.n_samples=dataset.n_samples
        return
    
    @staticmethod
    def gen_title(aim_params,beta):
        title=''
        for key in aim_params:
            try:
                unpacked_word=f'{key}: {aim_params[key]:.2f} '
            except:
                unpacked_word=f'\n {key}: {aim_params[key]}'
            title=title+unpacked_word
            
        title=title+f' beta: {beta}'
        return title

    def __len__(self):
        return self.n_samples