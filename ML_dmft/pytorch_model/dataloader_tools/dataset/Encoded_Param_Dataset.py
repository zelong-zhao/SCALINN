import warnings
import numpy as np
import torch
from copy import deepcopy
from functools import partial
from typing import Callable
from ML_dmft.database.dataset import Aim_Dataset_special
from ML_dmft.utility.tools import get_bool_env_var


def seq_dict(index,aim_params,sequence,src,trg,trg_y):
        out_dict={  'index':np.array(index).reshape(1,1),
                    'aim_params':aim_params,
                    'sequence':sequence,
                    'src':src,
                    'trg':trg,
                    'trg_y':trg_y,
                 }
        return out_dict

class Encoded_Param_Dataset():
    def __init__(self, root:str, db:str, solver:str, basis:str, solver_j:list,transform_dict:dict):
        super().__init__()
        r"""
        output
        sample,dict
        train,target=sample
        train=aim_params,feature_G
        """
        self.transform = transform_dict

        self.DEBUG = get_bool_env_var('ML_dmft_Dataset_DEBUG')

        print('\n'+50*'#')
        print('Encoded_Aim_Transformer_Dataset')
        print(f"{solver}\n{solver_j=}")
        self.num_src_solvers = len(solver_j)
        print(f"{self.num_src_solvers=}")

        self.G_Zscore: Callable[[np.ndarray],np.ndarray] = None
        self.ToTensor: Callable[[np.ndarray],torch.Tensor] = None

        self.beta_Zscore,self.mu_Zscore, = None,None
        self.U_Zscore,self.eps_Zscore = None,None
        self.E_k_Zscore,self.V_k_Zscore = None,None
        self.tail_len = None
        self.hyb_drop_solver_idx = None

        for item in self.transform:
            if not hasattr(self,item): 
                raise AssertionError(f'{item} transform is not desired.')
            setattr(self, item, self.transform[item])
        

        dataset_tgt = Aim_Dataset_special(root,db,solver,load_aim_csv_cpu=False)
        self.length = len(dataset_tgt)

        self.aim_params = lambda index:dataset_tgt(index,'aim_params')
        self.imp_params = lambda index:dataset_tgt(index,'imp_params')
        self.hyb_params = lambda index:dataset_tgt(index,'hyb_params')
        self.beta = lambda index:dataset_tgt(index,'beta')
        self.iw_obj = lambda index:dataset_tgt(index, basis)


        self.src_dn_name_list = []
        for idx,src_solver in enumerate(solver_j):
            _dataset_src =  Aim_Dataset_special(root,db,src_solver,load_aim_csv_cpu=False)
            _src_dataset_name = f'src_dataset_{idx}'
            self.src_dn_name_list.append(_src_dataset_name)
            print(f"creating {_src_dataset_name=} for {src_solver=}")
            if hasattr(self,_src_dataset_name):
                raise AssertionError('_src_dataset_name is asserted before')
            setattr(self,_src_dataset_name,deepcopy(_dataset_src))

        if len(self.src_dn_name_list)==1: assert self.hyb_drop_solver_idx==None
        if self.hyb_drop_solver_idx is not None:
            assert self.hyb_drop_solver_idx <= len(self.src_dn_name_list)
        
        self.src_iw_obj = self.__stacked_src_iw_obj(basis)
        self.src_hyb_params_list=self.__src_hyb_obj()
        self.__src_hyb_list_dim_check(self.src_hyb_params_list)

        print(f"{dataset_tgt.possible_keys=}")
        print(50*'#')

    def __src_hyb_list_dim_check(self,src_hyb_params_list):
        dim_prev = None
        arr_list = []
        for idx,_src_hyb_param in enumerate(src_hyb_params_list):
            src_onsite,src_hopping = _src_hyb_param(index=0)
            _src_hyb_params = np.array([np.array([item1,item2]).squeeze(-1) for item1,item2 in zip(src_onsite,src_hopping)])
            if dim_prev is None:
                dim_prev = _src_hyb_params.shape   
            assert dim_prev == _src_hyb_params.shape,'dimention of hyb should be same. Check Hyb time-step between different solver.'
            arr_list.append(_src_hyb_params)
        for i, arr1 in enumerate(arr_list):
            for j, arr2 in enumerate(arr_list):
                if i != j and np.array_equal(arr1, arr2):
                    warnings.warn('In Encoded Param Dataset: at least two src hyb are same')

    def __src_hyb_obj(self):
        __src_hyb_obj_list = []
        for idx,src_db in enumerate(self.src_dn_name_list):
            if idx != self.hyb_drop_solver_idx:
                __src_hyb_obj_list.append(partial(getattr(self,src_db),key='hyb_params'))
            else:
                print(f"!!!{self.hyb_drop_solver_idx=} so {getattr(self,src_db).solver} is not included hyb")
        return __src_hyb_obj_list

    def __stacked_src_iw_obj(self,basis):
        _src_iw_obj_list = [partial(getattr(self,src_db),key=basis) for src_db in self.src_dn_name_list]
        return lambda index: np.concatenate([f(index) for f in _src_iw_obj_list],axis=1)

    def __getitem__(self, index):
        r"""
        return
        output
        sample,dict
        train,target=sample
        train=aim_params,right_shifted_Sigma
        """

        aim_params = self.aim_params(index)


        onsite,hopping = self.hyb_params(index)
        if self.E_k_Zscore is not None and self.V_k_Zscore is not None:
            onsite,hopping = self.E_k_Zscore(onsite),self.V_k_Zscore(hopping)
        hyb_params = np.array([np.array([item1,item2]).squeeze(-1) for item1,item2 in zip(onsite,hopping)])
        hyb_params = np.expand_dims(hyb_params,axis=0)
        if self.DEBUG: print(f"in dataset  {hyb_params.shape=}")
        assert hyb_params.shape[0] == 1
        assert hyb_params.shape[2] == 2
        # hyb_params.size() == 1,#_bath,2

        beta = self.beta(index)
        if self.beta_Zscore is not None:
            self.beta_Zscore(beta)

        imp_params = self.imp_params(index).reshape(1,2)

        U,eps = imp_params[0,0],imp_params[0,1]
        
        mu = -eps-U/2. #mu find by shifting of eps.
        if self.mu_Zscore is not None:
            mu=self.mu_Zscore(mu)

        if self.U_Zscore is not None and self.eps_Zscore is not None:
            imp_params[0,0],imp_params[0,1]=self.U_Zscore(imp_params[0,0]),self.eps_Zscore(imp_params[0,1])

        glob_params = np.array([mu.squeeze(-1),beta]).reshape(2,)
        glob_params,imp_params,hyb_params = self.ToTensor(glob_params),self.ToTensor(imp_params),self.ToTensor(hyb_params)

        src_hyb_params = []
        for _src_hyb_param in self.src_hyb_params_list:
            src_onsite,src_hopping = _src_hyb_param(index)
            if self.DEBUG: print(f"in dataset {onsite.shape=} {hopping.shape=}")
            if self.E_k_Zscore is not None and self.V_k_Zscore is not None:
                src_onsite,src_hopping = self.E_k_Zscore(src_onsite),self.V_k_Zscore(src_hopping)
            _src_hyb_params = np.array([np.array([item1,item2]).squeeze(-1) for item1,item2 in zip(src_onsite,src_hopping)])
            if self.DEBUG: print(f"in dataset  {_src_hyb_params.shape=}")
            src_hyb_params.append(_src_hyb_params)
        src_hyb_params = np.array(src_hyb_params)
        if self.DEBUG: print(f"in dataset before ToTensor:{src_hyb_params.shape=}")
        src_hyb_params = self.ToTensor(src_hyb_params)
        if self.hyb_drop_solver_idx is None:
            assert src_hyb_params.size()[0]== self.num_src_solvers, 'hyb_params.shape = num_src_solvers,hyb-length,2'
        else:
            assert src_hyb_params.size()[0]== self.num_src_solvers-1, 'hyb_params.shape = num_src_solvers,hyb-length,2'


        if self.DEBUG: print(f"in dataset {index=}")
        if self.DEBUG: print(f"in dataset {beta=}")
        if self.DEBUG: print(f"in dataset {aim_params=} {aim_params.shape=}")
        if self.DEBUG: print(f"in dataset {imp_params=} {imp_params.shape=}")
        if self.DEBUG: print(f"in dataset {onsite=}")
        if self.DEBUG: print(f"in dataset {hopping=}")
        if self.DEBUG: print(f"in dataset {hyb_params=} {hyb_params.shape=}")
        if self.DEBUG: print(f"in dataset {glob_params=} {glob_params.shape=}")

        tgt_whole_sequence = self.iw_obj(index) #target sequence
        src_whole_sequence = self.src_iw_obj(index) #defined by solver_j, Whole length of solver is defined.

        if self.DEBUG: print(f"before transform {tgt_whole_sequence.T=} {src_whole_sequence.T=}")
        if  self.G_Zscore is not None:
            tgt_whole_sequence = self.G_Zscore(tgt_whole_sequence)
            src_whole_sequence = self.G_Zscore(src_whole_sequence)
        if self.DEBUG: print(f"after transform {tgt_whole_sequence.T=} {src_whole_sequence.T=}")
        
        tgt_whole_sequence = np.flip(tgt_whole_sequence)
        src_whole_sequence = np.flip(src_whole_sequence)

        assert src_whole_sequence.shape[1] == self.num_src_solvers

        rf_tgt = tgt_whole_sequence[0:len(tgt_whole_sequence)-1]
        tgt = tgt_whole_sequence[1:]

        # rf_tgt,tgt = tgt_whole_sequence[0:len(tgt_whole_sequence)-1],tgt_whole_sequence[1:len(tgt_whole_sequence)]
        assert self.tail_len <= len(rf_tgt)
        tail = tgt[0:self.tail_len]

        if self.DEBUG: print(f"{np.flip(tgt_whole_sequence).T=}")
        if self.DEBUG: print(f"{np.flip(src_whole_sequence).T=} {np.flip(src_whole_sequence).shape=}")
        if self.DEBUG: print(f"{np.flip(rf_tgt).T=} {np.flip(rf_tgt).shape=}")
        if self.DEBUG: print(f"{np.flip(tgt).T=} {np.flip(tgt).shape=}")
        
        #Sequence will be inverse transformed
        # index,aim_params,sequence,src,trg,trg_y
        out_dict = seq_dict(index=index,
                        aim_params=aim_params,
                        sequence=np.copy(tgt_whole_sequence),
                        src=np.copy(src_whole_sequence[:,0][:,np.newaxis]),
                        trg=np.copy(rf_tgt),
                        trg_y=np.copy(tgt))

        src_whole_sequence = self.ToTensor(src_whole_sequence)
        rf_tgt = self.ToTensor(rf_tgt)
        tail = self.ToTensor(tail)
        tgt = self.ToTensor(tgt)

        params=(glob_params,imp_params,hyb_params,src_hyb_params)
        seq = (src_whole_sequence,tail,rf_tgt)
        out_sample=(params,seq),tgt
        
        return out_sample,out_dict
    
    def __len__(self):
        return self.length