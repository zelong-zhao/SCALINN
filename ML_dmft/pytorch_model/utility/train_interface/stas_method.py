import math
import numpy as np
from ..loss_function import matsubara_weighted_err_numpy


class Welford(object):
    def __init__(self,beta=10):
        self.k = 0
        self.M = 0
        self.S = 0
        self.AE = 0
        self.SE = 0
        self.Matsu = 0
        self.beta = beta
        
    def __update(self,x,y):
        if len(x.shape) == 2:
            x=np.squeeze(x,-1)
            y=np.squeeze(y,-1)

        assert len(y.shape) == 1

        if x is None or y is None:
            return
        self.k += 1

        diff = x-y
        mean_diff = np.mean(np.square(diff))
    
        newM = self.M + (mean_diff - self.M)*1./self.k
        newS = self.S + (mean_diff - self.M)*(mean_diff - newM)
        self.M, self.S = newM, newS
        
        self.AE += np.abs(diff).mean()
        self.SE += np.square(diff).mean()
        self.Matsu += matsubara_weighted_err_numpy(y,x,self.beta,1,False)

    def __consume(self,true,pred):
        lst = iter(zip(true,pred))
        for x,y in lst:
            self.__update(x,y)
    
    def __call__(self,true,pred):

        if hasattr(true,"__iter__"):
            self.__consume(true,pred)
        else:
            self.__update(true,pred)

    @property
    def MAE(self):
        return self.AE/self.k
    @property
    def MSE(self):
        return self.SE/self.k
    @property
    def MATSU(self):
        return self.Matsu/self.k
    @property
    def mean(self):
        return self.M
    @property
    def meanfull(self):
        return self.mean, self.std/math.sqrt(self.k)
    @property
    def std(self):
        if self.k==1:
            return 0
        return math.sqrt(self.S/(self.k-1))
    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)
    

class Welford_Online_Stas(Welford):
    def __init__(self,beta:float=10,inverse_transform=None) -> None:
        super().__init__(beta=beta) 

        self.inverse_transform = inverse_transform

    def __call__(self,true:np.ndarray,pred:np.ndarray):
        true,pred = Welford_Online_Stas.three_dim_inver(true,pred,self.inverse_transform)
        super().__call__(true,pred)
    
    @staticmethod
    def three_dim_inver(true:np.ndarray,pred:np.ndarray,inverse_transform=None):
        if len(true.shape) == 3:
            _,t,d = true.shape
            assert d==1
            true,pred=true.squeeze(-1),pred.squeeze(-1)

        elif len(true.shape) == 2:
            _,t = true.shape
            assert t != 1

        if np.isnan(pred).any():
            raise ValueError('Model explod')
        if inverse_transform is not None:
            for idx,(item1,item2) in enumerate(zip(true,pred)):
                true[idx] = inverse_transform(item1)
                pred[idx] = inverse_transform(item2)
        return true,pred
    