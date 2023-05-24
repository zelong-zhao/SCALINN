import torch
import numpy as np

        

class Matsubara_Weighted_Loss():
    def __init__(self,beta:float=10.,alpha:float=1.0) -> None:
        super(Matsubara_Weighted_Loss,self).__init__()
        self.beta=beta
        self.alpha=alpha
        max_lenth = 64
        inverse_matsu_freq = 1+1/((2*torch.arange(max_lenth)+1)*torch.pi/self.beta)**alpha
        # inverse_matsu_freq[1:7] = inverse_matsu_freq[0]
        self.inverse_matsu_freq = inverse_matsu_freq

        print(f"{self.inverse_matsu_freq=}")

    def forward(self,output:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        return self.__call__(output,target)

    def __call__(self,output:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        r"""
        matsubara frequency assume between [0,len(target)]
        """
        device = output.device
        _,t,_ = target.shape
        # index_list = torch.arange(t).to(device)
        # matsu_freq = (((2*index_list+1)*torch.pi / self.beta)**self.alpha).view(t,1)
        # inverse_matsu_freq = 1+1/matsu_freq
        inverse_matsu_freq = self.inverse_matsu_freq[:t].view(t,1).to(device)
        
        inverse_matsu_freq = torch.flip(inverse_matsu_freq,[0]) # 1~16 -> 16~1 

        loss=((output - target)**2)*inverse_matsu_freq
        
        loss=torch.mean(torch.sum(loss,[1]))
        # loss=torch.sum(loss)
        # loss=torch.mean(loss) # MSE weighted MSE

        return loss



def matsubara_weighted_loss_numpy(output:np.ndarray, target:np.ndarray,
                                beta:float=10.,alpha:float=1.0,flip:bool=True)->np.ndarray:
    r"""
    Notes
    ------
    matsubara frequency assume between len(target)

    Inputs
    ------
    beta: temperature
    alpha 1/w**alpha |X-Y|**2
    """
    b,t = target.shape
    index_list = np.arange(t)
    matsu_freq = ((2*index_list+1)*np.pi / beta)**alpha

    if flip: matsu_freq = np.flip(matsu_freq) # 1~16 -> 16~1 

    loss=((output - target)**2)/matsu_freq
    # loss=np.mean(loss)
    loss=np.mean(np.sum(loss,axis=1))

    return loss


def matsubara_weighted_err_numpy(output:np.ndarray, target:np.ndarray,
                                beta:float=10.,alpha:float=1.0,flip:bool=True)->np.ndarray:
    r"""
    Notes
    ------
    matsubara frequency assume between len(target)

    Inputs
    ------
    beta: temperature
    alpha 1/w**alpha |X-Y|**2
    """
    len(target.shape)==1 
    t = target.shape[0]
    index_list = np.arange(t)
    matsu_freq = ((2*index_list+1)*np.pi / beta)**alpha

    if flip: matsu_freq = np.flip(matsu_freq) # 1~16 -> 16~1 

    loss=((output - target)**2)/matsu_freq
    # loss=np.mean(loss)
    loss=np.mean(np.sum(loss))

    return loss

