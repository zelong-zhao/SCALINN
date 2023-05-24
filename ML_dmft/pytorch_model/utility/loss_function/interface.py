from torch import nn
from .matsubara_weighted_mse import Matsubara_Weighted_Loss

class Loss_Function_Interface:
    def __init__(self,args) -> None:
        #Define Loss functions 
        if args.criteria == 'Matsu':
            Loss_dict=dict(beta=args.Loss_args[0],
                        alpha=args.Loss_args[1]
                        )
            long_text=f"""
    Matsubara_MSE:
    {Loss_dict=}
    """
            print(long_text)
            self.criterion = Matsubara_Weighted_Loss(**Loss_dict)
        elif args.criteria == 'MSE':
            self.criterion = nn.MSELoss()
        else:
            print(args.criteria)
            raise TypeError('no such options')

    @staticmethod    
    def methods()->list:
        return ['Matsu','MSE']