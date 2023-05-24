from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR

class Schedular_Interface:
    def __init__(self,args,optimizer) -> None:

        lr_schedular_dict = dict(method=str(args.lr_schedular[0]),
                                step_size=int(args.lr_schedular[1]),
                                gamma=float(args.lr_schedular[2])
                                )
        long_text=f"""
lr_schedular_dict:

{lr_schedular_dict=}
        """
        print(long_text)


        if lr_schedular_dict['method'] == 'CosineLR':
            
            self.scheduler =  CosineAnnealingLR(optimizer,T_max = lr_schedular_dict['step_size'])

        elif lr_schedular_dict['method'] == 'StepLR':

            self.scheduler = StepLR(optimizer, step_size=lr_schedular_dict['step_size'], gamma=lr_schedular_dict['gamma'])

        else:
            raise ValueError('lr_schedular_dict[method] wrong')
