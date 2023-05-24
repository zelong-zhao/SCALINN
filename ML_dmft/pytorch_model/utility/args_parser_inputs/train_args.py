import argparse,sys
from functools import partial
from ..train_tools.tool import Nullable_Str
from ..loss_function import Loss_Function_Interface
from .datasets_args import dataset_args
from .dataset_transform_args import dataset_transform_args

def read_args():
    parser = argparse.ArgumentParser(description='Anderson Transformer')

    parser.add_argument('--model_type', type=str, metavar='n',
                        help='type of model to load',required=True)
    parser.add_argument('--dataset_type', type=str, metavar='n',
                        help='type of dataset_type to load',required=True)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for loading data (default: 64)')
    
    parser.add_argument('--mini-batch-size', type=int, default=1, metavar='N',
                        help='min-batch-size batch size for training (default: 1)')
    parser.add_argument('--train-loop', type=int, default=10, metavar='N',
                        help='how many times should 1 mini-batch-size of data get trained')  

    parser.add_argument('--train-test-split', type=float, default=0.8, metavar='n',
                        help='train and test splite for dataset (default: 0.8)')

    parser.add_argument('--percent-samples', type=float, default=1, metavar='n',
                        help='percentages of samples to use (default: 1)')           

    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')

    parser.add_argument('--model_hyper_params',nargs='+',help='model hyper param',type=str,required=True)


    parser.add_argument('--optimizer-args',nargs='+',type=str,metavar='n',
                        help='(method lr moment weight_decay) dtype(floats))',default=['Adam','1.0e-5','0','0'])

    parser.add_argument('--lr_schedular',nargs=3,type=str,metavar='n',default=['StepLR','100','0.8'],
                        help='(method[CosineLR,StepLR]:str step_size:int gamma:float) )')

    parser.add_argument('--ES',nargs=2,type=str,metavar='n',
                        help='(patience(int) min_delta(float))',default=[1000,1000])

    parser.add_argument('--Loss-args',nargs=2,type=float,metavar='n',
                        help='(beta alpha) dtype(floats))',default=[10,1])
    parser.add_argument('--criteria',type=str,metavar='n',
                        help='train and rank criteria. dtype(str))',default='Matsu',choices=Loss_Function_Interface.methods())
   
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')  

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-interval-epoch', type=int, default=1, metavar='N', help='how many epoch to save the status')  

    parser.add_argument('--RESUME','-R', action='store_true', default=False,
                    help='will read ./model_parameter/train')

    parser.add_argument('--RESUME_FILE','-RR', default=None,type=str,
                    help='file includes net.')

    parser.add_argument('--model-name', type=str, default='PRW',
                    help='Parameter Random Walk) PRW_model.pth')   

    dataset_args(parser)

    parser.add_argument('--num-worker', type=int, default=1, metavar='N',
                    help='num of worker for loading the dataloader')
    parser.add_argument('--seed', type=Nullable_Str(int), default=None, metavar='N',
                    help='ramdon seed to ensure REPRODUCIBILITY') 

    dataset_transform_args(parser)

    # Hyper-parameters 
    args = parser.parse_args()
    return args