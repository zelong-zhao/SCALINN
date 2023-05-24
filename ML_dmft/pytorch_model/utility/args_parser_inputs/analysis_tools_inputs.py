import argparse,sys
from .datasets_args import dataset_args
from .dataset_transform_args import dataset_transform_args


def read_args():
    parser=argparse.ArgumentParser(description='Tools for analysing checkpoints')
    
    parser.add_argument('--model_hyper_params',nargs='+',help='model hyper param',type=str,default=None)

    parser.add_argument('--plot_one_file', action='store_true', default=False,
                    help='(action) --plot_one_file (False)')
                    
    parser.add_argument('--file', type=str, default='andT_model.pth',
                    help='file to load',required='--plot_one_file ' in sys.argv)

    parser.add_argument('--output_file', type=str, default='out.csv',
                    help='file to save default (out.csv)')

    # checkpoint dir
    parser.add_argument('--plot_checkpoints', action='store_true', default=False,
                help='(action) --plot_checkpoints (False)')
                
    parser.add_argument('--save_to_db', action='store_true', default=False,
                help='(action) save ML predicted to database/db/ dir defualt(False)')

    parser.add_argument('--db_name', type=str, default='AndT',help='dump to predicted G to directory/db/db_name/0 default(PRW)',required='--save_to_db' in sys.argv)

    parser.add_argument('--directory','-d', type=str, default='./model_parameter/train/',help='path to find --file-rule dafault(./model_parameter/train/)')
    parser.add_argument('--file-rule', type=str, default='model_epoch_*.pth',help='default(model_epoch_*.pth)')

    dataset_args(parser)

    dataset_transform_args(parser)

    parser.add_argument('--model_type', type=str, metavar='n',
                        help='type of model to load',required=True)
    parser.add_argument('--dataset_type', type=str, metavar='n',
                        help='type of dataset_type to load',required=True)

    parser.add_argument('--num-samples', type=int, default=10, metavar='N',
                        help='num of aim samples to set default(10) -1 for all datasets')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--num-worker', type=int, default=1, metavar='N',
                        help='num of worker for loading the dataloader')


    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass') 


    args=parser.parse_args()
    
    return args

