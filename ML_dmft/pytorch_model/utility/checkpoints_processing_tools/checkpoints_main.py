import argparse
import sys
from .checkpoints_tools import plot_loss_multiple
from .checkpoints_analysis.avg_data_utils import avg_data_dir

def read_args():
    parser = argparse.ArgumentParser(description='Tools for analysing checkpoints')


    parser.add_argument('--plot_mutiple_dir_loss','--plot_current_dir_loss', action='store_true', default=False,
                help='plot_current_dir_loss(False) prefix_list required')
    
    parser.add_argument('--avg_data_dir','-average_data_dir', action='store_true', default=False,
            help='avg_data_dir three RUNS')

    parser.add_argument('--directory','-d', type=str, default='./model_parameter/train/',help='path to find --file-rule dafault(./model_parameter/train/)')
    parser.add_argument('--file-rule', type=str, default='model_epoch_*.pth',help='default(model_epoch_*.pth)')
    parser.add_argument('--title', type=str, default='PRW',help='title to be printed default(PRW)')

    parser.add_argument('--plot_logscale','-log', action='store_true')
    parser.add_argument('--no-plot_logscale','-no-log', dest='plot_logscale', action='store_false')
    parser.set_defaults(plot_logscale=True)


    parser.add_argument('--prefix_list', nargs='+', help='prefix to directory 1 2 3 4 5',type=str,required='--plot_mutiple_dir_loss' in sys.argv,default=['./'])
    parser.add_argument('--legend_list', nargs='+', help='legend to prefix',type=str,default=None)
    parser.add_argument('--solver_list', nargs='+', help='solver name to prefix',type=str,required='--plot_mit' in sys.argv)


    parser.add_argument('--use_pth_first', action='store_true', default=False,
                help='use checkpoint other than csv in case needed')
    
    parser.add_argument('--update_csv', action='store_true', default=False,
                help='update checkpoints csv to train_loss')

    args = parser.parse_args()

    return args


def main():
    args=read_args()

    if args.plot_mutiple_dir_loss:
        plot_loss_multiple(args)

    if args.avg_data_dir:
        avg_data_dir()