from ..train_tools.tool import Nullable_Str
from argparse import ArgumentParser

def dataset_args(parser:ArgumentParser):
    parser.add_argument('--root-database', type=str, default='/home/zz/drive1/ML_DATABASE/', metavar='f',
                    help='root to the dir of database')
    parser.add_argument('--db', type=str, default='DEV1', metavar='f',
                help='name of the database default(DEV1)')
    parser.add_argument('--solver', type=str, default='ED_KCL_4',
                        help='solver to learn. root database')     
    parser.add_argument('--basis', type=str, default='G_tau', metavar='f',
                help='basis to study G_tau,G_iw,G_l G_tau_64,G_tau_128 default(G_tau)')
    parser.add_argument('--dataset_method', type=Nullable_Str(str),default=None,
                    help='dataset method aviliable number (0,1,2). default(None)')
    # parser.add_argument('--solver_j', type=Nullable_Str(str), default=None,
    #                     help='solver_j to learn only use when method=2. (G-IPT_G) ')   
    parser.add_argument('--solver_j',nargs='+',type=Nullable_Str(str), default=None,
                        help='solver_j learn')   
    parser.add_argument('--basis_j', type=Nullable_Str(str), default=None, metavar='f',
                help='basis_j to learn only use when method=1. (G-G0)')
    parser.add_argument('--basis_feature', type=Nullable_Str(str), default=None, metavar='f',
                help='basis_feature is used for ')
    return