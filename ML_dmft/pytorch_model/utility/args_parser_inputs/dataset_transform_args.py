
from ..train_tools.tool import Nullable_Str

def dataset_transform_args(parser):
    parser.add_argument('--transform_factors',
                        nargs='+', 
                        help='for norm_method 0,1: \
                        (squasher_factor,normalise_factor,squasher_factor_y,normalise_factor_y. dtype(floats))',
                        type=Nullable_Str(str),
                        required=True)

    parser.add_argument('--transform_method','--cal_norm_method',
                    help='different method cause different way of finding transform',type=str,choices=["0","1","1.1",'1.2','1.3','1.4','2.1','2.2'],
                    required=True)