#!/usr/bin/bash
export ML_dmft_DEBUG=0
export ML_dmft_Dataset_DEBUG=0
export ML_dmft_loss_DEBUG=0

title="'Transformer'"
model_name='andT'
model_type='7'
dataset_type='6'

prefix=$(dirname "$(realpath "$0")")"/"
root_database=$(dirname $(dirname "$(realpath "$0")"))"/"
echo "root_database: $root_database"
#database args
db='01_build_db'
beta=100
basis='G_iw_32'
solver='ED_KCL_7'
solver_src='IPT HubbardI'

#train parameter
num_worker=4
batch_size=128
epoch=200
mini_batch_size=8
train_loop=10
optimizer_args="adam 1e-7 0 0"
num_G_eval=-1

#datatransform args
tranform_method="2.2"
model_params_file='./params.yaml'

model_imputs(){
local file=$model_params_file
cat << EOF1 > $file
transform-factors:
    tail_len : 12
    hyb_drop_solver_idx : 0
    g_iw_method : 0 #0 raw 1 norm 2 standard 3 -1norm     4 -1stand
    beta_method : 0 #0 raw 1 norm 2 standard 3 merge_norm 4 merge_stand
    mu_method : 0
    U_method : 0
    eps_method : 0
    E_k_method : 0
    V_k_method : 0
    root_db : $root_database
    db: $db
    solver : $solver
    basis: $basis

model-params:
    seq_pos_method : 3 #0 Cosine 1 Learned_POS 2 Learned_POS cat 3 cat matsubara raw 3.1 normalised 3.2 standardlised
    param_cat_method : 2 # 0:cat2kv 1:cat2pe 2:tst

    tgt_src_mask_off : False
    tgt_mask_lookbackward_len : 5
    src_mask_lookbackward_len : 5
    src_mask_lookforward_len : 5

    dropout_pos_enc : 0.
    depth_input_layer : 1
    droupout_input_layer: 0

    dim_val : 256
    n_encoder_layers : 4
    n_decoder_layers : 4
    n_heads: 32
    dim_feedforward_encoder : 1024
    dim_feedforward_decoder : 1024
    dropout_encoder : 0.
    dropout_decoder : 0.

    depth_enc_param_layer : 0
    dropout_enc_param_layer : 0.
    glob_param_outsize : 0
    imp_param_outsize : 0
    hyb_param_outsize : 0
    src_hyb_param_outsize : 0
    hyb_use_transformer : False

    cat2kv_hyb_pool : 'mean' #['cat','minus','plus','add','+','-','subtraction','mean']
    sep_glob_layer : False
    sep_hyb_layer : False
    sep_imp_layer : False
    enc_dec_input_sep : True 

    predict_unknown_only : False
    decoder_positioanl_norm : False
    tail_from_file : False
EOF1
}
model_imputs

CMD_train="andT_main.py --model_type $model_type
                        --dataset_type $dataset_type
                        --batch-size $batch_size
                        --mini-batch-size $mini_batch_size
                        --train-loop $train_loop
                        --train-test-split 0.8
                        --percent-sample 1.
                        --root-database $root_database
                        --db $db
                        --solver $solver
                        --solver_j $solver_src
                        --basis $basis
                        --transform_method $tranform_method
                        --transform_factors $model_params_file
                        --model-name $model_name
                        --optimizer-args $optimizer_args
                        --lr_schedular StepLR 1000000 0.8
                        --model_hyper_params $model_params_file
                        --ES 20 0.05
                        --Loss-args $beta 1
                        --criteria Matsu
                        --log-interval 50
                        --log-interval-epoch 1
                        --epoch $epoch
                        --num-worker $num_worker
                "       

CMD_analysis="andT_checkpoints_analysis.py --plot_current_dir_loss --title $title"

CMD_benchmark_common="andT_analysis.py
                        --file "$model_name"_model.pth
                        --model_type $model_type
                        --dataset_type $dataset_type
                        --root-database $root_database
                        --batch-size $batch_size
                        --basis $basis
                        --solver $solver
                        --solver_j $solver_src
                        --transform_method $tranform_method
                        --num-samples $num_G_eval
                        --transform_factors $model_params_file
                        --model_hyper_params $model_params_file
                        --log-interval 1000
                        --num-worker $num_worker
                        "

CMD_bench_on_db="$CMD_benchmark_common --db $db --plot_one_file"

####################################################
###############line below to be exe#################
####################################################
set -xe
####################################################
############### trian              #################
####################################################

$CMD_train >> train_out.txt
eval $CMD_analysis
$CMD_bench_on_db #test on trained db