import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use(plt.style.available[-2])
from ML_dmft.database.dataset import AIM_dataset_meshG

def best_and_worst_index(Predicted_G):
    for dict in Predicted_G:
        dict['delta G prediction']
    Dict_2_list = lambda key : list(dict[key] for dict in Predicted_G)

    delta_G_target_list=np.array(Dict_2_list('delta G target'))
    delta_G_prediction_list=np.array(Dict_2_list('delta G prediction'))
    print(f"{delta_G_target_list.shape=}")
    err=np.sqrt(np.average((delta_G_target_list-delta_G_prediction_list)**2,axis=1))
    err_idx=np.argsort(err)
    return err_idx

def evaluate_model_sequencial_iw(args,Predicted_G_dict):

    print(Predicted_G_dict[0].keys())
    index=list(dict['index'] for dict in Predicted_G_dict)

    Dict_2_list = lambda key : np.array(list(dict[key] for dict in Predicted_G_dict))
    
    ED_KCL = Dict_2_list('sequence')
    seq_src = Dict_2_list('src') # should be right shifted sigma_iw
    seq_rf_tgt = Dict_2_list('trg')
    seq_trg = Dict_2_list('trg_y')

    G_pred=Dict_2_list('delta G prediction')
    target=Dict_2_list('delta G target')

    err_idx=best_and_worst_index(Predicted_G_dict)

    plot_best_and_worst_iw(args,
                                err_idx,
                                index,
                                seq_src,
                                seq_rf_tgt,
                                seq_trg,
                                G_pred)


def plot_best_and_worst_iw(args,
                                err_idx,index,
                                seq_src,seq_rf_tgt,seq_trg,
                                G_pred):
    print('error correction method find best and worst case')
    import seaborn as sns 
    color=sns.color_palette("tab10")
    fig, ax = plt.subplots(1,2,figsize=(8,3),dpi=300)
    for i,(idx,label) in enumerate(zip([err_idx[0],err_idx[-1]],['best','worst'])):
        aim_index=index[idx]
        print(f"{aim_index=} {idx} in loop")
        print(f"{label=}")
        print(f"{args.basis=}")
        assert aim_index==idx, 'aim index are different!'
        assert 'sigma_iw' or 'g_iw' in args.basis.lower(), 'must be iw'
 
        IPT=AIM_dataset_meshG(root=args.root_database,
                db=args.db,
                solver='IPT')

        ED=AIM_dataset_meshG(root=args.root_database,
                db=args.db,
                solver=args.solver) 

        (mesh,ED_G),(title,mesh_type)=ED(aim_index,args.basis)

        (_,IPT_G),(_,_)=IPT(aim_index,args.basis)

        assert np.iscomplexobj(ED_G), 'type error should be complex'

        seq_src_dim = seq_src.shape[1]
        seq_trg_dim = seq_rf_tgt.shape[1]
        seq_trg_y_dim = seq_trg.shape[1]

        mesh_src = mesh[-seq_src_dim:]
        print(f"in plot worst and best {mesh_src.shape=} {seq_src.shape=}")
        mesh_trg = mesh[1:seq_trg_dim+1]
        print(f"in plot worst and best {mesh_trg.shape=} {seq_rf_tgt.shape=}")
        mesh_trg_y = mesh[:seq_trg_y_dim]
        print(f"in plot worst and best {mesh_trg_y.shape=} {seq_trg.shape=}")

        src=seq_src[idx][:,np.newaxis]
        trg=seq_rf_tgt[idx][:,np.newaxis]
        trg_y=seq_trg[idx][:,np.newaxis]

        Gpred=G_pred[idx][:,np.newaxis]
        
        print(f"{mesh.shape} {len(Gpred)=}")
        mesh_pred = mesh[0:len(Gpred)]


        ax[i].plot(mesh,ED_G.imag,label='ED',color=color[0],marker='o',markersize=2)
        ax[i].plot(mesh_src,src,label='SRC',color=color[1])
        ax[i].plot(mesh_trg,trg,label='RF TRG',color=color[3])
        ax[i].plot(mesh_trg_y,trg_y,label='TRG',color=color[4])

        ax[i].plot(mesh,IPT_G.imag,label='IPT Approximation',color=color[2],marker=',',markersize=2)
        # ax[i].plot(mesh,IPT_loaded[idx,:],label='IPT Approximation',color=color[2],marker=',',markersize=2)

        ax[i].plot(mesh_pred,Gpred,label='%s AndT'%label,ls='--',color=color[4],marker='.',markersize=2)

        if 'sigma_iw' in args.basis.lower(): ax[i].set_ylim(-8,0)
        if 'g_iw' in args.basis.lower(): ax[i].set_ylim(-1.,0)
        ax[i].legend()
        ax[i].set_xlabel(f'{mesh_type}')
        ax[i].set_ylabel(f'{args.basis}')
        ax[i].set_title(title,fontsize=10)
    fig.tight_layout()
    fig.savefig(f'ML_predicted_{args.basis}_best_worst.png')


def evaluate_model(args,Predicted_G_dict):
    evaluate_model_sequencial_iw(args,Predicted_G_dict)