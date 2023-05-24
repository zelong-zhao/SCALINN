from ML_dmft.triqs_interface.v2 import get_gf_xy
import ML_dmft.numpy_GF.GF as GF
from triqs.plot.mpl_interface import oplot,plt
import triqs.gf
from scipy.optimize import minimize
from functools import partial
import warnings,sys
import numpy as np

def plot(iOmega_n,Semi_data,fitted_hyb):
    plt.figure()
    plt.plot(iOmega_n.imag,Semi_data.real,marker='.',label='Bethe real')
    plt.plot(iOmega_n.imag,Semi_data.imag,marker='.',label='Bethe imag')
    plt.plot(iOmega_n.imag,fitted_hyb.real,linestyle='',marker='x',label='fitted real')
    plt.plot(iOmega_n.imag,fitted_hyb.imag,linestyle='',marker='x',label='fitted imag')

    plt.xlim(0,10)
    plt.legend()
    plt.savefig('test.png')

def plot_DOS(fitted_hyb,n_iw=1024,beta=10):
    from triqs.plot.mpl_interface import oplot,plt

    Semi=triqs.gf.GfImFreq(indices=[0], beta=beta, n_points=n_iw)
    Semi<<triqs.gf.SemiCircular(2)

    fitted=Semi.copy()
    fitted.data[:,:,0]=GF.iw_obj_flip(fitted_hyb)

    Semi_real = triqs.gf.GfReFreq(indices = [1], window = (-5.0,5.0))
    Semi_real.set_from_pade(Semi, 100, 0.01)

    fitted_real=triqs.gf.GfReFreq(indices = [1], window = (-5.0,5.0))
    fitted_real.set_from_pade(fitted, 100, 0.01)
    plt.figure()
    oplot(-Semi_real.imag/np.pi,label='semi-circular')
    oplot(-fitted_real.imag/np.pi,label='fitted')

    plt.xlim(-5,5)
    plt.savefig('test2.png')


def distance_func_one_dim(concatenate_input,delta2,n_iw,beta,omega_c):
    dim=int(len(concatenate_input)/2)
    onsite,hopping=concatenate_input[0:dim],concatenate_input[dim:]
    Omega_n=GF.matsubara_freq(beta=beta,n_w=n_iw)
    delta1=GF.hyb_np(onsite,hopping,n_iw,beta)
    return GF.iwn_distance_func(delta1,delta2,Omega_n,omega_c)


def distance_func_one_dim_symmetry(concatenate_input,delta2,n_iw,beta,omega_c):
    dim=int(len(concatenate_input)/2)
    onsite,hopping=concatenate_input[0:dim],concatenate_input[dim:]

    onsite = np.concatenate((onsite,onsite*-1),axis=None)
    hopping = np.concatenate((hopping,hopping*-1),axis=None)

    Omega_n=GF.matsubara_freq(beta=beta,n_w=n_iw)
    delta1=GF.hyb_np(onsite,hopping,n_iw,beta)
    return GF.iwn_distance_func(delta1,delta2,Omega_n,omega_c)


def cal_err_semi_circular_DOS_symmetry(num_imp:int,beta:float,n_iw:int,
                                    W:float=1.,omega_c:int=100,
                                    err_tol:float=1e-5,
                                    max_fitting_err:float=2.81,
                                    delta_from_zero:float=1e-2,
                                    V_bound:list=[-3,3],
                                    E_bound:list=[-3,3],
                                    fit_function:np.ndarray=None,
                                    ):
    """
    cal err for semi circular DOS
    args:
    -----
        num_imp : number of bath site
        beta    : temperature
        n_iw    : number matsubara frequency to gen hyb
        omega_c : matsubara frequency cutoff.
        err_tol : err tolerence of fitting hyb
        delta_from_zero : make V and E shift from zero.
        V_bound : hopping term boundary of fitting
        E_bound : on site term bounday of fitting
    """

    err_tol = err_tol
    if omega_c > n_iw:
        omega_c = n_iw
    
    max_iter = 100

    Omega_n=GF.matsubara_freq(beta=beta,n_w=n_iw)

    if fit_function is None:
        fit_function=GF.greenF(Omega_n=Omega_n,D=W)

    assert num_imp%2 == 0 ,'num_imp has to be even'
    num_imp_half = int(num_imp/2)

    err = partial(distance_func_one_dim_symmetry,delta2=fit_function,n_iw=n_iw,beta=beta,omega_c=omega_c)
    #V_bound 0~3
    assert V_bound[0]==-1*V_bound[1],'symmetry asked to fit semi-o'
    assert E_bound[0]==-1*E_bound[1],'symmetry asked to fit semi-o'
    assert V_bound[0] < V_bound[1], '0 idex corresponding to low limit'
    assert E_bound[0] < E_bound[1], '0 idex corresponding to low limit'
    print(f"V bound {V_bound}")
    print(f"E bound {E_bound}")

    Fit_Success = True
    Keep_run,i=True,0
    while Keep_run:
        V_guess = np.random.uniform(low=delta_from_zero,high=V_bound[1],size=(num_imp_half,))
        E_guess = np.random.uniform(low=delta_from_zero,high=E_bound[1],size=(num_imp_half,))

        V_p = np.concatenate((V_guess,V_guess*-1),axis=None)
        E_p = np.concatenate((E_guess,E_guess*-1),axis=None)

        # initial_guess = np.concatenate([V_p,E_p])

        initial_guess=np.concatenate([E_guess,V_guess,])

        print(20*'#',num_imp,20*'#')
        print(f"{num_imp=}")
        print(f"{beta=}")
        print('V_p: ', V_p)
        print('E_p: ', E_p)
        print(f"{err(initial_guess)=}")
###########################################################################
# minimize

        options=dict(disp=True,maxiter=int(100))
        res=minimize(err,x0=initial_guess,method='CG',options=options,tol=err_tol)
        e_list=res.x[:num_imp_half]
        V_list=np.abs(res.x[num_imp_half:2*num_imp_half])
        Keep_run=False

        for element in e_list:
            if element < E_bound[0] or element > E_bound[1] or np.abs(element)<delta_from_zero:
                print(element,'I am not in E_bound and recalculate')
                Keep_run=True
        for element in V_list:
            if element < V_bound[0] or element > V_bound[1] or np.abs(element)<delta_from_zero:
                print(element,'I am not in V_bound and recalculate')
                Keep_run=True
        if err_tol < err(res.x):
            print(element,'Err too larger refit')
            Keep_run=True
            i+=1

        if i > max_iter:
            i=0
            err_tol=err_tol*10
            print(f'err_tol adaptive new value {err_tol=}')
            if err_tol > max_fitting_err:
                Keep_run=True
                print(f"{err(res.x)=} {err_tol=}")
                warnings.warn('Fitting failed')
                Fit_Success = False
        
    
    e_list=res.x[:num_imp_half]
    e_list=np.concatenate((e_list,e_list*-1),axis=None)

    V_list=res.x[num_imp_half:2*num_imp_half]
    V_list=np.concatenate((V_list,V_list*-1),axis=None)
    
    fitted_erro=err(res.x)
    print('Fit success: ',res.message)
    print('Number of iterations: ',res.nit)
    print('Final error from fit: ',fitted_erro)
    print('Final parameters: ',res.x)
    print('E_p: ',e_list)
    print('V_p: ',V_list)
    print(20*'#',num_imp,20*'#')

    # fitted_hyb=GF.hyb_np(e_list,V_list,n_iw=n_iw,beta=beta)

    # plot(1j*Omega_n,bethe_iw,fitted_hyb)
    # plot_DOS(fitted_hyb,n_iw=1024,beta=10)
    return num_imp,fitted_erro,Fit_Success,e_list,V_list
    

def cal_err_semi_circular_DOS(num_imp:int,beta:float,n_iw:int,W:float=1.,omega_c:int=100):
    """
    cal err for semi circular DOS
    """

    Omega_n=GF.matsubara_freq(beta=beta,n_w=n_iw)
    bethe_iw = GF.greenF(Omega_n=Omega_n,D=W)

    err = partial(distance_func_one_dim,delta2=bethe_iw,n_iw=n_iw,beta=beta,omega_c=omega_c)

    #V_bound 0~3
    V_bound=[0,3]
    E_bound=[-3,3]

    Keep_run=True
    while Keep_run:
        V_guess=np.random.rand(num_imp)*2
        E_guess=(np.random.rand(num_imp)*2-1)*2
        initial_guess=np.concatenate([E_guess,V_guess])

        print(20*'#',num_imp,20*'#')
        print(f"{num_imp=}")
        print(f"{beta=}")
        print('E_p: ',initial_guess[:num_imp])
        print('V_p: ',initial_guess[num_imp:2*num_imp])
        print(f"{err(initial_guess)=}")

###########################################################################
# minimize

        options=dict(disp=True,maxiter=int(1e4))
        res=minimize(err,x0=initial_guess,method='CG',options=options)
        e_list=res.x[:num_imp]
        V_list=np.abs(res.x[num_imp:2*num_imp])
        Keep_run=False
        for element in e_list:
            if element < E_bound[0] or element > E_bound[1]:
                print(element,'I am not in E_bound and recalculate')
                Keep_run=True
        for element in V_list:
            if element < V_bound[0] or element > V_bound[1]:
                print(element,'I am not in V_bound and recalculate')
                Keep_run=True
    
    e_list=res.x[:num_imp]
    V_list=np.abs(res.x[num_imp:2*num_imp])
    fitted_erro=err(res.x)[0]

    print('Fit success: ',res.message)
    print('Number of iterations: ',res.nit)
    print('Final error from fit: ',fitted_erro)
    print('Final parameters: ',res.x)
    print('E_p: ',e_list)
    print('V_p: ',V_list)
    print(20*'#',num_imp,20*'#')

    # fitted_hyb=GF.hyb_np(e_list,V_list,n_iw=n_iw,beta=beta)

    # plot(1j*Omega_n,bethe_iw,fitted_hyb)
    # plot_DOS(fitted_hyb,n_iw=1024,beta=10)
    return num_imp,fitted_erro,e_list,V_list

def fit_semi_circular_one_shot():
    imp_list=np.arange(1,10)
    out_dict=[]
    for i in imp_list:
        num_imp,err,E_p,V_p=cal_err_semi_circular_DOS(i)
        out_array=[num_imp,err,*E_p,*V_p]
        print(f'{num_imp=} {err=}')
        out_dict.append(out_array)
    out_dict=np.array(out_dict)

    header=['#imp','err','Ep','Vp']
    import csv
    with open('fitted.csv','w') as f:
        writer=csv.writer(f)
        writer.writerow(header)
        writer.writerows(out_dict)

if __name__ == '__main__':
    # cal_err_semi_circular_DOS(3)
    cal_err_semi_circular_DOS_symmetry(num_imp=4,beta=10,n_iw=64,omega_c=32)
    # fit_semi_circular_one_shot()