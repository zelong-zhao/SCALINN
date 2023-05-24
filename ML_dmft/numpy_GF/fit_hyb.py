from ML_dmft.triqs_interface.v2 import get_gf_xy
import ML_dmft.numpy_GF.GF as GF
from triqs.plot.mpl_interface import oplot,plt
import warnings
from scipy.optimize import minimize
from functools import partial
import numpy as np

def distance_func_one_dim_symmetry(concatenate_input,delta2,n_iw,beta,omega_c):
    dim=int(len(concatenate_input)/2)
    onsite,hopping=concatenate_input[0:dim],concatenate_input[dim:]

    onsite = np.concatenate((onsite,onsite*-1),axis=None)
    hopping = np.concatenate((hopping,hopping*-1),axis=None)

    Omega_n=GF.matsubara_freq(beta=beta,n_w=n_iw)
    delta1=GF.hyb_np(onsite,hopping,n_iw,beta)
    return GF.iwn_distance_func(delta1,delta2,Omega_n,omega_c)

def distance_func_one_dim_symmetry_with_odd(concatenate_input:np.ndarray,
                                            delta2:np.ndarray,
                                            n_iw:int,
                                            beta:float,
                                            omega_c:int):
  
    half_bath_num=int(len(concatenate_input)/2)

    onsite,hopping=concatenate_input[0:half_bath_num],concatenate_input[half_bath_num:]

    if len(onsite) == len(hopping):
        onsite = np.concatenate((onsite,onsite*-1),axis=None)
        hopping = np.concatenate((hopping,hopping*-1),axis=None)
    else:
        onsite = np.concatenate((onsite,onsite*-1,0),axis=None)
        hopping = np.concatenate((hopping[:half_bath_num],
                                hopping[:half_bath_num]*-1,
                                hopping[-1],
                                ),axis=None)

    Omega_n=GF.matsubara_freq(beta=beta,n_w=n_iw)
    delta1=GF.hyb_np(onsite,hopping,n_iw,beta)
    return GF.iwn_distance_func(delta1,delta2,Omega_n,omega_c)


def cal_err_semi_circular_DOS_symmetry(num_imp:int,beta:float,n_iw:int,
                                    W:float=1.,omega_c:int=100,
                                    max_fitting_err:float=2.81,
                                    err_tol:float=1e-3,
                                    delta_from_zero:float=1e-9,
                                    V_bound:list=[-3,3],
                                    E_bound:list=[-3,3],
                                    V_bound_init:list=None,
                                    E_bound_init:list=None,
                                    fit_function:np.ndarray=None,
                                    max_iter:int=100,
                                    minimizer_maxiter:int=100,
                                    err_tol_fix:bool=False,
                                    method = 'BFGS',
                                    disp = True,
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


    if omega_c > n_iw:
        omega_c = n_iw
    
    if fit_function is None:
        Omega_n=GF.matsubara_freq(beta=beta,n_w=n_iw)
        fit_function=GF.greenF(Omega_n=Omega_n,D=W)*((W**2)/4.)
    else:
        assert W == None, 'Force to assert not to give unneeded value'


    NUM_BATH_HALF = int(num_imp/2)
    num_V_bath = NUM_BATH_HALF
    OLD_BATH_NUM = False
    if num_imp%2 != 0: 
        OLD_BATH_NUM = True
        num_V_bath = NUM_BATH_HALF+1

    err = partial(distance_func_one_dim_symmetry_with_odd,delta2=fit_function,n_iw=n_iw,beta=beta,omega_c=omega_c)
    #V_bound 0~3
    assert V_bound[0]==-1*V_bound[1],'symmetry asked to fit semi-o'
    assert E_bound[0]==-1*E_bound[1],'symmetry asked to fit semi-o'
    assert V_bound[0] < V_bound[1], '0 idex corresponding to low limit'
    assert E_bound[0] < E_bound[1], '0 idex corresponding to low limit'
    if disp: print(f"V bound {V_bound}")
    if disp: print(f"E bound {E_bound}")

    V_bound_init = V_bound_init or V_bound
    E_bound_init = E_bound_init or E_bound

    Fit_Success = True
    Keep_run,i=True,0
    fitted_min_err = np.inf
    
    while Keep_run:
        V_guess = np.random.uniform(low=delta_from_zero,high=V_bound_init[1],size=(num_V_bath,))
        E_guess = np.random.uniform(low=delta_from_zero,high=E_bound_init[1],size=(NUM_BATH_HALF,))
        if NUM_BATH_HALF != num_V_bath:
            V_guess[-1]=V_guess[-1]*int(np.random.choice([-1,1],1))

        if OLD_BATH_NUM:
            V_p = np.concatenate((V_guess[:NUM_BATH_HALF],V_guess[:NUM_BATH_HALF]*-1,V_guess[-1]),axis=None)
            E_p = np.concatenate((E_guess[:NUM_BATH_HALF],E_guess[:NUM_BATH_HALF]*-1,0),axis=None)
        else:
            V_p = np.concatenate((V_guess,V_guess*-1),axis=None)
            E_p = np.concatenate((E_guess,E_guess*-1),axis=None)

        # initial_guess = np.concatenate([V_p,E_p])

        initial_guess=np.concatenate([E_guess,V_guess,])

        if disp: print(20*'#',num_imp,20*'#')
        if disp: print(f"{num_imp=}")
        if disp: print(f"{beta=}")
        if disp: print('V_p: ', V_p)
        if disp: print('E_p: ', E_p)
        if disp: print(f"{err(initial_guess)=}")
###########################################################################
# minimize

        options=dict(disp=disp,maxiter=max(minimizer_maxiter,max_iter))
        res=minimize(err,x0=initial_guess,method=method,options=options,tol=err_tol)
        e_list=res.x[:NUM_BATH_HALF]
        V_list=np.abs(res.x[NUM_BATH_HALF:2*NUM_BATH_HALF])
        Keep_run=False

        for element in e_list:
            if element < E_bound[0] or element > E_bound[1]:
                if err(res.x) < err_tol:
                    print(f'Not in E_bound and recalculate {element=:.5f} {E_bound=} {err(res.x)=:.5e}')
                Keep_run=True
        for element in V_list:
            if element < V_bound[0] or element > V_bound[1] or np.abs(element)<delta_from_zero:
                if err(res.x) < err_tol:
                    print(f'Not in V_bound and recalculate {element=:.5f} {V_bound=} {err(res.x)=:.5e}')
                Keep_run=True

        if not Keep_run:
            if err(res.x) < fitted_min_err:
                fitted_min_err = err(res.x)
                fitted_resx = res.x
                print(f"!As within the bound those value recorded:\n!{fitted_min_err=:.5e}\n!{fitted_resx=}")

        if  err(res.x) > err_tol:
            print(f'Err>err_tol Refit. Iter:{i} Err:{err(res.x):.5e} {err_tol=:.5e} FITTED_MIN={fitted_min_err:.5e}')
            Keep_run=True
        
        if Keep_run:
            i+=1

        if i > max_iter:
            i=0
            err_tol=err_tol*2
            print(f'Err_tol adaptive new value {err_tol=:.5e} and FITTED_MIN={fitted_min_err:.5e} {max_fitting_err=:.5e}')
            if err_tol > fitted_min_err:
                Keep_run = False
                res.x = fitted_resx
            if err_tol > max_fitting_err:
                if fitted_min_err < max_fitting_err:
                    Keep_run = False
                    res.x = fitted_resx
                else:
                    Keep_run = False
                    warnings.warn(f'Fitting failed: \nFinal Err:{err(res.x)=} {err_tol=}')
                    Fit_Success = False
                    if err_tol_fix: raise RuntimeError('Fitting Failed')
        
    if OLD_BATH_NUM:
        e_list=res.x[:NUM_BATH_HALF]
        e_list=np.concatenate((e_list,e_list*-1,0),axis=None)

        V_list=res.x[NUM_BATH_HALF:]
        V_list=np.concatenate((V_list[:NUM_BATH_HALF],V_list[:NUM_BATH_HALF]*-1,V_list[-1]),axis=None)
    else:
        e_list=res.x[:NUM_BATH_HALF]
        e_list=np.concatenate((e_list,e_list*-1),axis=None)

        V_list=res.x[NUM_BATH_HALF:2*NUM_BATH_HALF]
        V_list=np.concatenate((V_list,V_list*-1),axis=None)
        
    fitted_erro=err(res.x)
    if disp: print('Fit success: ',res.message)
    if disp: print('Number of iterations: ',res.nit)
    if disp: print('Final error from fit: ',fitted_erro)
    if disp: print('Final parameters: ',res.x)
    if disp: print('E_p: ',e_list)
    if disp: print('V_p: ',V_list)
    if disp: print(20*'#',num_imp,20*'#')

    # fitted_hyb=GF.hyb_np(e_list,V_list,n_iw=n_iw,beta=beta)

    # plot(1j*Omega_n,bethe_iw,fitted_hyb)
    # plot_DOS(fitted_hyb,n_iw=1024,beta=10)
    return num_imp,fitted_erro,Fit_Success,e_list,V_list
