import ML_dmft.numpy_GF.GF as GF
import ML_dmft.numpy_GF.common as PYDMFT_GF
import triqs.gf as tgf 
import numpy as np

def self_energy_tails(sigma_tau_triqs:tgf.GfImTime,
                          beta:float,U:float,
                          n_iw=1024)->tgf.GfImFreq:
    """
    Fourier Transform Sigma(tau)->Sigma(iwn)
    input
    ------
        sigma_tau : triqs GfImTime
        beta      : inverse temperature

    """
    sigma_tau = sigma_tau_triqs.data[:,0,0]
    n_tau=sigma_tau.shape[0]
    _, w_n = PYDMFT_GF.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=n_iw))
    tau = np.linspace(0,beta,n_tau)
    sigma_iwn = PYDMFT_GF.gt_fouriertrans(sigma_tau, tau, w_n, [U**2 / 4., 0., 0.])
    out_triqs_sigma = tgf.GfImFreq(indices=[0],
                            beta=beta,
                            n_points=n_iw)
    sigma_iwn = GF.iw_obj_flip(sigma_iwn[:,np.newaxis])
    out_triqs_sigma.data[:,:,0] = sigma_iwn
    return out_triqs_sigma

#TODO replace Sigma_iw with high frequency tails