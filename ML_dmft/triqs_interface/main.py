from triqs.gf import *
import numpy as np
import matplotlib
from triqs.plot.mpl_interface import oplot
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from ML_dmft.database.constant import default_solver

matplotlib.style.use(plt.style.available[-2])

class triqs_interface():
    
    """
    triqs_interface, triqs interface lightweight
    should just construct from
    1) input of impurity Green's function
    2) input of hybridisation
    3) input of impurity self energy
    """

    def __init__(self,data_dict):
        self.data_dict=data_dict
        self.beta=data_dict['data info']['beta']
        self.U=data_dict['AIM params']['U']
        self.aim_params=data_dict['AIM params']
        self.n_l=data_dict['data info']['n_l']
        self.n_iw=data_dict['data info']['n_iw']
        self.n_tau_target_tau=data_dict['data info']['n_tau']
        self.G_tau_read=data_dict['G_tau']

        self.title=''
        for key in self.aim_params:
            try:
                unpacked_word=f'{key}: {self.aim_params[key]:.2f} '
            except:
                unpacked_word=f'\n {key}: {self.aim_params[key]}'
            self.title=self.title+unpacked_word

        self.G_tau=GfImTime(indices=[0],beta=self.beta,n_points=10*self.n_iw+1)
        self.G0_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.G_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.G_l=GfLegendre(indices=[0],beta=self.beta,n_points = self.n_l)
        self.Sigma_iw=self.G_iw.copy()
        self.__load__()

    def __load__(self):
        self.G0_iw.data[:,0,0]=self.data_dict['G0_iw'][0] 
        self.G_iw.data[:,0,0]=self.data_dict['G_iw'][0]
        # self.Sigma_iw << inverse(self.G0_iw) - inverse(self.G_iw)
    
    def Gtau_from_Gtau_expand_test(self):
        tau_mash=np.linspace(0, self.beta, self.n_tau_target_tau, endpoint=True)

        G_tau_fit=UnivariateSpline(tau_mash,self.G_tau_read)
        G_tau_fit.set_smoothing_factor(1.0e-6)

        tau_mash_fit=np.linspace(0, self.beta, 1+(self.n_iw*10), endpoint=True)
        self.G_tau.data[:,:,0]=G_tau_fit(tau_mash_fit).reshape((len(tau_mash_fit),1))
        self.G_iw_from_G_tau = self.G_iw.copy()
        self.G_iw_from_G_tau << Fourier(self.G_tau)

        point1=self.G_iw_from_G_tau.data[:,:,0][self.n_iw:self.n_iw+10]
        point2=self.G_iw.data[:,:,0][self.n_iw:self.n_iw+10]

        if np.abs(point1.imag-point2.imag) < 1e-3:
            raise ValueError('G_tau reconstructed error too large')

    def Gtau_from_Gtau_expand(self,input_G_tau,smoothing_factor=1.0e-6):
        tau_mash=np.linspace(0, self.beta, self.n_tau_target_tau, endpoint=True)
        self.G_tau_read=input_G_tau.flatten()
        G_tau_fit=UnivariateSpline(tau_mash,input_G_tau,s=0)
        G_tau_fit.set_smoothing_factor(smoothing_factor)

        tau_mash_fit=np.linspace(0, self.beta, 1+(self.n_iw*10), endpoint=True)
        self.G_tau.data[:,:,0]=G_tau_fit(tau_mash_fit).reshape((len(tau_mash_fit),1))

        self.G_iw << Fourier(self.G_tau)
        self.G_l <<  MatsubaraToLegendre(self.G_iw)
        self.Sigma_iw << inverse(self.G0_iw) - inverse(self.G_iw)
    
        self.findZ()

    def findZ(self):
        self.Z = 1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi))
        return self.Z


class triqs_plot():
    def __init__(self,triqs_interface,solver):

        # copy variable from triqs_interface
        self.beta=triqs_interface.beta
        self.U=triqs_interface.U
        self.aim_params=triqs_interface.aim_params
        self.n_l=triqs_interface.n_l
        self.n_iw=triqs_interface.n_iw

        self.G_l=triqs_interface.G_l
        self.G_tau=triqs_interface.G_tau
        self.G_iw=triqs_interface.G_iw
        self.Sigma_iw=triqs_interface.Sigma_iw
        self.G0_iw=triqs_interface.G0_iw
        self.G_tau_read=triqs_interface.G_tau_read

        self.Z=triqs_interface.Z

        #name of the solver to plot
        self.solver=solver

        pass

    def plot_gf_object(self,ax,
                             x,
                             y,
                             bottom=None,
                             xlabel=' ',
                             ylabel=' ',
                             legend=' ',
                             ls='-',
                             marker='',
                             alpha=1,
                             markersize=5
                            ):

        ax.plot(x,y,label=legend,linestyle=ls,marker=marker,alpha=alpha,markersize=markersize)
        ax.legend()
        # ax.set_xlim(-4,4)
        if bottom is not None: ax.set_ylim(bottom=bottom)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def density_of_states(self,Matsu_points=1024):
        G_w= GfReFreq(indices = [0], window = (-8, 8))
    
        G_w.set_from_pade(self.G_iw,Matsu_points, 1/self.beta)
        print('calculating desity of states: Matsu_points {:d} Freq_Offset {:.2f}'.format(Matsu_points,1/self.beta))
        mesh,G_w_=self.get_gf_xy(Gf_input=G_w)
        A_w=-G_w_.imag/np.pi
        return mesh,A_w

    def get_gf_xy(self,Gf_input):
        mesh=[]
        for t in Gf_input.mesh:
            mesh.append(t.value)
        mesh=np.array(mesh).reshape((len(mesh),1))

        return mesh,Gf_input.data[:,:,0]

    def __OUTDICT__(self,iw_kep=10,DOS_points=200):
        out_dict={}
        out_dict['aim params']=self.aim_params
        out_dict['Quasi Z']=self.Z
        out_dict['n']=self.G_iw.density().real[0][0]
        out_dict['Z']=self.Z
        out_dict['beta']=self.beta

        _,y=self.get_gf_xy(self.G_iw)
        shape_m=int(y.shape[0]/2)
        out_dict['G_iw']=y[shape_m:shape_m+iw_kep]

        _,y=self.get_gf_xy(self.Sigma_iw)
        shape_m=int(y.shape[0]/2)
        out_dict['Sigma_iw']=y[shape_m:shape_m+iw_kep]

        x_,y_=self.density_of_states()
        idxs = np.linspace(0, y_.shape[0] - 1, DOS_points).astype(int)
        x=np.zeros((len(idxs),1))
        y=np.zeros((len(idxs),1))

        for i,idx in enumerate(idxs):
            x[i][0]=x_[idx][0]
            y[i][0]=y_[idx][0]
        out_dict['A_w']=y
        out_dict['A_w_w']=x

        _,y=self.get_gf_xy(self.G_l)
        out_dict['G_l']=y

        out_dict['G_tau']=self.G_tau_read #G_tau predicted 

        return out_dict

    def plot_all_info(self,ax,ls='-',marker='',Z_ground_truth=0,ground_truth_solver=default_solver):
            """
            method: self.method_g_tau=[from G_l read,'from G_tau fit] 
            """
            solver=self.solver

            if solver == ground_truth_solver:
                plot_ground_truth=True
            else:
                plot_ground_truth=False

            x,y=self.get_gf_xy(self.G_tau)
            self.plot_gf_object(ax[0,0],x=x.real,y=y.real,xlabel='tau',
                            ylabel=r'G($\tau$)',
                            legend=solver)

            x,y=self.get_gf_xy(self.G_l)
            self.plot_gf_object(ax[0,1],ls='',marker=marker,x=x.real,y=y.real,xlabel='l',
                            ylabel='G_l',alpha=0.5,
                            legend=solver)

            x,y=self.get_gf_xy(self.G_iw)
            self.plot_gf_object(ax[1,0],x=x.imag,y=y.real,xlabel='i$\omega_n$',
                            ylabel='ReG(i$\omega_n$)',ls=ls,
                            legend=solver)
            self.plot_gf_object(ax[1,1],x=x.imag,y=y.imag,xlabel='i$\omega_n$',
                            ylabel='ImG(i$\omega_n$)',ls=ls,
                            legend=solver)

            x,y=self.density_of_states()

            self.plot_gf_object(ax[3,0],x=x,y=y,xlabel='$\omega (eV)$',
                                    ylabel='A($\omega$) 1/eV',
                                    bottom=0,
                                    legend=solver)

            ax[1,0].set_xlim(-10,10)
            ax[1,1].set_xlim(-10,10)

            x,y=self.get_gf_xy(self.Sigma_iw)
            self.plot_gf_object(ax[2,0],ls=ls,x=x.imag,y=y.real,xlabel='i$\omega_n$',
                            ylabel='$Re\Sigma$(i$\omega_n$)',
                            legend=solver)
            self.plot_gf_object(ax[2,1],ls=ls,x=x.imag,y=y.imag,xlabel='i$\omega_n$',
                            ylabel='$Im\Sigma$(i$\omega_n$)',legend=solver
                            )

            if plot_ground_truth:
                ax[3,1].plot([0,1],[0,1],ls='--',color='gray',alpha=1,label='ground truth')
            
            self.plot_gf_object(ax[3,1],marker=marker,x=[Z_ground_truth],y=[self.Z],
                            legend='{:s}'.format(solver),xlabel='Z',ylabel='Z')
                                

            ax[2,0].set_xlim(-10,10)
            ax[2,1].set_xlim(-10,10)

    def plot_one_solver(self):
        # self.Gtau_from_Gtau_expand() 

        fig, ax = plt.subplots(4, 2,figsize=(8,12),dpi=300)


        self.plot_all_info(ax,ls='--',marker='o')

        solver='AndT'
        title=''
        for key in self.aim_params:
            unpacked_word=f'{key}: {self.aim_params[key]:.2f} '
            title=title+unpacked_word
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig('ALL_GF_%s.png'%solver)