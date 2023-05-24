from triqs.gf import *
from triqs.plot.mpl_interface import oplot,plt
from math import pi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline,InterpolatedUnivariateSpline
from ML_dmft.database.constant import default_solver
from copy import deepcopy
from ML_dmft.utility.mpi_tools import mpi_rank
import typing

matplotlib.style.use(plt.style.available[-2])

def hyb(onsite, hopping):
    """
    Returns hybridisation function
    """
    return sum((V_**2)*inverse(iOmega_n - E_) for E_, V_ in zip(onsite, hopping))

def get_gf_xy(Gf_input):
        mesh=[]
        for t in Gf_input.mesh:
                mesh.append(t.value)
        mesh=np.array(mesh).reshape((len(mesh),1))
        return mesh,Gf_input.data[:,:,0]


class triqs_interface():
    """
    should just construct from
    1) input of impurity Green's function
    2) input of hybridisation
    3) input of impurity self energy
    """
    def __init__(self,dataset):
        """
        dataset contains all aim info.
        """
        self.solver=dataset.solver

        # load from database
        self.G_l_from_read_data=dataset.G_l
        self.G_tau_from_read_data=dataset.G_tau
        self.G_iw_from_read_data=dataset.G_iw
        self.G0_iw_from_read_data=dataset.G0_iw
        self.Sigma_iw_from_read_data=dataset.Sigma_iw

        self.aim_params=dataset.aim_params
        self.n_l=dataset.n_l
        self.beta=dataset.beta
        self.n_iw=dataset.n_iw
        self.n_tau_from_read=dataset.n_tau
        self.n_tau=(self.n_iw*10)+1
        self.Z_from_read=dataset.Z

        self.__load__()
        self.__init_GF__()


    def __load__(self):
        self.G_l_from_read=GfLegendre(indices=[0],beta=self.beta,n_points = self.n_l)
        self.G_l_from_read.data[:,:,0] = self.G_l_from_read_data

        self.G_iw_from_read=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.G_iw_from_read.data[:,:,0]= self.G_iw_from_read_data

        self.Sigma_iw_from_read=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.Sigma_iw_from_read.data[:,:,0]=self.Sigma_iw_from_read_data

        self.G0_iw_from_read=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.G0_iw_from_read.data[:,:,0]= self.G0_iw_from_read_data

    def __init_GF__(self):
        self.G_l=GfLegendre(indices=[0],beta=self.beta,n_points = self.n_l)
        self.G_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.G_tau=GfImTime(indices=[0],beta=self.beta,n_points=10*self.n_iw+1)
        self.Sigma_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.G0_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)

    def G0_more_basis(self):
        self.G0_l=GfLegendre(indices=[0],beta=self.beta,n_points = self.n_l)
        self.G0_tau=GfImTime(indices=[0],beta=self.beta,n_points=10*self.n_iw+1)
        
        self.G0_iw << self.G0_iw_from_read
        self.G0_tau << Fourier(self.G0_iw)
        self.G0_l << MatsubaraToLegendre(self.G0_iw)

    def __proceed__(self):

        self.method_g_tau=['from G_tau fit','from G_l read','from G_iw read']
        self.method_hybrid=['discret','S.inp']
    
    def G_from_Gtau_expand(self):
        data_tau,_=get_gf_xy(self.G_tau)

        self.target_n_tau=len(self.G_tau_from_read_data)
        idxs = np.linspace(0, len(data_tau) - 1, self.n_tau_from_read).astype(int)
        capped_tau=data_tau[idxs]
  
        G_tau_fit=UnivariateSpline(capped_tau,self.G_tau_from_read_data,s=0)
        
        self.G_tau.data[:,:,0]=G_tau_fit(data_tau).reshape((len(data_tau),1))
        self.G_iw << Fourier(self.G_tau)
        self.G_l << MatsubaraToLegendre(self.G_iw)
        self.density=self.G_iw.density().real[0][0]  

    def G_from_Giw_read(self): 
        self.G_iw.data[:,:,0]=self.G_iw_from_read_data
        self.G_tau << Fourier(self.G_iw)
        self.G_l << MatsubaraToLegendre(self.G_iw)
        self.density=self.G_iw.density().real[0][0]

    def G_from_Gl_read(self): 
        self.G_l.data[:,:,0]=self.G_l_from_read_data
        self.G_iw << LegendreToMatsubara(self.G_l)
        self.G_tau << Fourier(self.G_iw)
        self.density=self.G_iw.density().real[0][0]

    def G0_from_read(self):
        self.G0_iw << self.G0_iw_from_read
        self.Sigma_iw << inverse(self.G0_iw) - inverse(self.G_iw)
        self.quasi_particle_z()
    
    def Sigma_from_read(self):
        self.Sigma_iw << self.Sigma_iw_from_read
        self.G0_iw << self.Sigma_iw + inverse(self.G_iw)
        self.quasi_particle_z()

    def quasi_particle_z(self):
        self.Z = 1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / pi))
        return self.Z

    @staticmethod
    def quasi_particle_weight(sigma_iw,beta):
        r"""
        Input
        -----
        triqs format $\Sigma$
        """
        Z = 1 / (1 - (sigma_iw(0)[0,0].imag * beta / pi))
        return Z 

class triqs_plot(triqs_interface):
    def __init__(self,dataset):
        super().__init__(dataset)
        self.title=self.gen_title()
        pass
    
    def gen_title(self):
        title=''
        for key in self.aim_params:
            try:
                unpacked_word=f'{key}: {self.aim_params[key]:.2f} '
            except:
                unpacked_word=f'\n {key}: {self.aim_params[key]}'
            title=title+unpacked_word
            
        title=title+f' beta: {self.beta}'
        return title

    def plot_gf_object(self,ax,
                             x,
                             y,
                             bottom=None,
                             xlabel=' ',
                             ylabel=' ',
                             legend=' ',
                             ls='-',
                             marker='',
                             alpha=0.5,
                             markersize=5
                            ):

        ax.plot(x,y,label=legend,linestyle=ls,marker=marker,alpha=alpha,markersize=markersize)
        ax.legend()
        # ax.set_xlim(left=0)
        if bottom is not None: ax.set_ylim(bottom=bottom)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def density_of_states(self,Matsu_points=1024,
                            xlim_left=-8,xlim_right=8,
                            freq_offset:float=None,
                            )->typing.Tuple[np.ndarray,np.ndarray]:
        freq_offset = freq_offset or 1/self.beta
        G_w= GfReFreq(indices = [0], window = (xlim_left, xlim_right))
        G_w.set_from_pade(self.G_iw,Matsu_points, freq_offset)
        print('calculating desity of states: Matsu_points {:d} Freq_Offset {:.2f}'.format(Matsu_points,freq_offset))
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

    def plot_Gtau(self,ax,legend):
        x,y=self.get_gf_xy(self.G_tau)
        self.plot_gf_object(ax,x=x.real,y=y.real,xlabel='tau',
                            ylabel=r'G($\tau$)',
                            legend=legend)
    
    def plot_Gl(self,ax,legend,marker,ls=''):
        x,y=self.get_gf_xy(self.G_l)
        self.plot_gf_object(ax,ls='',marker=marker,x=x.real,y=y.real,xlabel='l_n',
                        ylabel='$G_l$',alpha=0.5,
                        legend=legend)
    
    def plot_iw_obj(self,ax_left,ax_right,iw_obj,legend,Ylabel,marker,xwindow,ls=''):
        x,y=self.get_gf_xy(iw_obj)
        if ax_left is not None: self.plot_gf_object(ax_left,x=x.imag,y=y.real,xlabel='i$\omega_n$',
                            ylabel=f'Re{Ylabel}(i$\omega_n$)',ls=ls,marker=marker,
                            legend=legend)
        if ax_right is not None: self.plot_gf_object(ax_right,x=x.imag,y=y.imag,xlabel='i$\omega_n$',
                        ylabel=f'Im{Ylabel}(i$\omega_n$)',ls=ls,marker=marker,
                        legend=legend)
        if ax_left is not None: ax_left.set_xlim(-xwindow,xwindow)
        if ax_right is not None: ax_right.set_xlim(-xwindow,xwindow)
        
    def plot_all_info(self,ax,
                        ls='-',
                        marker='',
                        plot_ground_truth=False,
                        xwindow=10,
                        legend='',
                        Z_GT=0,
                        density_GT=0,
                        ):

            self.plot_Gtau(ax[0,0],legend=legend)
            self.plot_Gl(ax[0,1],legend=legend,marker=marker,ls='')

            self.plot_iw_obj(ax[0,2],ax[0,3],self.G_iw,legend=legend,Ylabel='G',marker=marker,xwindow=xwindow)

            self.plot_iw_obj(ax[1,0],ax[1,1],self.G0_iw,legend=legend,Ylabel='G0',marker=marker,xwindow=xwindow)

            self.plot_iw_obj(ax[1,2],ax[1,3],self.Sigma_iw,legend=legend,Ylabel='$\Sigma$',marker=marker,xwindow=xwindow)

            x,y=self.density_of_states()

            self.plot_gf_object(ax[2,0],x=x,y=y,xlabel='$\omega (eV)$',
                                    ylabel='A($\omega$) 1/eV',
                                    bottom=0,
                                    legend=legend)


            if plot_ground_truth:
                ax[2,1].plot([0,1],[0,1],ls='--',color='gray',alpha=1,label='ground truth')
                ax[2,2].plot([0,1],[0,1],ls='--',color='gray',alpha=1,label='ground truth')
            
            #groud_truth Z is supposed from read

            self.plot_gf_object(ax[2,1],marker=marker,x=[Z_GT],y=[self.Z],ls='',legend='{:s}'.format(legend),xlabel='Z',ylabel='Z')

            self.plot_gf_object(ax[2,2],marker=marker,x=[density_GT],y=[self.density],ls='',legend='{:s}'.format(legend),xlabel='n',ylabel='n')


class triqs_plot_variation(triqs_plot):
    def __init__(self,dataset) -> None:
        super().__init__(dataset)
        self.len_G_method=3
        self.len_Sigma_method=2
        self.xwindow=20
        long_text=f"""
{50*'#'}
triqs_plot_variation.proceed
Options
-------
G_method,Sigma_method

G_method:
0:  G_from_Gtau_expand
1:  G_from_Giw_read
2:  G_from_Gl_read

Sigma_method:
0:  G0_from_read
1:  Sigma_from_read
2:  G0 from aim params

{50*'#'}
"""  
        print(long_text)

    def proceed(self,G_method,Sigma_method):
        """"
        G0:G_from_Gtau_expand
        G1:G_from_Giw_read
        G2:G_from_Gl_read
        
        S0:G0_from_read
        S1:Sigma_from_read:
        """
        if G_method==0:
            self.G_from_Gtau_expand()
            legend_p1='G from Gtau expand \n'
        elif G_method==1:
            self.G_from_Giw_read()
            legend_p1='G from Giw read \n'
        elif G_method==2:
            self.G_from_Gl_read()
            legend_p1='G from Gl read \n'
        if Sigma_method==0:
            self.G0_from_read()
            legend_p2='G0 from read'
        elif Sigma_method==1:
            self.Sigma_from_read()
            legend_p2='Sigma from read'
        elif Sigma_method==2:
            legend_p2='Aim hyb'
            #TODO:
            raise SyntaxError('function not developed yet')
        
        self.legend=f'{self.solver} {legend_p1} {legend_p2}'
        return 


    def plot_one_solver(self):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.style.use(plt.style.available[-2])

        self.proceed(1,1)
        Z_GT=self.Z
        density_GT=self.density
        
        fig, ax = plt.subplots(3, 4,figsize=(16,9),dpi=300)
        self.plot_all_info(ax,ls='',
                        marker='.',
                        plot_ground_truth=False,
                        xwindow=self.xwindow,
                    legend=self.legend,
                    Z_GT=Z_GT,
                    density_GT=density_GT)
        fig.suptitle(self.title,fontsize=15)
        fig.tight_layout()
        print(f'plot.png is created')
        fig.savefig(f'plot.png')


    def plot_one_solver_all_method(self):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.style.use(plt.style.available[-2])

        self.proceed(2,1)
        Z_GT=self.Z
        density_GT=self.density
        
        for j in range(self.len_Sigma_method):
            fig, ax = plt.subplots(3, 4,figsize=(16,9),dpi=300)
            plot_ground_truth=True
            for i in range(self.len_G_method):
                self.proceed(i,j)
                self.plot_all_info(ax,ls='',
                             marker='.',
                             plot_ground_truth=plot_ground_truth,
                             xwindow=self.xwindow,
                            legend=self.legend,
                            Z_GT=Z_GT,
                            density_GT=density_GT,
                            )
                plot_ground_truth=False
            fig.suptitle(self.title,fontsize=15)
            fig.tight_layout()
            print(f'ALL_GF_{self.solver}_{j}.png is created')
            fig.savefig(f'ALL_GF_{self.solver}_{j}.png')