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

    def __init__(self, AIM_solved_dict,index=0,
                solver=default_solver,
                default_solver=default_solver,
                ML_method=None,
                ML_predict=None
                ):
        self.solver=solver
        self.default_solver=default_solver
        self.aim_params=AIM_solved_dict[solver][index]['AIM params']
        
        self.bethe=AIM_solved_dict[solver][index]['bethe']

        self.flat_aim_params=self.flat_aim(self.aim_params)

        self.G_l_from_read_data=AIM_solved_dict[solver][index]['G_l']
        self.G_tau_read=AIM_solved_dict[solver][index]['G_tau']

        self.n_l=AIM_solved_dict[solver][index]['data info']['n_l']
        self.beta=AIM_solved_dict[solver][index]['data info']['beta']
        self.n_tau=AIM_solved_dict[solver][index]['data info']['n_tau']
        self.n_iw=AIM_solved_dict[solver][index]['data info']['n_iw']

        self.title=''
        for key in self.aim_params:
            try:
                unpacked_word=f'{key}: {self.aim_params[key]:.2f} '
            except:
                unpacked_word=f'\n {key}: {self.aim_params[key]}'
            self.title=self.title+unpacked_word
            
        self.title=self.title+f'n_tau: {self.n_tau}'
        self.title=self.title+f' beta: {self.beta}'

        self.G_0_iw_from_read_data=AIM_solved_dict[solver][index]['G0_iw'] # ZZ: have to included here

        self.G_l_from_read=GfLegendre(indices=[0],beta=self.beta,n_points = self.n_l)
        self.G_l_from_read.data[:,:,0] = self.G_l_from_read_data.reshape((len(self.G_l_from_read_data),1))

        self.G_l=GfLegendre(indices=[0],beta=self.beta,n_points = self.n_l)
        self.G_tau=GfImTime(indices=[0],beta=self.beta,n_points=10*self.n_iw+1)
        self.G_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)

        self.Sigma_iw_from_read=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw) #Only DEFAULT_SOLVER support correct Sigma_iw
        self.Sigma_iw_from_read_data=AIM_solved_dict[solver][index]['Sigma_iw']
        self.Sigma_iw_from_read.data[:,0,0]=self.Sigma_iw_from_read_data

        self.Z=AIM_solved_dict[solver][index]['Z']
        self.Z_ground_truth=AIM_solved_dict[default_solver][index]['Z'] # have to include here

        self.G_iw_from_read=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.G_iw_from_read_data=AIM_solved_dict[solver][index]['G_iw']
        self.G_iw_from_read.data[:,0,0]= self.G_iw_from_read_data

        if ML_method is None:
            # Database checking mode
            self.ML_method = None

        else:
            print(20*"#",'Procceeding Machine Learning Predictor',20*"#")
            print('# ML_method:',ML_method)
            self.ML_method=ML_method
            self.G_tau_read=ML_predict(self.flat_aim_params).T[:,0] # overwrite this data

        self.method_g_tau=['from G_tau fit','from G_l read','from G_iw read']
        self.method_hybrid=['discret','S.inp']

    def flat_aim(self,params):
        """
        aim params in, generate flat aim output.
        """
        out_list = []
        for item2 in params:
            out_list.append(params[item2]) 

        out_list=np.hstack(out_list)
        out_list=out_list.reshape(len(out_list),1).T

        return out_list
        

    def hybridisation_from_discret(self):
        """
        use after g_tau fit
        """
        self.Delta_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.Sigma_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.G_0_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)

        if not self.bethe:
            self.Delta_iw << hyb(self.aim_params["E_p"], self.aim_params["V_p"])
            self.G_0_iw << inverse(iOmega_n - self.aim_params["eps"] - self.Delta_iw)
        else:
            self.W=self.aim_params["W"]
            self.G_0_iw << inverse(iOmega_n - self.aim_params["eps"] - self.W**2*SemiCircular(2*self.W)) 


        self.Sigma_iw << inverse(self.G_0_iw) - inverse(self.G_iw)
        return self.G_0_iw 

    def from_ED_KCL_Sinp(self):
        """
        use after g_tau fit
        """
        self.G_0_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)
        self.G_0_iw.data[:,0,0]=self.G_0_iw_from_read_data
        self.Sigma_iw=GfImFreq(indices=[0],beta=self.beta,n_points=self.n_iw)

        # dyson equation use read G_0_iw constructed by ED_KCL G and Sigma
        self.Sigma_iw << inverse(self.G_0_iw) - inverse(self.G_iw)
    
    def Gtau_from_Gl_read(self):
        if self.ML_method is not None:
            raise ValueError('Value predicted is likely G_tau')

        self.G_l << self.G_l_from_read
        self.G_tau.set_from_legendre(self.G_l_from_read)
        self.G_iw << Fourier(self.G_tau)

    def Gtau_from_Giw_read(self): 
        if self.ML_method is not None: # since currently ML_predict G(tau) only.
            raise ValueError('Value predicted is likely G_tau')

        self.G_iw.data[:,0,0]=self.G_iw_from_read_data
        self.G_tau << Fourier(self.G_iw)
        self.G_l << MatsubaraToLegendre(self.G_iw)
        # print(f'from G_iw,{self.G_tau.data[:,:,0][0]},{self.G_tau.data[:,:,0][-1]}')

    def Gtau_from_Gtau_expand_old(self,smoothing_factor=1.0e-6):
        tau_mash=np.linspace(0, self.beta, len(self.G_tau_read), endpoint=True)
        # G_tau_fit=UnivariateSpline(tau_mash,self.G_tau_read)
        G_tau_fit=InterpolatedUnivariateSpline(tau_mash,self.G_tau_read)
        G_tau_fit.set_smoothing_factor(smoothing_factor)

        tau_mash_fit=np.linspace(0, self.beta, 1+(self.n_iw*10), endpoint=True)
        self.G_tau.data[:,:,0]=G_tau_fit(tau_mash_fit).reshape((len(tau_mash_fit),1))


        self.G_iw << Fourier(self.G_tau)
        self.G_l << MatsubaraToLegendre(self.G_iw)

    def Gtau_from_Gtau_expand_stable(self):
        data_tau,_=self.get_gf_xy(self.G_tau)

        self.target_n_tau=len(self.G_tau_read)
        idxs = np.linspace(0, len(data_tau) - 1, self.target_n_tau).astype(int)
        capped_tau=data_tau[idxs]
  
        G_tau_fit=UnivariateSpline(capped_tau,self.G_tau_read,s=0)
        
        self.G_tau.data[:,:,0]=G_tau_fit(data_tau).reshape((len(data_tau),1))
        self.G_iw << Fourier(self.G_tau)
        self.G_l << MatsubaraToLegendre(self.G_iw)

    def Gtau_from_Gtau_expand(self):
        data_tau,_=self.get_gf_xy(self.G_tau)

        self.target_n_tau=len(self.G_tau_read)
        idxs = np.linspace(0, len(data_tau) - 1, self.target_n_tau).astype(int)
        capped_tau=data_tau[idxs]
  
        G_tau_fit=UnivariateSpline(capped_tau,self.G_tau_read,s=0)
        
        self.G_tau.data[:,:,0]=G_tau_fit(data_tau).reshape((len(data_tau),1))
        self.G_iw << Fourier(self.G_tau)
        self.G_l << MatsubaraToLegendre(self.G_iw)


    def get_gf_xy(self,Gf_input):
        mesh=[]
        for t in Gf_input.mesh:
            mesh.append(t.value)
        mesh=np.array(mesh).reshape((len(mesh),1))

        return mesh,Gf_input.data[:,:,0]

    def density_of_states(self,Matsu_points=1024):
        G_w= GfReFreq(indices = [0], window = (-8, 8))
    
        G_w.set_from_pade(self.G_iw,Matsu_points, 1/self.beta)
        if mpi_rank()== 0: print('calculating desity of states: Matsu_points {:d} Freq_Offset {:.2f}'.format(Matsu_points,1/self.beta))
        mesh,G_w_=self.get_gf_xy(Gf_input=G_w)
        A_w=-G_w_.imag/pi
        return mesh,A_w

    def quasi_particle_z(self):
        Z = 1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / pi))
        return Z

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

    def calculate(self,method='from G_l read',hyb_method='S.inp'):

        if method == self.method_g_tau[0]:
            self.Gtau_from_Gtau_expand()

        elif method == self.method_g_tau[1]:
            self.Gtau_from_Gl_read() 

        elif method == self.method_g_tau[2]:
            self.Gtau_from_Giw_read()
        else: 
            raise ValueError('method not right')

        if hyb_method==self.method_hybrid[1]:
            self.from_ED_KCL_Sinp()
        elif hyb_method==self.method_hybrid[0]:
            self.hybridisation_from_discret()
        else:
            raise ValueError('method not right')
    

    def plot_all_info(self,ax,method='from G_l read',hyb_method='S.inp',ls='-',marker='',plot_ground_truth=True):
        """
        method: self.method_g_tau=[from G_l read,'from G_tau fit] 
        """
        if self.ML_method is not None:
            solver=self.ML_method
        else:
            solver=self.solver
        

        self.calculate(method=method,hyb_method=hyb_method)

        x,y=self.get_gf_xy(self.G_tau)
        self.plot_gf_object(ax[0,0],x=x.real,y=y.real,xlabel='tau',
                        ylabel=r'G($\tau$)',
                        legend=solver+' '+method)

        x,y=self.get_gf_xy(self.G_l)
        self.plot_gf_object(ax[0,1],ls='',marker=marker,x=x.real,y=y.real,xlabel='l',
                        ylabel='G_l',alpha=0.5,
                        legend=solver+' '+method)

        x,y=self.get_gf_xy(self.G_iw)
        self.plot_gf_object(ax[1,0],x=x.imag,y=y.real,xlabel='i$\omega_n$',
                        ylabel='ReG(i$\omega_n$)',ls=ls,
                        legend=solver+' '+method)
        self.plot_gf_object(ax[1,1],x=x.imag,y=y.imag,xlabel='i$\omega_n$',
                        ylabel='ImG(i$\omega_n$)',ls=ls,
                        legend=solver+' '+method)
        
        if plot_ground_truth:
            x,y=self.get_gf_xy(self.G_iw_from_read)
            self.plot_gf_object(ax[1,0],x=x.imag,y=y.real,xlabel='i$\omega_n$',
                            ylabel='ReG(i$\omega_n$)',ls=ls,
                            legend='ground truth')
            self.plot_gf_object(ax[1,1],x=x.imag,y=y.imag,xlabel='i$\omega_n$',
                            ylabel='ImG(i$\omega_n$)',ls=ls,
                            legend='ground truth')



        x,y=self.density_of_states()

        self.plot_gf_object(ax[3,0],x=x,y=y,xlabel='$\omega (eV)$',
                                ylabel='A($\omega$) 1/eV',
                                bottom=0,
                                legend=solver+' '+method)
        xlim=50
        ax[1,0].set_xlim(-xlim,xlim)
        ax[1,1].set_xlim(-xlim,xlim)

        x,y=self.get_gf_xy(self.Sigma_iw)
        self.plot_gf_object(ax[2,0],ls=ls,x=x.imag,y=y.real,xlabel='i$\omega_n$',
                        ylabel='$Re\Sigma$(i$\omega_n$)',
                        legend=solver+' %s, G0:%s'%(method,hyb_method))
        self.plot_gf_object(ax[2,1],ls=ls,x=x.imag,y=y.imag,xlabel='i$\omega_n$',
                        ylabel='$Im\Sigma$(i$\omega_n$)',legend=solver+' %s, G0:%s'%(method,hyb_method)
                        )

        if  plot_ground_truth:
            x,y=self.get_gf_xy(self.Sigma_iw_from_read)
            self.plot_gf_object(ax[2,0],ls='--',x=x.imag,y=y.real,xlabel='i$\omega_n$',
                            ylabel='$Re\Sigma$(i$\omega_n$)',
                            legend='ground truth')
            self.plot_gf_object(ax[2,1],ls='--',x=x.imag,y=y.imag,xlabel='i$\omega_n$',
                            ylabel='$Im\Sigma$(i$\omega_n$)',legend='ground truth'
                            )

        if plot_ground_truth:
            ax[3,1].plot([0,1],[0,1],ls='--',color='gray',alpha=1,label='ground truth')
        
        self.plot_gf_object(ax[3,1],marker=marker,x=[self.Z_ground_truth],y=[self.quasi_particle_z()],
                        legend='{:s} {:s} ,hyb {:s}'.format(solver,method,hyb_method),xlabel='Z',ylabel='Z')

        ax[2,0].set_xlim(-xlim,xlim) 
        ax[2,1].set_xlim(-xlim,xlim)

    def GF2DICT(self,method='from G_tau fit',hyb_method='S.inp'):
        """
        self.method_g_tau=['from G_tau fit','from G_l read','from G_iw read']
        """

        if self.ML_method is not None:
            solver=self.ML_method
        else:
            solver=self.solver

        self.calculate(method=method,hyb_method=hyb_method)

        out_dict=self.__OUTDICT__()

        return out_dict


    def __OUTDICT__(self,iw_kep=10,DOS_points=200):
        out_dict={}
        out_dict['aim params']=self.aim_params
        out_dict['Quasi Z']=self.quasi_particle_z()
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

        out_dict['G_tau']=self.G_tau_read
        return out_dict

    def plot_one_solver(self):
        # self.Gtau_from_Gtau_expand() 

        fig, ax = plt.subplots(4, 2,figsize=(8,12),dpi=300)

        method=self.method_g_tau[0] #['from G_tau fit','from G_l read']
        self.plot_all_info(ax,method=method,hyb_method=self.method_hybrid[1],ls='--',marker='o')

        if self.ML_method is not None:
            solver=self.ML_method
        else:
            solver=self.solver

        fig.suptitle(self.title)
        fig.tight_layout()
        fig.savefig('ALL_GF_%s.png'%solver)

    def plot_one_solver_all_method(self):
        # self.Gtau_from_Gtau_expand() 

        fig, ax = plt.subplots(4, 2,figsize=(8,12),dpi=300)
        # fig, ax = plt.subplots()

        plot_ground_truth=True
        for method,ls,marker in zip(self.method_g_tau,['-','-.','--'],['*','^','v']):
            self.plot_all_info(ax,method=method,hyb_method=self.method_hybrid[1],ls=ls,marker=marker,plot_ground_truth=plot_ground_truth)
            plot_ground_truth=False
        # self.plot_all_info(ax,method=self.method_g_tau[1],hyb_method=self.method_hybrid[0],ls='--',marker='o')
        solver=self.solver

        fig.suptitle(self.title,fontsize=20)
        fig.tight_layout()
        fig.savefig('ALL_GF_%s1.png'%solver)

        fig, ax = plt.subplots(4, 2,figsize=(8,12),dpi=300)
        for method,ls,marker in zip(self.method_g_tau,['-','-.','--'],['*','^','v']):
            self.plot_all_info(ax,method=method,hyb_method=self.method_hybrid[0],ls=ls,marker=marker,plot_ground_truth=plot_ground_truth)
        solver=self.solver
        title=''
        for key in self.aim_params:
            unpacked_word=f'{key}: {self.aim_params[key]:.2f} '
            title=title+unpacked_word
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig('ALL_GF_%s2.png'%solver)


def plot_all_database_solvers(AIM_solved_dict,index=0,solvers=[default_solver,'IPT']):
    
    fig, ax = plt.subplots(4, 2,figsize=(8,12),dpi=300)
    plot_ground_truth=True
    for solver,ls,marker in zip(solvers,['-','-.'],['.','o']):
        data_triqs=triqs_interface(AIM_solved_dict=AIM_solved_dict,index=index,solver=solver)
        data_triqs.plot_all_info(ax,ls=ls,marker=marker,method=data_triqs.method_g_tau[1],plot_ground_truth=plot_ground_truth)
        plot_ground_truth=False

    fig.tight_layout()
    fig.savefig('ALL_GF.png')

def plot_all_database_solver_and_ML(AIM_solved_dict,index=0,solvers=[default_solver,'IPT'],ML_method=None,ML_predict=None):
    
    fig, ax = plt.subplots(4, 2,figsize=(8,12),dpi=300)
    plot_ground_truth=True
    for solver,ls,marker in zip(solvers,['-','-.'],['.','o']):
        data_triqs=triqs_interface(AIM_solved_dict=AIM_solved_dict,index=index,solver=solver)
        data_triqs.plot_all_info(ax,ls=ls,marker=marker,method=data_triqs.method_g_tau[0],plot_ground_truth=plot_ground_truth)
        plot_ground_truth=False

    data_triqs=triqs_interface(AIM_solved_dict=AIM_solved_dict,index=index,solver=solver,ML_method=ML_method,ML_predict=ML_predict)
    data_triqs.plot_all_info(ax,ls='--',marker='^',
                            method=data_triqs.method_g_tau[0])
    fig.suptitle(data_triqs.title)
    fig.tight_layout()
    fig.savefig('ALL_GF.png')
    print('ALL_GF.png created')


    