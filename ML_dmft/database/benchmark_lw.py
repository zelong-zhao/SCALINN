from ML_dmft.database.ml_model import ML_model
from ML_dmft.database.constant import default_solver
from ML_dmft.database.database import load_database
import pickle,os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import argparse

class benchmark():
    def __init__(self,
            all_data_dict,
            test_db='Mott',
            ML_method=None,
            AIM_method=None):

        self.test_db=test_db

        self.AIM_method=AIM_method
        self.ML_method=ML_method
        self.ground_truth=default_solver

        if self.AIM_method == None and self.ML_method == None:
            raise ValueError('AIM_method or ML_method must have one')
        elif self.AIM_method != None and self.ML_method != None:
            raise ValueError('AIM_method or ML_method only one')

        if self.AIM_method == self.ground_truth:
            raise ValueError('Cannot Benchmark with ED_KCL')

        if self.AIM_method != None and self.AIM_method !='ED_KCL':
            print('Proceeding AIM Solver Benchmark:',self.AIM_method)
            self.solver2test=self.AIM_method

        if self.ML_method != None:
            self.solver2test=self.ML_method
        
        self.all_data_dict=all_data_dict

        self.number_of_solver=len(self.all_data_dict)
        self.number_of_data=len(self.all_data_dict[self.ground_truth])
        self.proceed()


    def proceed(self):
        self.err_dict={}
        self.err_dict['info']={'ground truth':self.ground_truth,
                               'solver test':self.solver2test
                              }

        self.err_dict['Z']=self.get_Z()

        self.err_dict['n']=self.get_n()

        self.err_dict['G_tau']=self.get_G_tau()

        self.err_dict['G_l']=self.get_G_l()

        self.err_dict['Re G_iw']=self.get_iw_object(real=True,gtype='G_iw')
        self.err_dict['Imag G_iw']=self.get_iw_object(real=False,gtype='G_iw')

        self.err_dict['Re Sigma_iw']=self.get_iw_object(real=True,gtype='Sigma_iw')
        self.err_dict['Imag Sigma_iw']=self.get_iw_object(real=False,gtype='Sigma_iw')

        self.err_dict['Aw']=self.get_Aw()

        if self.test_db == 'Mott':
            self.err_dict['Mott']=self.get_Z_Mott()
        
        self.std_err={}
        for item in self.err_dict:
            if 'y true' in self.err_dict[item] and item != 'Mott':
                self.std_err[item] = {'mean': np.mean(self.err_dict[item]['y true']-self.err_dict[item]['y test'],axis=1),
                                'std':np.std(self.err_dict[item]['y true']-self.err_dict[item]['y test'],axis=1)
                                }


    def get_Aw(self) -> dict:
        y_true,y_test=self.G_object_y_y_truth(y_arg='A_w')
        x_true,x_test=self.G_object_y_y_truth(y_arg='A_w_w')
        err=y_true-y_test
        err_dict={'x':x_true,
                'y true':y_true,
                'y test':y_test,
                'err':err}
        return err_dict

    def get_iw_object(self,gtype,real=True) -> dict:
        n_l=self.all_data_dict[self.ground_truth][0][gtype].shape[0]
        y_true,y_test=self.G_object_y_y_truth(y_arg=gtype,real=real)

        x=np.zeros((n_l,self.number_of_data))
        for i in range(x.shape[-1]):
            x[:,i]=self.matsu(number_iw=n_l,beta=1)
        err=y_true-y_test

        err_dict={'x':x,
                'y true':y_true,
                'y test':y_test,
                'err':err}

        return err_dict

    def get_Z_Mott(self):
        x,y,U=[],[],[]
        for idx,item in enumerate(self.all_data_dict[self.solver2test]):
            x.append(self.all_data_dict[self.ground_truth][idx]['Z'])
            y.append(item['Quasi Z'])
            U.append(item['aim params']['U'])

        x=np.array(x)
        y=np.array(y)
        U=np.array(U)

        err_dict={'y true':x,
                'y test':y,
                'U':U
                }
        return err_dict
        
    def get_Z(self) -> dict:

        x,y=[],[]
        for idx,item in enumerate(self.all_data_dict[self.solver2test]):
            x.append(self.all_data_dict[self.ground_truth][idx]['Z'])
            y.append(item['Quasi Z'])
        
        x=np.array(x).reshape((len(x),1))
        y=np.array(y).reshape((len(y),1))
        err=x-y

        err_dict={}
        err_dict['y true']=x
        err_dict['y test']=y
        err_dict['err']=err
        return err_dict

    def get_n(self) -> dict:
        x,y=[],[]
        for idx,item in enumerate(self.all_data_dict[self.solver2test]):
            x.append(self.all_data_dict[self.ground_truth][idx]['n'])
            y.append(item['n'])
        x=np.array(x).reshape((len(x),1))
        y=np.array(y).reshape((len(y),1))
        err=x-y
        err_dict={}
        err_dict['y true']=x
        err_dict['y test']=y
        err_dict['err']=err
        return err_dict

    def get_G_tau(self) -> dict:
        n_l=self.all_data_dict[self.ground_truth][0]['G_tau'].shape[0]

        y_true_,y_test_=self.G_object_y_y_truth(y_arg='G_tau')
        err_=y_true_-y_test_

        x_=np.zeros((n_l,self.number_of_data))
        for i in range(x_.shape[-1]):
            x_[:,i]=np.arange(0,n_l)

        reserved_length=10

        x=np.zeros((reserved_length,self.number_of_data))
        y_true=np.zeros((reserved_length,self.number_of_data))
        y_test=np.zeros((reserved_length,self.number_of_data))
        err=np.zeros((reserved_length,self.number_of_data))
        
        idxs=np.linspace(0,n_l-1,reserved_length,dtype=int)
        for line_num in range(self.number_of_data):
            for idx,item in enumerate(idxs):
                x[idx,line_num]=x_[item,line_num]
                y_true[idx,line_num]=y_true_[item,line_num]
                y_test[idx,line_num]=y_test_[item,line_num]
                err[idx,line_num]=err_[item,line_num]

        err_dict={'x':x,'y true':y_true,'y test':y_test,'err':err}

        return err_dict

    def get_G_l(self) -> dict:
        n_l=self.all_data_dict[self.ground_truth][0]['G_l'].shape[0]
        y=np.zeros((n_l,1,self.number_of_data))
        y1=np.zeros((n_l,1,self.number_of_data))
        x=np.zeros((n_l,1,self.number_of_data))
        for i in range(x.shape[-1]):
            x[:,:,i]=np.arange(n_l).reshape(n_l,1)

        for idx,item in enumerate(self.all_data_dict[self.solver2test]):
            y[:,:,idx]=self.all_data_dict[self.ground_truth][idx]['G_l'].real
            y1[:,:,idx]=item['G_l'].real

        y1=y1.reshape(n_l,self.number_of_data)
        y=y.reshape(n_l,self.number_of_data)
        x=x.reshape(n_l,self.number_of_data)

        y1=y1[:10,:]
        y=y[:10,:]
        x=x[:10,:]

        err=y-y1

        err_dict={'x':x,'y true':y,'y test':y1,'err':err}
        return err_dict


    def quasi(self,ax,marker='.',ls='-'):

        x=self.err_dict['Z']['y true']
        y=self.err_dict['Z']['y test']

        ax.scatter(x,y,marker=marker,label='Method %s'%self.solver2test)
        ax.plot([0,1],[0,1],ls='--',color='gray',alpha=0.3,label='ground truth')
        ax.set_xlabel('Z (ground truth)')
        ax.set_ylabel('Z (%s)'%self.solver2test)
        ax.set_ylim(bottom=0-0.01,top=1+0.01)
        ax.legend()
    
    
    def n(self,ax,marker='.',ls='-'):
        x=self.err_dict['n']['y true']
        y=self.err_dict['n']['y test']

        ax.scatter(x,y,marker=marker,label='Method %s'%self.solver2test)
        ax.plot([0,1],[0,1],ls='--',color='gray',alpha=0.3,label='ground truth')

        ax.set_xlabel('n (ground truth)')
        ax.set_ylabel('n (%s)'%self.solver2test)
        ax.legend()

    def G_l(self,ax,marker='.',ls='-'):
        x,err=self.err_dict['G_l']['x'],self.err_dict['G_l']['err']
        if self.errbar:
            item='G_l'
            err_x,mean_y,std_y=self.err_dict[item]['x'][:,0], \
                                self.std_err[item]['mean'],\
                                self.std_err[item]['std']
            ax.errorbar(err_x,mean_y,std_y,label='Method %s'%self.solver2test,capsize=3)
        else:
            ax.scatter(x,err,marker=marker,label='Method %s'%self.solver2test)

        ax.plot([x.max(),x.min()],[0,0],label='ground truth',color='gray')
        ax.set_xlabel('l')
        ax.set_ylabel('G_l({}-{})'.format(self.ground_truth,self.solver2test))
        ax.set_ylim(top=np.abs(err).max(),bottom=-np.abs(err).max())
        ax.legend()


    def G_object_y_y_truth(self,y_arg,real=True):
        n_l=self.all_data_dict[self.ground_truth][0][y_arg].shape[0]
        
        y_true=np.zeros((n_l,self.number_of_data))
        y_test=np.zeros((n_l,self.number_of_data))
        for idx,item in enumerate(self.all_data_dict[self.solver2test]):
            if real: 
                y_true[:,idx]=self.all_data_dict[self.ground_truth][idx][y_arg].real.flatten()
                y_test[:,idx]=item[y_arg].real.flatten()
            else:
                y_true[:,idx]=self.all_data_dict[self.ground_truth][idx][y_arg].imag.flatten()
                y_test[:,idx]=item[y_arg].imag.flatten()
        return y_true,y_test

    def G_tau(self,ax,marker='.'):

        x,err=self.err_dict['G_tau']['x'],self.err_dict['G_tau']['err']

        n_l=x.shape[0]
        print('ensure number of tau points are same!')

        if self.errbar:
            item='G_tau'
            err_x,mean_y,std_y=self.err_dict[item]['x'][:,0], \
                                self.std_err[item]['mean'],\
                                self.std_err[item]['std']
            ax.errorbar(err_x,mean_y,std_y,label='Method %s'%self.solver2test,capsize=3)
        else:
            ax.scatter(x,err,marker=marker,label='Method %s'%self.solver2test)

        ax.plot([x.max(),x.min()],[0,0],label='ground truth',color='gray')

        ax.set_xticks([0,x.max()])
        ax.set_xticklabels([0,r'$\beta$'])
        xticks=np.linspace(0,x.max(),10,dtype=int)
        ax.set_xticks(xticks)

        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('G_tau({}-{})'.format(self.ground_truth,self.solver2test))
        ax.set_ylim(top=np.abs(err).max(),bottom=-np.abs(err).max())
        ax.legend()
        ax.grid(True,which='major', axis='both')
        return

    def matsu(self,number_iw,beta):
        idx=np.arange(0,number_iw)
        out=[]
        for n in idx:
            out.append((2*n+1)*np.pi/beta)
        return np.array(out)

    def iw_object(self,ax,g_type,marker='o',real=True):
        if real:
            name='Re '+g_type
            x,err=self.err_dict['Re '+g_type]['x'],self.err_dict['Re '+g_type]['err']
        else:
            name='Imag '+g_type
            x,err=self.err_dict['Imag '+g_type]['x'],self.err_dict['Imag '+g_type]['err']

        n_l=err.shape[0]

        if self.errbar:
            item=name
            err_x,mean_y,std_y=self.err_dict[item]['x'][:,0], \
                                self.std_err[item]['mean'],\
                                self.std_err[item]['std']
            ax.errorbar(err_x,mean_y,std_y,label='Method %s'%self.solver2test,capsize=3)        
        else:
            ax.scatter(x,err,marker=marker,label='Method %s'%self.solver2test)
        
        ax.plot([x.max(),x.min()],[0,0],label='ground truth',color='gray')


        ax.legend()
        ax.set_xlabel(r'i$\omega_n$')
        if real:
            ax.set_ylabel('Re {}n({}-{})'.format(g_type,self.ground_truth,self.solver2test))
        else:
            ax.set_ylabel('Im {}n({}-{})'.format(g_type,self.ground_truth,self.solver2test))

        ax.set_ylim(top=np.abs(err).max(),bottom=-np.abs(err).max())

        xticks=[]
        idx=np.linspace(0,n_l,5,dtype = int, endpoint=False)
        for i in idx:
            xticks.append(x[i,0])
        ax.set_xticks(xticks)
        x_ticks=[]
        for i in idx:
            x_ticks.append('%d'%(2*i+1)+r'$\pi$/'+r'$\beta$')
        ax.set_xticklabels(x_ticks)
        
    
    def Z_Mott(self,ax):
        Z_true,Z_test,U=self.err_dict['Mott']['y true'],self.err_dict['Mott']['y test'],self.err_dict['Mott']['U']
        ax.plot(U,Z_true,label='ground truth',color='gray')

        if self.errbar:
            std=np.std(Z_true-Z_test)
            std_=[]
            for item in range(self.number_of_data):
                std_.append(std)
            std_=np.array(std_)

            ax.errorbar(U,Z_test,std_,label='Method %s'%self.solver2test,capsize=3)
        else:
            ax.scatter(U,Z_test,label='Method %s'%self.solver2test)
        ax.set_xlabel('U(eV)')
        ax.set_ylabel('$Z$')
        ax.legend()
        ax.set_ylim(bottom=0,top=1)

    def Aw(self,ax):
        x,err=self.err_dict['Aw']['x'],self.err_dict['Aw']['err']
        ax.plot([x.max(),x.min()],[0,0],label='ground truth',color='gray')

        if self.errbar:
            item='Aw'
            err_x,mean_y,std_y=self.err_dict[item]['x'][:,0], \
                                self.std_err[item]['mean'],\
                                self.std_err[item]['std']
            ax.errorbar(err_x,mean_y,std_y,label='Method %s'%self.solver2test,capsize=3)
        else:
            ax.scatter(x,err,label='Method %s'%self.solver2test)

        ax.set_xlabel('$\omega$')
        ax.set_ylabel('A($\omega$ ({}-{})'.format(self.ground_truth,self.solver2test))
        ax.set_ylim(top=np.abs(err).max(),bottom=-np.abs(err).max())
        ax.legend()

    def plot_mott_transition(self):
        U=np.zeros((self.number_of_data))
        for idx,item in enumerate(self.all_data_dict[self.solver2test]):
            U[idx]=(item['aim params']['U'])
        A_w_ED,A_w_test=self.G_object_y_y_truth(y_arg='A_w')
        A_w_w,x_test=self.G_object_y_y_truth(y_arg='A_w_w')
        fig,ax = plt.subplots(self.number_of_data,1,
                            figsize=(4,int(3*self.number_of_data)),
                            sharex=True,dpi=300
                            )
        for idx,item in enumerate(U):
            ax[idx].plot(A_w_w[:,idx],A_w_ED[:,idx],label='ground truth U=%d'%item,color='gray')
            ax[idx].plot(A_w_w[:,idx],A_w_test[:,idx],label='Method %s U=%d'%(self.solver2test,item))
            ax[idx].set_xlim(-4,4)
            ax[idx].legend()
            ax[idx].set_ylim(bottom=0)
        plt.tight_layout()
        fig.savefig('Mott.png')
        plt.show()

    def plot_(self,ax):
        self.errbar=True
        self.quasi(ax[3,1])
        self.n(ax[4,1])
        self.G_l(ax[0,1])
        self.G_tau(ax[0,0])
        self.iw_object(ax[1,0],g_type='G_iw')
        self.iw_object(ax[1,1],g_type='G_iw',real=False)
        self.iw_object(ax[2,0],g_type='Sigma_iw')
        self.iw_object(ax[2,1],g_type='Sigma_iw',real=False)
        self.Aw(ax[3,0])
        if 'Mott' in self.test_db:
            self.Z_Mott(ax[4,0])
        return ax

    def plot(self):
        fig, ax = plt.subplots(5, 2,figsize=(8,15),dpi=300)
        self.plot_(ax)
        fig.tight_layout()
        fig.savefig('benchmark.png')

        if 'Mott' in self.test_db :
            self.plot_mott_transition()

    def summary(self) -> dict:
        print(20*'#','Summary Benchmark',20*'#')
        print(' Solver test:',self.err_dict['info']['solver test'])
        print('ground truth:',self.err_dict['info']['ground truth'])

        for item in self.err_dict:
            if 'y true' in self.err_dict[item]:
                y_tr=self.err_dict[item]['y true']
                y_te=self.err_dict[item]['y test']
                err=mean_absolute_error(y_tr,y_te)
                print('{:>13.12} mean abs error: {:2f}'.format(item,err))

        print(20*'#','Summary Benchmark end',17*'#')

        return self.err_dict
# we like it j
def example():
    test=benchmark(force2procceed=False,
                test_db='Mott',
                ML_method='PKML1',
            #    AIM_method='IPT'
                )
    test.plot()
    test.summary()

def main():
    parser = argparse.ArgumentParser(description='Anderson Transformer Example')

    parser.add_argument('--db',type=str,
                    help='db argument to be directory in /drive1/ML_DATABASE/..')
    parser.add_argument('--ML-file','-f',type=str,
                        help='for pytorch it is a pth file')
    
    args = parser.parse_args()

    pass

if __name__  =='__main__':
    main()


