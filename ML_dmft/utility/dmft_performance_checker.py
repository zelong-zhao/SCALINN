import numpy as np
import contextlib
import shutil 
import os
import warnings
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
matplotlib.style.use(plt.style.available[-2])
from matplotlib.backends.backend_pdf import PdfPages

class Monitor_Dmft_Performance:
    def __init__(self,outname,CWD='./') -> None:
        self.outname = outname
        self.CWD=CWD
        self.init_record_hyb_pdf(outname)
    
    def init_record_hyb_pdf(self,outname):
        ############################
        outpdf_name=f"./{outname}.pdf"
        outpdf_name=os.path.join(self.CWD,f"./{outname}.pdf")
        print(f"out pdf name:{outpdf_name}")
        if os.path.isfile(outpdf_name):
            os.remove(outpdf_name)
        self.pp = PdfPages(outpdf_name)

        ############################
        self.out_dir=os.path.join(self.CWD,'./out_imag/')
        print(f'dumping to {self.out_dir}')
        if os.path.isdir(self.out_dir): 
            shutil.rmtree(self.out_dir)
        os.mkdir(self.out_dir)
        self.fig_idx=0
    
    def init_record_hyb_param(self):
        out_csv=os.path.join(self.CWD,f"{self.outname}_param.csv")
        print(f"{out_csv=}")
        if os.path.isfile(out_csv):
            os.remove(out_csv)
        self.out_csv=out_csv
        
    def record_hyb_param(self,params):
        with open(self.out_csv, "a") as f:
            np.savetxt(f, params, delimiter=",", fmt="%1.6f",)

    def init_oneframe(self)->None:
        self.fig, self.ax = plt.subplots(1,1,figsize=(4,3),dpi=100)

    def plot_oneframe(self,mesh:np.ndarray,G:np.ndarray,leg_name:str):
        if not np.iscomplex(G).all():
            raise ValueError('Inputs Must be Complex')
        mesh,G = deepcopy(mesh),deepcopy(G)
        mesh,G = mesh[:32],G[:32]
        self.ax.plot(mesh,G.imag,label=leg_name)

    def finish_oneframe(self,ylim_bottom=-2.0):
        self.ax.set_ylim(ylim_bottom,0)
        self.ax.set_xlabel('omega_n')
        self.ax.set_ylabel('G(iwn)')
        self.ax.set_title(f'loop: {self.fig_idx}')
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.savefig(os.path.join(self.out_dir,f"loop_{self.fig_idx}.png"))

        self.pp.savefig(self.fig)
        plt.close(self.fig)
        self.fig_idx +=1

    def finish_allframe(self):
        self.pp.close()