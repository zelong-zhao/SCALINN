import os
from ML_dmft.database.dataset import AIM_dataset_meshG
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns

matplotlib.style.use(plt.style.available[-2])

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

line_color=sns.color_palette("tab10")


class plot_tool():
    def __init__(self,root='./',db='./',solver='ED_KCL_7',index=0) -> None:
        self.root=root
        self.db=db
        self.solver=solver
        self.index=index
        print(f"{index}")
        
    def plot_iw_obj(self,basis='G_iw'):
        num2p=32
        min_iw=np.inf

        print(f"{self.root=}")
        print(f"{self.db=}")
        print(f"{self.solver=}")

        Loaded_data=AIM_dataset_meshG(root=self.root,
                db=self.db,
                solver=self.solver)

        (mesh,Iw_Obj),(_,_)=Loaded_data(self.index,basis)

        fig,ax = plt.subplots(dpi=300)
        ax.plot(mesh[:num2p],Iw_Obj[:num2p].imag,
                markersize=2)
        if min(Iw_Obj[:num2p].imag) < min_iw:
            min_iw = min(Iw_Obj[:num2p].imag)

        ax.legend(prop={'size': 12})
        if 'g_iw' in basis.lower(): 
            ax.set_ylim(min_iw*1.1,0)
            ax.set_yticks(np.linspace(0,min_iw*1.1,5,endpoint=False))
            ax.set_ylabel('ImG(i$\omega_n$)')

        fig.tight_layout()
        print('saving G_iw.png')
        fig.savefig('G_iw.png')
        return fig