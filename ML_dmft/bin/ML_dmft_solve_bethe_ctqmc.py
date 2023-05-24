#! /usr/bin/env python

import argparse,warnings
import triqs.gf as gf
import triqs.operators as op
from h5 import HDFArchive
from triqs_cthyb import Solver
import triqs.utility.mpi as mpi
import os, sys
import numpy as np
import contextlib

OUT_DIR = 'ctqmc_out'
n_iw = 1024
w_c = 32


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def read_args():
    parser = argparse.ArgumentParser(description='CTQMC Bethe Lattice Solver')

    parser.add_argument('-t', type=float, default=1., metavar='f',
                        help='semicircular length (default: 1.)')
    parser.add_argument('-U', type=float, metavar='f',required=True,
                        help='Hubbard U')
    parser.add_argument('-beta', type=float, default=10., metavar='f',
                        help='temerature 1/eV (default: 10.)')
    parser.add_argument('-alpha', type=float, default=0.2)
    parser.add_argument('-n_loops', type=int, default=100., metavar='f',
                        help='n dmft loops')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--OUT_DIR', type=str, default='ctqmc_out', metavar='F',
                        help='OUT_DIR (default: ctqmc_out) ')
    args = parser.parse_args()
    return args


def run_bethe_dmft(U:float,
                    beta:float=10,
                    t:float=1.0,
                    n_loops:int = 10,
                    alpha:float =0.2,
                    OUT_DIR='ctqmc_out',
                    dry_run=False
                    )->None:
    r"""
    args
    ----
        U: hubbard U (eV)
        beta: temerature 1/eV
        t: semicircular length
        n_loops: # dmft loops
    """

    # Construct the impurity solver
    S = Solver(beta = beta, 
            gf_struct = [('up',[0]), ('down',[0])],
            n_iw=n_iw,
            )
    W=t
    # This is a first guess for G
    # S.G_iw << t**2/4.*gf.SemiCircular(t)
    delta_iw = S.G_iw['up'].copy()
    delta_iw << t**2/4.*gf.SemiCircular(t)
    # DMFT loop with self-consistency
    for i in range(n_loops):

        # Symmetrize the Green's function and use self-consistency
        for name, g0 in S.G0_iw:
            g0 << gf.inverse( gf.iOmega_n + U/2.0 - delta_iw)

        if dry_run:
            solver_params=dict( n_cycles  = 5000,                      # Number of QMC cycles
                                length_cycle = 200,                      # Length of one cycle
                                n_warmup_cycles = 1000,                 # Warmup cycles
                                )
        else:
            solver_params=dict( n_cycles  = 500000,                      # Number of QMC cycles
                    length_cycle = 200,                      # Length of one cycle
                    n_warmup_cycles = 10000,                 # Warmup cycles
                    )


        S.solve(h_int = U * op.n('up',0) * op.n('down',0),   # Local Hamiltonian 
                **solver_params)

        
        g = 0.5 * ( S.G_iw['up'] + S.G_iw['down'] )
        delta_new_iw = delta_iw.copy()
        delta_new_iw << ((t**2)/4.)*g
        delta_iw << alpha * delta_new_iw + (1-alpha) * delta_iw

        # Save iteration in archive
        if mpi.is_master_node():
            outfile_name = os.path.join('./',OUT_DIR,'ctqmc_out.h5')
            print(f"writing to {outfile_name}")
            with HDFArchive(outfile_name) as A:
                A['G-%i'%i] = S.G_iw['up']
                A['G_tau-%i'%i] = S.G_tau['up']
                A['Sigma-%i'%i] = S.Sigma_iw['up']

        err=max(np.abs(delta_iw.data[:,0,0].imag[n_iw:n_iw+w_c]-delta_new_iw.data[:,0,0].imag[n_iw:n_iw+w_c]))
        if mpi.is_master_node():
            print(f"{50*'#'}")
            print(5*'\n')
            print("Iteration = %i / %i" % (i+1, n_loops))
            print(f"{err=}")
            print(5*'\n')
            print(f"{50*'#'}")
        if err < 1.0e-5:
            break

    if mpi.is_master_node():

        iw_obj={'Sigma_iw':S.Sigma_iw['up'].data[:,:,0],
                    'G0_iw':S.G0_iw['up'].data[:,:,0],
                    'G_iw':S.G_iw['up'].data[:,:,0]
                    }

        for key in iw_obj:
                np.savetxt(os.path.join(OUT_DIR,f'{key}.dat'),iw_obj[key][n_iw:2*n_iw],fmt='%20.10e \t %20.10e',delimiter='\t')

        G_tau = S.G_tau['up'].data[:,:,0].real.flatten()
        np.savetxt(os.path.join(OUT_DIR,f'G_tau.dat'),G_tau)

if __name__ == '__main__':

    args = read_args()


    if mpi.is_master_node():
        if not os.path.exists(args.OUT_DIR):
            os.makedirs(args.OUT_DIR)

    # Parameters of the model
    t = args.t
    beta = args.beta
    U=args.U
    n_loops = args.n_loops
    alpha = args.alpha
    run_bethe_dmft(U=U,beta=beta,n_loops=n_loops,t=t,dry_run=args.dry_run)