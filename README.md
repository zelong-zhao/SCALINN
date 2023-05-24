# SIAM Transformer

SIAM Transformer is a python package to simulate DMFT with Machine Learning code.


## Installation 

```bash
bash ./install.sh dmft_env
```

### Optional Package

A interface of Exact Diagolisation (ED) code of solving Anderon impurity model (AIM) is given in the libaray. However, the kernal need to be installed manually via https://github.com/francoisjamet/ed_solver. As well as HubbardI solver, the kernal is refered to https://triqs.github.io/hubbardI/latest/install.html .


## Tutorial 

Tutorial is given in main direcory. This code including two main parts. Gerenating database and fitting the database with neuron network. The instruction on generating database in tutorial 1 and the instruction on training the database is in tutorial 2. Furthermore, a train Transformer is also attached.


## Author Info.

any issues please contact with zelongzhao@hotmail.com


# LICENSE
GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.