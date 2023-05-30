# SCALINN package 

Strongly Correlated Approach with Language Inspired Neural Network (SCALINN) is a Python package that uses nueron network to simulate Dynamical Mean-Field Theory (DMFT). 

## Installation 

```bash
bash ./install.sh dmft_env
```

### Optional Package

SCALINN provides an interface for the Exact Diagonalization (ED) method of solving the Anderson Impurity Model (AIM). This ED method implementation can be found in an external library, and it needs to be installed manually from this GitHub repository: https://github.com/francoisjamet/ed_solver. As well as HubbardI solver, the kernal is refered to https://triqs.github.io/hubbardI/latest/install.html.

It's important to note that these libraries need to be installed separately and properly configured to ensure the full functionality of the SCALINN package. The library's documentation provides detailed instructions on how to do this.

## Tutorial 

The SCALINN package includes comprehensive tutorials located in the main directory. The tutorials cover two major components of using the software: 

1. **Generating the Database**: The first tutorial guides you through the process of creating a database suitable for training the neural network. This involves preparing and structuring your data in a way that can be efficiently processed by the model. 

2. **Training the Model**: The second tutorial explains how to train your neural network using the prepared database. It provides instructions on setting up the training parameters, running the training process, and evaluating the model's performance.

Moreover, the package comes with a pre-trained Transformer model, which allows you to get started quickly without having to train a model from scratch. This can be especially useful for understanding how the Transformer model works or for testing it on your own data.


## Author Info.

Any issues please contact with zelongzhao@hotmail.com


# LICENSE
GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
