from .transform import ToTensor,Squasher,Normalise,Squasher_Y,Normalise_Y,Y_Inverse_Transform

import numpy as np
from torchvision import transforms

def inputs_normalise_factor_xy(root,db,solver,basis,solver_j,basis_j,method,high_ram=False,transform=None) -> dict:
    r"""
    return
    -------
    Squasher factor ,Normalise factor

    sample,_=dataset[0]
    inputs,target=sample

    squasher_factor=min(input)
    normalise_factor=max(Squasher(sample))
    """

    transform=None
    dataset=Train_dataset(root,db,solver,basis,solver_j,basis_j,method,high_ram,transform)
    squasher_factor=1
    squasher_factor_y=1
    for index in range(len(dataset)):
        sample,_=dataset[index]
        inputs,target=sample

        factor=min(min(inputs),0)
        squasher_factor=min(squasher_factor,factor)

        factor_y=min(min(target),0)
        squasher_factor_y=min(squasher_factor_y,factor_y)

    squasher_factor=np.abs(squasher_factor)+1+1e-6
    squasher_factor_y=np.abs(squasher_factor)+1+1e-6

    print(f"{squasher_factor=} {squasher_factor_y=}")

    # find norm
    transform=transforms.Compose([Squasher(squasher_factor),Squasher_Y(squasher_factor_y)])
    dataset=Train_dataset(root,db,solver,basis,solver_j,basis_j,method,high_ram,transform)
    normalise_factor=1
    normalise_factor_y=1
    for index in range(len(dataset)):
        sample,_=dataset[index]
        inputs,target=sample
        assert inputs.any() >=0, 'Squashed inputs should be positive'
        assert target.any() >=0, 'Squashed inputs should be positive'

        normalise_factor=max(max(inputs),normalise_factor)
        normalise_factor_y=max(max(target),normalise_factor_y)

    normalise_factor=np.abs(normalise_factor)+1e-6
    normalise_factor_y=np.abs(normalise_factor_y)+1e-6

    print(f"{normalise_factor=},{normalise_factor_y=}")

    # verify factors
    print(f"verifying transform and Inverse transform")
    transform=transforms.Compose([Squasher(squasher_factor),Squasher_Y(squasher_factor_y),Normalise(normalise_factor),Normalise_Y(normalise_factor_y)])
    inverse_transform=Y_Inverse_Transform(squasher_factor_y, normalise_factor_y)

    transformed_dataset=Train_dataset(root,db,solver,basis,solver_j,basis_j,method,high_ram,transform)
    raw_dataset=Train_dataset(root,db,solver,basis,solver_j,basis_j,method,high_ram,None)

    for index in range(len(transformed_dataset)):
        (inputs,target),_=transformed_dataset[index]
        (_,raw_target),_=raw_dataset[index]
        recovered_transform=inverse_transform(target)
        assert inputs.any() >=0 and inputs.any() <=1., 'Squashed and normed inputs should be positive'
        assert target.any() >=0 and target.any() <=1., 'Squashed and normed inputs should be positive'
        assert np.allclose(raw_target,recovered_transform,atol=1e-5)

    out_dict=dict(squasher_factor=squasher_factor,
            normalise_factor=normalise_factor,
            squasher_factor_y=squasher_factor_y,
            normalise_factor_y=normalise_factor_y,
            )

    return out_dict