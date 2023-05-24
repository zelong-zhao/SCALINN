import torch

optimizer_list=['sgd','adagrad','adam','adadelta']

def optimizer_interface(args,model):
    optimizer_method = str(args.optimizer_args[0])
    lr = float(args.optimizer_args[1])
    momentum = float(args.optimizer_args[2])
    weight_decay = float(args.optimizer_args[3])

    optimizer_dict=dict(optimizer_method=optimizer_method,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    )
    long_text=f"""
{50*'#'}
optimizer_dict:

{optimizer_dict=}

aviliable optimizer {optimizer_list}
{50*'#'}
"""
    print(long_text)
    if optimizer_method.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)

    elif optimizer_method.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(),lr=lr,weight_decay=weight_decay)

    elif optimizer_method.lower() == 'adam':
        print(*optimizer_dict)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

    elif optimizer_method.lower() == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(),lr=lr,weight_decay=weight_decay)
    else:
        raise ValueError('not such optimizer method')
    return optimizer