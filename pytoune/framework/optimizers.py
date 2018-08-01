import torch.optim as optim

all_optimizers_dict = dict(
    adadelta=optim.Adadelta,
    adagrad=optim.Adagrad,
    adam=optim.Adam,
    sparseadam=optim.SparseAdam,
    adamax=optim.Adamax,
    asgd=optim.ASGD,
    lbfgs=optim.LBFGS,
    rmsprop=optim.RMSprop,
    rprop=optim.Rprop,
    sgd=optim.SGD,
)

def get_optimizer(optimizer, module):
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        params = (p for p in module.parameters() if p.requires_grad)
        if optimizer != 'sgd':
            return all_optimizers_dict[optimizer](params)

        return all_optimizers_dict[optimizer](params, lr=1e-2)

    return optimizer
