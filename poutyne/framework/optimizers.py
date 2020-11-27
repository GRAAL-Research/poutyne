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
    if isinstance(optimizer, (str, dict)):
        kwargs = {}
        if isinstance(optimizer, dict):
            kwargs = optimizer
            optimizer = optimizer.pop('optim')

        optimizer = optimizer.lower()

        if optimizer == 'sgd':
            kwargs.setdefault('lr', 1e-2)

        params = (p for p in module.parameters() if p.requires_grad)
        return all_optimizers_dict[optimizer](params, **kwargs)

    return optimizer
