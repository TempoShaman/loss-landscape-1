import os
import cifar10.model_loader

def load(dataset, model_name, model_file, data_parallel=False):
    print(dataset)
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
        return net
    elif dataset == 'miniimagenet':
        net = cifar10.model_loader.load(model_name , model_file , data_parallel)
        return net
    elif dataset == 'omniglot':
        net = cifar10.model_loader.load(model_name , model_file , data_parallel)

        return net

