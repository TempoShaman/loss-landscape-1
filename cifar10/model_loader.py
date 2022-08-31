import os
import torch, torchvision
import importlib

from cifar10.models import modely
# import cifar10.models.vgg as vgg
import cifar10.models.resnet as resnet
# import cifar10.models.densenet as densenet
from torch.nn.modules.loss import CrossEntropyLoss
# map between model name and function
models = {
    'fewshot'               : modely.FewShotClassifier,
    'fewshot_encoder'               : modely.get_few_shot_encoder,
    'fewshot_maml' : modely.OmniglotNet,
    # 'vgg9'                  : vgg.VGG9,
    # 'densenet121'           : densenet.DenseNet121,
    # 'resnet18'              : resnet.ResNet18,
    # 'resnet18_noshort'      : resnet.ResNet18_noshort,
    # 'resnet34'              : resnet.ResNet34,
    # 'resnet34_noshort'      : resnet.ResNet34_noshort,
    # 'resnet50'              : resnet.ResNet50,
    # 'resnet50_noshort'      : resnet.ResNet50_noshort,
    # 'resnet101'             : resnet.ResNet101,
    # 'resnet101_noshort'     : resnet.ResNet101_noshort,
    # 'resnet152'             : resnet.ResNet152,
    # 'resnet152_noshort'     : resnet.ResNet152_noshort,
    # 'resnet20'              : resnet.ResNet20,
    # 'resnet20_noshort'      : resnet.ResNet20_noshort,
    # 'resnet32_noshort'      : resnet.ResNet32_noshort,
    # 'resnet44_noshort'      : resnet.ResNet44_noshort,
    # 'resnet50_16_noshort'   : resnet.ResNet50_16_noshort,
     'resnet56'              : resnet.ResNet56,
    # 'resnet56_noshort'      : resnet.ResNet56_noshort,
    # 'resnet110'             : resnet.ResNet110,
    # 'resnet110_noshort'     : resnet.ResNet110_noshort,
    # 'wrn56_2'               : resnet.WRN56_2,
    # 'wrn56_2_noshort'       : resnet.WRN56_2_noshort,
    # 'wrn56_4'               : resnet.WRN56_4,
    # 'wrn56_4_noshort'       : resnet.WRN56_4_noshort,
    # 'wrn56_8'               : resnet.WRN56_8,
    # 'wrn56_8_noshort'       : resnet.WRN56_8_noshort,
    # 'wrn110_2_noshort'      : resnet.WRN110_2_noshort,
    # 'wrn110_4_noshort'      : resnet.WRN110_4_noshort,
}

# def load(model_name, model_file=None, data_parallel=False):
#     net = models[model_name]()
#     if data_parallel: # the model is saved in data paralle mode
#         net = torch.nn.DataParallel(net)
#
#     if model_file:
#         assert os.path.exists(model_file), model_file + " does not exist."
#         torch.cuda.empty_cache()
#         stored = torch.load(model_file, map_location=lambda storage, loc: storage)
#         if 'state_dict' in stored.keys():
#             net.load_state_dict(torch.load('/home/oliver/Documents/loss-landscape/cifar10/trained_nets/proto_weights.41-1.02.t7'),
#             strict=False)
#         else:
#             net.load_state_dict(torch.load('/home/oliver/Documents/loss-landscape/cifar10/trained_nets/proto_weights.41-1.02.t7'),
#             strict=False)
#
#     if data_parallel: # convert the model back to the single GPU version
#         net = net.module
#
#     net.eval()
#     return net

def load(model_name, model_file=None, data_parallel=False):
    print("loading of net started")

    net = modely.OmniglotNet(num_classes=20, loss_fn=CrossEntropyLoss)
    #net = models[model_name](num_input_channels = 1)
    print("done")
    print(model_name)
    print(net)
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(models.model_name)

    if model_file:

        assert os.path.exists(model_file), model_file + " does not exist."

        model_fewshot_weights = torch.load(model_file, map_location=lambda storage, loc: storage)
        net.load_state_dict(model_fewshot_weights)
        #
        # stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        # if 'state_dict' in stored.keys():
        #     net.load_state_dict(stored['state_dict'])
        # else:
        #     net.load_state_dict(stored)

    if data_parallel:  # convert the model back to the single GPU version
        net = net.module

    assert net is not None
    net.eval()
    return net


