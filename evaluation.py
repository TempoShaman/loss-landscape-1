
from few_shot.eval import evaluate
from few_shot.datasets import MiniImageNet
from few_shot.core import NShotTaskSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
from torch.autograd.variable import Variable
from torch.nn.modules import Module
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.utils import pairwise_distances2
import dataloader
import few_shot
from few_shot.utils import pairwise_distances
from few_shot.core import prepare_nshot_task


def count_correct(pred, target):
    ''' count number of correct classification predictions in a batch '''
    pairs = [int(x==y) for (x, y) in zip(pred, target)]
    return sum(pairs)


loss_fn = nn.CrossEntropyLoss().cuda()
def forward_pass(net, in_, target):
    #optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    ''' forward in_ through the net, return loss and output '''

    input_var = Variable(in_).cuda()
    target_var = Variable(target).cuda()
    #optimizer.zero_grad()

    # input_var = input_var.cuda()
        # target_var = target_var.cuda()
    out = net.net_forward(input_var)
        #out = out.softmax(dim=1)
    # print('size of input: ', out)
    # print('size of target: ', target_var)
    loss = loss_fn(out, target_var)
    #loss.backward()
    #optimizer.step()
    return loss, out

def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)
    return class_prototypes
def eval_proto_loss(net, loader):
    net.eval()
    k_way = 5
    n_shot = 1
    distance = 'l2'
    num_correct = 0
    loss = 0
    x, y = prepare_nshot_task(1, 20, 1)
    with torch.no_grad():
        for i, (in_, y) in enumerate(loader):
            # print(in_.size())
            # print("_______________________________________")
            # print(y)
            batch_size = in_.numpy().shape[0]

            inputs = Variable(in_)
            targets = Variable(y)
            inputs = inputs.type(torch.FloatTensor)
            #print(type(inputs))

            outputs = net(inputs)



            # print('////////////////////')
            # print(outputs.size())
            # print('////////////////////')
            support = outputs[:n_shot * k_way]
            queries = outputs[n_shot * k_way:]
            target = targets[n_shot * k_way:]
            # print("_______________________________________")
            # print(support.size())
            # print('=======================================')
            # print(queries.size())
            # print("_______________________________________")

            prototypes = compute_prototypes(support, k_way, n_shot)
            # print('+++++++++++++++++++++++++++++')
            # print(l)
            # print('+++++++++++++++++++++++++++++')

            #print(prototypes.size())

            distances = pairwise_distances(queries, prototypes, distance)
            #print(distances.size())

            log_p_y = (-distances).log_softmax(dim=1)
            loss = loss_fn(log_p_y, target)

            # Prediction probabilities are softmax over distances
            y_pred = (-distances).softmax(dim=1)

            loss += loss.data.cpu().numpy()
            num_correct += count_correct(np.argmax(y_pred.data.cpu().numpy(), axis=1), target.numpy())
    return float(loss) / len(loader), float(num_correct) / (len(loader)*batch_size)


def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.
    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.NLLLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total
def evaluate_xd(net, loader):
    ''' evaluate the net on the data in the loader '''
    net.cuda()

    num_correct = 0
    loss = 0
    with torch.no_grad():
        for i, (in_, target) in enumerate(loader):
            batch_size = in_.numpy().shape[0]
            l, out = forward_pass(net.cuda(), in_, target)
            # print('+++++++++++++++++++++++++++++')
            # print(l)
            # print('+++++++++++++++++++++++++++++')
            loss += l.data.cpu().numpy()
            num_correct += count_correct(np.argmax(out.data.cpu().numpy(), axis=1), target.numpy())
    return float(loss) / len(loader), float(num_correct) / (len(loader)*batch_size)
