
"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
import os
import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
from script.task import OmniglotTask, MNISTTask
import numpy as np
import torchvision
import torch.nn as nn
from torch.optim import Adam
from few_shot.proto import proto_net_episode
import dataloader
import evaluation
import projection as proj
import net_plotter
import plot_2D
from few_shot.train import fit
import plot_1D
import model_loader
from torch.nn.modules.loss import CrossEntropyLoss
import scheduler
import mpi4pytorch as mpi
from dataloader import get_data_loader
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.utils.data import DataLoader
from few_shot.eval import evaluate
from few_shot.datasets import MiniImageNet
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from cifar10.models import modely
from dataloader import OmniglotDataset, MiniImageNet
from few_shot.models import get_few_shot_encoder




def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):

    # skip if the direction file already exists
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if os.path.exists(surf_file):

        f = h5py.File(surf_file, 'r')
        #with h5py.File(surf_file, 'r') as f:
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)

            return


    f = h5py.File(surf_file, 'a')
    #with h5py.File(surf_file, 'a') as f:

    f['dir_file'] = dir_file

    xnum = int(args.xnum)
    ynum = int(args.ynum)
    # Create the coordinates(resolutions) at which the function is evaluated
    print("|||||||||||||||||||||||||||||||")
    print("_______________________________")
    print(args.xmin)
    print(type(args.xmin))
    print("_______________________________")
    print(args.xmax)
    print(type(args.xmax))
    print("_______________________________")
    print(args.xnum)
    print(type(args.xnum))
    print("_______________________________")
    print("|||||||||||||||||||||||||||||||")
    xcoordinates = np.linspace(args.xmin, args.xmax, num=xnum)
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=ynum)
        f['ycoordinates'] = ycoordinates
        assert len(f['ycoordinates']) != 0
        print(len(f['ycoordinates']))
    f.close()



    return surf_file





def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args, very_total_time=0):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    #losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0




    if args.loss_name == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_name == 'nllloss':
        criterion = nn.NLLLoss()
    elif args.loss_name == 'crossentropyloss':
        criterion = nn.CrossEntropyLoss()


    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]


        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()

        #loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda)
        #loss, acc = evaluation.evaluate(net, dataloader)
        #loss, acc = evaluation.eval_proto_loss(net, dataloader)
        loss, acc = evaluation.evaluate_xd(net, dataloader)
        # loss, acc = loss_fn(model_output, target=y,
        #                     n_support=opt.num_support_val)



        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses     = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))
        very_total_time = float((very_total_time + time.time() - start_time) / (count + 1))
        est = len(inds) - count
        hodiny: float = float(very_total_time / 60.0 / 60.0 * est)

        #hodiny = "{:.2f}".format(hodiny)

        print(very_total_time)
        dni = hodiny / 24.0
        minuty = hodiny * 60.0
        if float(hodiny) > 24:
            print('Approximate time left is: %6.2f days' % dni)
        elif 0 < float(hodiny) < 24:
            print('Approximate time left is: %6.2f  hours' % hodiny)
        elif float(hodiny) < 0:
            print('Approximate time left is: %6.2f  minutes' % minuty)

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))



    f.close()


###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=30, type=int, help='minibatch size')

    # data parameters
    parser.add_argument('--dataset', default='omniglot', help='cifar10 | miniimagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    # model parameters
    parser.add_argument('--model', default='fewshot', help='model name')
    parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--loss_name', '-l', default='nllloss', help='loss functions: crossentropyloss | mse | nllloss')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=True, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=True, help='plot figures after computation')

    args = parser.parse_args()

    torch.manual_seed(123)
    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
              (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------


    #net = modely.OmniglotNet(20, loss_fn, 1)
    net = model_loader.load(args.dataset, args.model, args.model_file)


    #assert net is not None, 'Net in plot_surface is empty'
    print(net)
    w = net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references


    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = net_plotter.name_direction_file(args) # name the direction file
    if rank == 0:
        net_plotter.setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # wait until master has setup the direction file and surface file
    mpi.barrier(comm)

    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    #--------------------------------------------------------------------------
    # Setup dataloader
    #--------------------------------------------------------------------------
    # download CIFAR10 if it does not exit

    def get_task(root, n_cl, n_inst, split='train'):
        if 'mnist' in root:
            return MNISTTask(root, n_cl, n_inst, split)
        elif 'omniglot' in root:
            return OmniglotTask(root, n_cl, n_inst, split)
        else:
            print ('Unknown dataset')
            raise(Exception)

    # exp = 'maml-omniglot-20way-1shot'

    dataset = 'omniglot'
    # num_cls = 20
    # num_inst = 1
    # batch = 5
    # m_batch = 16
    # num_updates = 25000
    # num_inner_updates = 5
    # lr = '1e-1'
    # meta_lr = '1e-3'
    # gpu = 0
    #
    num_cls = 5
    num_inst = 1
    batch = 1
    m_batch = 32
    num_updates = 15000
    num_inner_updates = 5

    #task = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst, split='test')
    task = get_task('../data/{}'.format(dataset), num_cls, num_inst)
    train_loader = get_data_loader(task, batch, split='train')

    mpi.barrier(comm)
    #w = w.cuda()

    dataset_class = OmniglotDataset
    background = dataset_class('images_background')
    # trainloader = DataLoader(
    #     background,
    #     batch_sampler=NShotTaskSampler(background, 100, 1, 5, 5),
    #     num_workers=4
    # )

    trainloader = get_data_loader(task, batch, split='train')
    # trainloader, testloader = dataloader.load_dataset(args.dataset, args.datapath,
    #                                                   args.batch_size, args.threads, args.raw_data,
    #                                                   args.data_split, args.split_idx,
    #                                                   args.trainloader, args.testloader)

    # trainloader, testloader = dataloader.load_dataset(args.dataset, args.datapath,
    #                             args.batch_size, args.threads, args.raw_data,
    #                             args.data_split, args.split_idx,
    #                             args.trainloader, args.testloader)

    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    crunch(surf_file, net, w, s, d, trainloader, 'train_loss', 'train_acc', comm, rank, args)
    #crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    if args.plot and rank == 0:
        if args.y and args.proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
        elif args.y:
            plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
        else:
            plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
# #mpirun -n 4 python3 plot_surface.py --mpi --cuda --model fewshot --x=-1:1:51 --y=-1:1:51 --model_file /home/oliver/Documents/loss-landscape/cifar10/trained_nets/proto_weights.41-1.02 (1).t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot