"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
import os
import sys
from multiprocessing import freeze_support

from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


from few_shot.datasets import MiniImageNet
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH


setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='miniImageNet')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=5, type=int)
parser.add_argument('--k-test', default=1, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
args = parser.parse_args()



evaluation_episodes = 1000
episodes_per_epoch = 100


if args.dataset == 'miniImageNet':
    n_epochs = 100
    dataset_class = MiniImageNet
    num_input_channels = 3
    drop_lr_every = 20
else:
    raise(ValueError, 'Unsupported dataset')



param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_'\
f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'



# print(param_str)

###################
# Create datasets #
###################
def background():
    
    background = dataset_class('images_background')
    print("background images are loaded")
    background_taskloader = DataLoader(
        background,
        batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
        num_workers=0
    )
    return background_taskloader


def evaluation():
    evaluation = dataset_class('images_evaluation')
    print("eval images are loaded")

    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
        num_workers=0
    )
    return evaluation_taskloader


#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.double)


############
# Training #
############
print(f'Training Prototypical network on {args.dataset}...')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation(),
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/proto_nets/{param_str}.h5',
        monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc',

    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
]




if __name__ == '__main__':
    print("start")
    fit(
        model,
        optimiser,
        loss_fn,
        epochs=n_epochs,
        dataloader=background(),
        prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                             'distance': args.distance},

    )
    torch.save(model.state_dict(), '/home/oliver/Documents/fewshot/fewshot/data/miniImageNet/models/proto_nets'
                                   '/model_weights.h5')
    freeze_support()
    print("start")
