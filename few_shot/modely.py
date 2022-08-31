from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from typing import Dict
from collections import OrderedDict
import math
##########
# Layers #
##########
class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].

    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalMaxPool1d(nn.Module):
    """Performs global max pooling over the entire length of a batched 1D tensor

    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return nn.functional.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool2d(nn.Module):
    """Performs global average pooling over the entire height and width of a batched 2D tensor

    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def functional_conv_block(x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor,
                          bn_weights, bn_biases) -> torch.Tensor:
    """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.

    # Arguments:
        x: Input Tensor for the conv block
        weights: Weights for the convolutional block
        biases: Biases for the convolutional block
        bn_weights:
        bn_biases:
    """
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


##########
# Models #
##########

class OmniglotNet(nn.Module):
    '''
    The base model for few-shot learning on Omniglot
    '''

    def __init__(self, num_classes, loss_fn, num_in_channels=3):
        super(OmniglotNet, self).__init__()
        # Define the network

        # self.layer1 = nn.Sequential(nn.Conv2d(num_in_channels, 64, 3),
        #                             nn.BatchNorm2d(64, momentum=0.1, affine=True),
        #                             nn.ReLU(inplace=True),
        #                             nn.MaxPool2d(2, 2))
        #
        # self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3),
        #                             nn.BatchNorm2d(64, momentum=0.1, affine=True),
        #                             nn.ReLU(inplace=True),
        #                             nn.MaxPool2d(2, 2))
        # self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3),
        #                             nn.BatchNorm2d(64, momentum=0.1, affine=True),
        #                             nn.ReLU(inplace=True),
        #                             nn.MaxPool2d(2, 2))

        # self.features = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(num_in_channels, 64, 3)),
        #     ('bn1', nn.BatchNorm2d(64, momentum=0.1, affine=True)),
        #     ('relu1', nn.ReLU(inplace=True)),
        #     #('dropout0', nn.Dropout2d(0.0)),
        #     ('pool1', nn.MaxPool2d(2, 2)),
        #     ('conv2', nn.Conv2d(64, 64, 3)),
        #     ('bn2', nn.BatchNorm2d(64, momentum=0.1, affine=True)),
        #     ('relu2', nn.ReLU(inplace=True)),
        #     #('dropout1', nn.Dropout2d(0.0)),
        #     ('pool2', nn.MaxPool2d(2, 2)),
        #     ('conv3', nn.Conv2d(64, 64, 3)),
        #     ('bn3', nn.BatchNorm2d(64, momentum=0.1, affine=True)),
        #     ('relu3', nn.ReLU(inplace=True)),
        #     #('dropout2', nn.Dropout2d(0.0)),
        #     ('pool3', nn.MaxPool2d(2, 2))
        # ]))

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(num_in_channels, 64, 3)),
            ('bn1', nn.BatchNorm2d(64, momentum=0.1, affine=True, track_running_stats=False)),
            ('relu1', nn.ReLU(inplace=True)),
            ('dropout0', nn.Dropout2d(0.0)),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('bn2', nn.BatchNorm2d(64, momentum=0.1, affine=True, track_running_stats=False)),
            ('relu2', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout2d(0.0)),
            ('pool2', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(64, 64, 3)),
            ('bn3', nn.BatchNorm2d(64, momentum=0.1, affine=True, track_running_stats=False)),
            ('relu3', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout2d(0.0)),
            ('pool3', nn.MaxPool2d(2, 2))
        ]))
        self.add_module('fc', nn.Linear(64, 5))

        self.loss_fn = loss_fn

        # Initialize weights
        #self._init_weights()

    def net_forward(self, x):
        return self.forward(x)

    # def forward(self, x):
    #     ''' Define what happens to data in the net '''
    #     if weights == None:
    #         x = self.features(x)
    #         x = x.view(x.size(0), 64)
    #         x = self.fc(x)
    #     else:
    #         x = conv2d(x, weights['features.conv1.weight'], weights['features.conv1.bias'])
    #         x = batchnorm(x, weight=weights['features.bn1.weight'], bias=weights['features.bn1.bias'], momentum=1)
    #         x = relu(x)
    #         x = maxpool(x, kernel_size=2, stride=2)
    #         x = conv2d(x, weights['features.conv2.weight'], weights['features.conv2.bias'])
    #         x = batchnorm(x, weight=weights['features.bn2.weight'], bias=weights['features.bn2.bias'], momentum=1)
    #         x = relu(x)
    #         x = maxpool(x, kernel_size=2, stride=2)
    #         x = conv2d(x, weights['features.conv3.weight'], weights['features.conv3.bias'])
    #         x = batchnorm(x, weight=weights['features.bn3.weight'], bias=weights['features.bn3.bias'], momentum=1)
    #         x = relu(x)
    #         x = maxpool(x, kernel_size=2, stride=2)
    #         x = x.view(x.size(0), 64)
    #         x = linear(x, weights['fc.weight'], weights['fc.bias'])
    #     return x
    #
    # def _init_weights(self):
    #     ''' Set weights to Gaussian, biases to zero '''
    #     torch.manual_seed(1337)
    #     torch.cuda.manual_seed(1337)
    #     torch.cuda.manual_seed_all(1337)
    #     print('init weights')
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 0.01)
    #             # m.bias.data.zero_() + 1
    #             m.bias.data = torch.ones(m.bias.data.size())
    #
    # def copy_weights(self, net):
    #     ''' Set this module's weights to be the same as those of 'net' '''
    #     # TODO: breaks if nets are not identical
    #     # TODO: won't copy buffers, e.g. for batch norm
    #     for m_from, m_to in zip(net.modules(), self.modules()):
    #         if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
    #             m_to.weight.data = m_from.weight.data.clone()
    #             if m_to.bias is not None:
    #                 m_to.bias.data = m_from.bias.data.clone()

    def forward(self, x):
        ''' Define what happens to data in the net '''
        x = self.features(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        #x = F.avg_pool2d(x, 1)
        #out = F.avg_pool2d(out, 4)
        x = x.view(x.size(0), 64)
        x = self.fc(x)

        return x


def get_few_shot_encoder(num_input_channels=1, hidden=64, final_layer_size=1600) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3

    """
    return nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        Flatten(),
    )

    # return nn.Sequential(
    #
    #     nn.Conv2d(in_channels=num_input_channels, out_channels=hidden, kernel_size=(7, 7), stride=(2, 2),
    #               padding=(3, 3), bias=False),
    #     #nn.BatchNorm2d(num_features=hidden),
    #     # nn.MaxPool2d(kernel_size=2),
    #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
    #     nn.BatchNorm2d(num_features=hidden,momentum=0.1, track_running_stats=False),
    #
    #     nn.Conv2d(in_channels=hidden, out_channels=int(hidden * 1.5), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
    #               bias=False),
    #     #nn.BatchNorm2d(num_features=int(hidden * 1.5)),
    #     #nn.MaxPool2d(kernel_size=2),
    #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #     nn.BatchNorm2d(num_features=int(hidden * 1.5),momentum=0.5, track_running_stats=False),
    #     nn.MaxPool2d(kernel_size=2),
    #     #nn.Dropout2d(0.3),
    #
    #     nn.Conv2d(in_channels=int(hidden * 1.5), out_channels=hidden * 2, kernel_size=(3, 3), stride=(2, 2),
    #               padding=(1, 1), bias=True),
    #     #nn.BatchNorm2d(num_features=hidden * 2),
    #     #nn.MaxPool2d(kernel_size=2),
    #     nn.LeakyReLU(negative_slope=0.3, inplace=True),
    #     nn.BatchNorm2d(num_features=hidden * 2,momentum=0.5, track_running_stats=False),
    #     nn.MaxPool2d(kernel_size=2),
    #     nn.Dropout2d(0.0),
    #
    #     nn.Conv2d(in_channels=int(hidden * 2), out_channels=hidden * 4, kernel_size=(3, 3), stride=(2, 2),
    #               padding=(1, 1), bias=True),
    #     #nn.BatchNorm2d(num_features=hidden * 4),
    #     #nn.MaxPool2d(kernel_size=2),
    #     nn.LeakyReLU(negative_slope=0.3, inplace=True),
    #     nn.BatchNorm2d(num_features=hidden * 4,momentum=0.5, track_running_stats=False),
    #     nn.MaxPool2d(kernel_size=2),
    #     # nn.Dropout2d(0.0))
    #     nn.Flatten(),
    #     nn.Linear(hidden * 4, final_layer_size, bias=False)
    #     # Flatten()
    # )


# def get_few_shot_encoder(num_input_channels=3, hidden: int = 64, final_layer_size: int = 1600) -> nn.Module:
#     """Creates a few shot encoder as used in Matching and Prototypical Networks
#
#     # Arguments:
#         num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
#             miniImageNet = 3
#     """
#     return nn.Sequential(
#         conv_block(num_input_channels, 64),
#         conv_block(64, 64),
#         conv_block(64, 64),
#         conv_block(64, 64),
#         Flatten(),
#     )


    # return nn.Sequential(
    #     nn.Sequential(nn.Conv2d(in_channels=num_input_channels, out_channels=hidden, kernel_size=(7, 7), stride=(2, 2),
    #                             padding=(3, 3), bias=False),
    #                   nn.BatchNorm2d(num_features=hidden),
    #                   #nn.MaxPool2d(kernel_size=2),
    #                   nn.LeakyReLU(negative_slope=0.1, inplace=True)),
    #
    #     nn.Sequential(nn.Conv2d(in_channels=hidden, out_channels=int(hidden * 1.5), kernel_size=(3, 3), stride=(2, 2),
    #                             padding=(1, 1), bias=False),
    #                   nn.BatchNorm2d(num_features=int(hidden * 1.5)),
    #                   nn.MaxPool2d(kernel_size=2),
    #                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #                   nn.Dropout2d(0.0)),
    #
    #     nn.Sequential(
    #         nn.Conv2d(in_channels=int(hidden * 1.5), out_channels=hidden * 2, kernel_size=(3, 3), stride=(2, 2),
    #                   padding=(1, 1), bias=True),
    #         nn.BatchNorm2d(num_features=hidden * 2),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.LeakyReLU(negative_slope=0.3, inplace=True),
    #         nn.Dropout2d(0.0)),
    #
    #     nn.Sequential(nn.Conv2d(in_channels=hidden * 2, out_channels=final_layer_size, kernel_size=(3, 3), stride=(2, 2),
    #                             padding=(1, 1), bias=True),
    #                   nn.BatchNorm2d(num_features=hidden * 4),
    #                   nn.MaxPool2d(kernel_size=2),
    #                   nn.LeakyReLU(negative_slope=0.3, inplace=True),
    #                   nn.Dropout2d(0.0)),
    #     Flatten(),
    #
    #     # def forward(self, x):
    #     #     out = F.relu(self.bn1(self.conv1(x)))
    #     #     out = self.layer1(out)
    #     #     out = self.layer2(out)
    #     #     out = self.layer3(out)
    #     #     out = self.layer4(out)
    #     #     out = F.avg_pool2d(out, 4)
    #     #     out = out.view(out.size(0), -1)
    #     #     out = self.linear(out)
    #     #     return out
    # )


class FewShotClassifier(nn.Module):
    def __init__(self, num_input_channels: int, k_way: int = 5,
                 downsample=None):
        hidden = 64
        """Creates a few shot classifier as used in MAML.

        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        # num_input_channels = 1

        super(FewShotClassifier, self).__init__()
        # self.conv1 = conv_block(num_input_channels, 64)
        # self.conv2 = conv_block(64, 64)
        # self.conv3 = conv_block(64, 64)
        # self.conv4 = conv_block(64, 64)

        # super(FewShotClassifier, self).__init__()
        # # self.conv1 = conv_block(num_input_channels, 64)
        # # self.conv2 = conv_block(64, 64)
        # # self.conv3 = conv_block(64, 64)
        # # self.conv4 = conv_block(64, 64)
        # #self.hidden = 64

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=num_input_channels, out_channels=hidden, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        #nn.BatchNorm2d(num_features=hidden),
        #nn.MaxPool2d(kernel_size=2),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.BatchNorm2d(num_features=hidden, momentum = 0.1, affine=True, track_running_stats=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=hidden, out_channels=int(hidden * 1.5), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        #nn.BatchNorm2d(num_features=int(hidden * 1.5)),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(num_features=int(hidden * 1.5), momentum = 0.1, affine=True, track_running_stats=False),
        nn.MaxPool2d(kernel_size=2))
        #nn.Dropout2d(0.3),)

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=int(hidden * 1.5), out_channels=hidden * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True),
        #nn.BatchNorm2d(num_features=hidden * 2),
        nn.LeakyReLU(negative_slope=0.3, inplace=True),
        nn.BatchNorm2d(num_features=hidden * 2, momentum = 0.1, affine=True, track_running_stats=False),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout2d(0.3))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=int(hidden * 2), out_channels=hidden * 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True),
        #nn.BatchNorm2d(num_features=hidden * 4),
        nn.LeakyReLU(negative_slope=0.3, inplace=True),
        nn.BatchNorm2d(num_features=hidden * 4, momentum = 0.1, affine=True, track_running_stats=False),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout2d(0.0))

        # self.keep_avg_pool = avg_pool
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # self.logits = nn.Linear(final_layer_size, k_way)

        # self.downsample = downsample

        self.logits = nn.Linear(1600, k_way)

    def forward(self, x):
        x = self.conv1(x)
        # print('Size of input X after conv1: ', x.size())
        x = self.conv2(x)
        # print('Size of input X after conv2: ', x.size())
        x = self.conv3(x)
        # print('Size of input X after conv3: ', x.size())
        x = self.conv4(x)
        # print('Size of input X after conv4: ', x.size())
        #print(x.shape)
        x = x.view(x.size(0), -1)

        return self.logits(x)



class MatchingNetwork(nn.Module):
    def __init__(self, n: int, k: int, q: int, fce: bool, num_input_channels: int,
                 lstm_layers: int, lstm_input_size: int, unrolling_steps: int, device: torch.device):
        """Creates a Matching Network as described in Vinyals et al.

        # Arguments:
            n: Number of examples per class in the support set
            k: Number of classes in the few shot classification task
            q: Number of examples per class in the query set
            fce: Whether or not to us fully conditional embeddings
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            lstm_layers: Number of LSTM layers in the bidrectional LSTM g that embeds the support set (fce = True)
            lstm_input_size: Input size for the bidirectional and Attention LSTM. This is determined by the embedding
                dimension of the few shot encoder which is in turn determined by the size of the input data. Hence we
                have Omniglot -> 64, miniImageNet -> 1600.
            unrolling_steps: Number of unrolling steps to run the Attention LSTM
            device: Device on which to run computation
        """
        super(MatchingNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.fce = fce
        self.num_input_channels = num_input_channels
        self.encoder = get_few_shot_encoder(self.num_input_channels)
        if self.fce:
            self.g = BidrectionalLSTM(lstm_input_size, lstm_layers).to(device, dtype=torch.double)
            self.f = AttentionLSTM(lstm_input_size, unrolling_steps=unrolling_steps).to(device, dtype=torch.double)

    def forward(self, inputs):
        pass


class BidrectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        """Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
                layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size,
                                     hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise (ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).cuda().double()
        c = torch.zeros(batch_size, embedding_dim).cuda().double()

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries

            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)

            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries

        return h


