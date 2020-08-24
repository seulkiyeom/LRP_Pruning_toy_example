import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_multilabel_classification
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from modules.lrp import lrp
from modules.prune import prune_layer_toy
from modules.data import get_multitoy_data
from heapq import nsmallest
from operator import itemgetter
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def fhook(self, input, output):
    self.input = input[0]
    self.output = output.data

class Net(nn.Module):
    def __init__(self, num_units=1000, nonlin=F.relu):
        super(Net, self).__init__()

        self.network = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(2, num_units)),
            ('nonlin1', nn.ReLU()),
            ('do1', nn.Dropout()),
            ('linear2', nn.Linear(num_units, 1000)),
            ('nonlin2', nn.ReLU()),
            ('linear3', nn.Linear(num_units, 1000)),
            ('nonlin2', nn.ReLU()),
            ('output', nn.Linear(1000, 4))]))

    def forward(self, X, **kwargs):
        X = self.network(X)
        return X

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self, method_type = 'lrp'):
        self.filter_ranks = {}
        self.forward_hook()
        self.method_type = method_type

    def forward_hook(self):
        for name, module in self.model.network._modules.items():
            module.register_forward_hook(fhook)

    def forward_lrp(self, x):
        self.activation_to_layer = {}
        self.grad_index = 0
        self.activation_index = 0

        for layer, (name, module) in enumerate(self.model.network._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.linear.Linear) and name != 'output':
                self.activation_to_layer[self.activation_index] = layer
                self.activation_index += 1

        return x
    
    def backward_lrp(self, R, relevance_method='z'):
        for name, module in enumerate(self.model.network[::-1]):  # ?? ??
            # print('module: {}, R_sum: {}, R_size: {}'.format(module, R.sum(), R.shape))

            if isinstance(module, torch.nn.modules.linear.Linear) and name != 0:  # !!!
                activation_index = self.activation_index - self.grad_index - 1

                values = \
                    torch.sum(R, dim=0, keepdim=True)[0, :].data

                if activation_index not in self.filter_ranks:
                    self.filter_ranks[activation_index] = torch.FloatTensor(
                        R.size(1)).zero_().cuda() if torch.cuda.is_available() else torch.FloatTensor(
                        R.size(1)).zero_()

                self.filter_ranks[activation_index] += values
                self.grad_index += 1

            R = lrp(module, R.data, relevance_method, 1)

    def forward(self, x):
        in_size = x.size(0)
        self.activations = []  # ?? conv_layer? activation map ?
        self.weights = []
        self.gradients = []
        self.activation_to_layer = {}
        self.grad_index = 0
        activation_index = 0

        for layer, (name, module) in enumerate(self.model.network._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.linear.Linear) and name != 'output':
                x.register_hook(self.compute_rank)
                self.weights.append(module.weight)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1
        return x

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1  # ????? ??? ??? ?
        activation = self.activations[activation_index]
        # print('Index: {}, activation: {}, grad: {}, weight: {}'.format(activation_index, activation.shape, grad.shape, self.weights[activation_index].shape))

        if self.method_type == 'taylor':
            values = \
                torch.sum((activation * grad), dim=0, keepdim=True)[0, :].data  # P. Molchanov et al., ICLR 2017
            # Normalize the rank by the filter dimensions
            values = \
                values / activation.size(0)

        elif self.method_type == 'grad':
            values = \
                torch.sum(grad, dim=0, keepdim=True)[0, :].data  # X. Sun et al., ICML 2017
            # Normalize the rank by the filter dimensions
            values = \
                values / activation.size(0)

        elif self.method_type == 'weight':
            weight = self.weights[activation_index]
            values = \
                torch.sum(weight.abs(), dim=1, keepdim=True)[:,
                0].data  # Many publications based on weight and activation(=feature) map
            # Normalize the rank by the filter dimensions
            values = \
                values / activation.size(0)

        else:
            raise ValueError('No criteria')

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(
                activation.size(1)).zero_().cuda() if torch.cuda.is_available() else torch.FloatTensor(activation.size(1)).zero_()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def normalize_ranks_per_layer(self, norm = True):
        for i in range(len(self.filter_ranks)):

            if self.method_type == 'lrp':  # average over trials - LRP case (this is not normalization !!)
                v = self.filter_ranks[i]
                v = v / torch.sum(v)  # torch.sum(v) = total number of dataset
                self.filter_ranks[i] = v.cpu()
            else:
                if norm:  # L2-norm for global rescaling
                    if self.method_type == 'weight':  # weight & L1-norm (Li et al., ICLR 2017)
                        v = self.filter_ranks[i]
                        v = v / torch.sum(v)  # L1
                        # v = v / torch.sqrt(torch.sum(v * v)) #L2
                        self.filter_ranks[i] = v.cpu()
                    elif self.method_type == 'taylor':  # |grad*act| & L2-norm (Molchanov et al., ICLR 2017)
                        v = torch.abs(self.filter_ranks[i])
                        v = v / torch.sqrt(torch.sum(v * v))
                        self.filter_ranks[i] = v.cpu()
                    elif self.method_type == 'grad':  # |grad| & L2-norm (Sun et al., ICML 2017)
                        v = torch.abs(self.filter_ranks[i])
                        v = v / torch.sqrt(torch.sum(v * v))
                        self.filter_ranks[i] = v.cpu()
                else:
                    if self.method_type == 'weight':  # weight
                        v = self.filter_ranks[i]
                        self.filter_ranks[i] = v.cpu()
                    elif self.method_type == 'taylor':  # |grad*act|
                        v = torch.abs(self.filter_ranks[i])
                        self.filter_ranks[i] = v.cpu()
                    elif self.method_type == 'grad':  # |grad|
                        v = torch.abs(self.filter_ranks[i])
                        self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        # filters_to_prune: filters to be pruned 1) layer number, 2) filter number, 3) its value

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):  # ?? ? ???!
                # ? ? layer? ???? ??? ??? ?? ?? ?? ??????
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune  # ??? ? filter?? 1) layer number, 2) filter number

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
                # data ??? ?? layer? ?? filter? ?? ?? ?? ???.

        return nsmallest(num, data, itemgetter(2))  # data list ??? ?? ?? ?? num(=512?) ?? ??? ???? ??

class PruningFineTuner:
    def __init__(self, model):
        torch.manual_seed(1)

        # N = 1000  # number of points per class
        # D = 2  # dimensionality
        # K = 4  # number of classes
        # X = np.zeros((N * K, D))  # data matrix (each row = single example)
        # y = np.zeros(N * K, dtype='uint8')  # class labels
        # for j in range(K):
        #     ix = range(N * j, N * (j + 1))
        #     r = np.linspace(0.0, 1, N)  # radius
        #     t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        #     X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        #     y[ix] = j
        #
        # X = X.astype(np.float32)
        # y = y.astype(np.int64)
        # np.save('mult_train_X', X)
        # np.save('mult_train_y', y)

        train_dataset, test_dataset = get_multitoy_data()

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=100)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=100,
                                                       shuffle=False)

        X = np.load('mult_train_X.npy')
        y = np.load('mult_train_y.npy')

        if torch.cuda.is_available():
            X, y = torch.from_numpy(X).cuda(), torch.from_numpy(y).cuda()
        else:
            X, y = torch.from_numpy(X), torch.from_numpy(y)

        self.X, self.y = Variable(X), Variable(y)

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        # self.train()
        # torch.save(self.model.state_dict(), 'model_mult')
        self.model.load_state_dict(torch.load('model_mult'))
        self.test()
        self.prunner = FilterPrunner(self.model)
        # self.visualize_before()
        # self.visualize_after()

    def total_num_filters(self):
        # Conv layer? ?? filter ?? counting
        dense_filters = 0
        for name, module in self.model.network._modules.items():
            if isinstance(module, torch.nn.modules.linear.Linear) and name != 'output' and name != '6':
                dense_filters += module.out_features

        return dense_filters

    def get_candidates_to_prune(self, num_filters_to_prune, method_type = 'lrp'):
        self.prunner.reset(method_type)

        if method_type == 'lrp':
            output = self.prunner.forward_lrp(self.X)

            T = torch.zeros_like(output)
            for ii in range(self.y.size(0)):  # each data (= 20)
                T[ii, self.y[ii]] = 1.0

            self.prunner.backward_lrp(output * T)

        else:
            output = self.prunner.forward(self.X)
            loss = self.criterion(output, self.y)
            loss.backward()

        self.prunner.normalize_ranks_per_layer()

        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.model.train()

        for i_epoch in range(1000):
            loss_all = 0
            for batch_idx, (X, y) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                X, y = Variable(X), Variable(y)

                self.model.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                optimizer.step()
                loss_all += loss.item()

            print('Train Epoch: {}, Loss: {}'.format(i_epoch, loss_all))


    def test(self):
        self.model.eval()
        output = self.model(self.X)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(self.y.data.view_as(pred)).cpu().sum()
        print('Test Accuracy: {}'.format(correct))

    def prune(self, method_type = 'lrp'):

        number_of_dense = self.total_num_filters()
        filters_to_prune_per_iteration = 1000 #the number of pruned filter
        prune_targets = self.get_candidates_to_prune(filters_to_prune_per_iteration, method_type)

        layers_prunned = {}
        for layer_index, filter_index in prune_targets:
            if layer_index not in layers_prunned:
                layers_prunned[layer_index] = 0
            layers_prunned[layer_index] += 1

        print("Dense layers that will be prunned", layers_prunned)  # ? ?? layer ? filter ?
        print("Prunning filters.. ")
        model = self.model.cpu()
        for layer_index, filter_index in prune_targets:  # ??? ??? ??? ??
            # print("Layer index: {}, Filter index: {}".format(layer_index, filter_index))
            model = prune_layer_toy(model, layer_index, filter_index, cuda_flag=torch.cuda.is_available())

        self.model = model.cuda() if torch.cuda.is_available() else model
        message = str(100 * float(self.total_num_filters()) / number_of_dense) + "%"
        print("Dense layers prunned", str(message))
        self.test()
        #test

    def visualize_before(self, type = 'train'):
        # scatter plot, dots colored by class value
        X = np.load('mult_' + type + '_X.npy')
        y = np.load('mult_' + type + '_y.npy')

        if torch.cuda.is_available():
            X, y = torch.from_numpy(X).cuda(), torch.from_numpy(y).cuda()
        else:
            X, y = torch.from_numpy(X), torch.from_numpy(y)

        self.X, self.y = Variable(X), Variable(y)

        df = DataFrame(
            dict(x=X[:, 0].cpu().numpy().squeeze(), y=self.X[:, 1].cpu().numpy().squeeze(), label=self.y.cpu().numpy().squeeze()))
        colors = {0: 'red', 1: 'blue'}
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            print(key)
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

        plt.show()

    def visualize_after(self, type = 'train'):
        # scatter plot, dots colored by class value
        X = np.load('mult_' + type + '_X.npy')
        y = np.load('mult_' + type + '_y.npy')

        if torch.cuda.is_available():
            X, y = torch.from_numpy(X).cuda(), torch.from_numpy(y).cuda()
        else:
            X, y = torch.from_numpy(X), torch.from_numpy(y)

        X, y = Variable(X), Variable(y)

        self.model.eval()
        output = self.model(X)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).cpu().sum()
        print(correct)

        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

        # Set grid spacing parameter
        spacing = min(x_max - x_min, y_max - y_min) / 100

        # Create grid
        XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                             np.arange(y_min, y_max, spacing))

        # Concatenate data to match input
        data = np.hstack((XX.ravel().reshape(-1, 1),
                          YY.ravel().reshape(-1, 1)))

        # Pass data to predict method
        if torch.cuda.is_available():
            data_t = torch.FloatTensor(data).cuda()
        else:
            data_t = torch.FloatTensor(data)

        db_prob = self.model(data_t)


        # Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
        clf = np.argmax(db_prob.cpu().detach().numpy(), axis=1)

        Z = clf.reshape(XX.shape)

        # scatter plot, dots colored by class value
        df = DataFrame(dict(x=X[:, 0].cpu().numpy().squeeze(), y=X[:, 1].cpu().numpy().squeeze(), label=pred.cpu().numpy().squeeze()))
        colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            print(key)
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

        plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)
        plt.xlim(-0.9933979511260986, 1.0972111225128174) #multiclass
        plt.ylim(-1.0907000303268433, 1.0626273155212402) #multiclass
        plt.show()
        fig.savefig('multi_train_2.svg', dpi=50)

if __name__ == "__main__":
    '''
    Test with Toy dataset
    '''

    model = Net()
    if torch.cuda.is_available():
        model = model.cuda()
    fine_tuner = PruningFineTuner(model)
    fine_tuner.visualize_after(type= 'train')
    # fine_tuner.visualize_after(type= 'test')
    # fine_tuner.prune(method_type='lrp')
    # fine_tuner.prune(method_type='grad')
    fine_tuner.prune(method_type = 'taylor')
    # fine_tuner.prune(method_type = 'weight')
    fine_tuner.visualize_after(type= 'train')
    # fine_tuner.test()
    print('Finish')
