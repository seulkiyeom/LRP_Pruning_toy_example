from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from pandas import DataFrame

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import models

from modules.lrp import lrp
from modules.prune import prune_layer_toy

from heapq import nsmallest
from operator import itemgetter




class Net(nn.Module):
    def __init__(self, num_units=1000, num_class = 2):
        super(Net, self).__init__()

        self.network = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(2, num_units)),
            ('nonlin1', nn.ReLU()),
            ('do1', nn.Dropout()),
            ('linear2', nn.Linear(num_units, 1000)),
            ('nonlin2', nn.ReLU()),
            ('linear3', nn.Linear(num_units, 1000)),
            ('nonlin2', nn.ReLU()),
            ('output', nn.Linear(1000, num_class))]))

    def forward(self, X, **kwargs):
        X = self.network(X)
        return X




def fhook(self, input, output):
    self.input = input[0]
    self.output = output.data

class FilterPruner:
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
        for name, module in enumerate(self.model.network[::-1]):
            # print('module: {}, R_sum: {}, R_size: {}'.format(module, R.sum(), R.shape))

            if isinstance(module, torch.nn.modules.linear.Linear) and name != 0:  # !!!
                activation_index = self.activation_index - self.grad_index - 1
                values = torch.sum(R, dim=0, keepdim=True)[0, :].data

                if activation_index not in self.filter_ranks:
                    zero = torch.FloatTensor(R.size(1)).zero_()
                    if torch.cuda.is_available():
                        zero = zero.cuda()
                    self.filter_ranks[activation_index] = zero

                self.filter_ranks[activation_index] += values
                self.grad_index += 1

            R = lrp(module, R.data, relevance_method, 1)

    def forward(self, x):
        in_size = x.size(0)
        self.activations = []
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
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        # print('Index: {}, activation: {}, grad: {}, weight: {}'.format(activation_index, activation.shape, grad.shape, self.weights[activation_index].shape))

        if self.method_type == 'taylor':
            values = torch.sum((activation * grad), dim=0, keepdim=True)[0, :].data  # P. Molchanov et al., ICLR 2017
            # Normalize the rank by the filter dimensions
            values /= activation.size(0)

        elif self.method_type == 'grad':
            values = torch.sum(grad, dim=0, keepdim=True)[0, :].data  # X. Sun et al., ICML 2017
            # Normalize the rank by the filter dimensions
            values /= activation.size(0)

        elif self.method_type == 'weight':
            weight = self.weights[activation_index]
            values = torch.sum(weight.abs(), dim=1, keepdim=True)[:, 0].data  # Many publications based on weight and activation(=feature) map
            # Normalize the rank by the filter dimensions
            values /= activation.size(0)

        else:
            raise ValueError('No criterion given')

        if activation_index not in self.filter_ranks:
            zero = torch.FloatTensor(activation.size(1)).zero_()
            if torch.cuda.is_available():
                zero = zero.cuda()
            self.filter_ranks[activation_index] = zero

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def normalize_ranks_per_layer(self, norm = True):
        for i in range(len(self.filter_ranks)):

            if self.method_type == 'lrp':  # average over trials - LRP case (this is not normalization !!)
                v = self.filter_ranks[i]
                v /= torch.sum(v)  # torch.sum(v) = total number of dataset
                self.filter_ranks[i] = v.cpu()
            else:
                if norm:  # L2-norm for global rescaling
                    if self.method_type == 'weight':  # weight & L1-norm (Li et al., ICLR 2017)
                        v = self.filter_ranks[i]
                        v /= torch.sum(v)  # L1
                        # v = v / torch.sqrt(torch.sum(v * v)) #L2
                        self.filter_ranks[i] = v.cpu()
                    elif self.method_type == 'taylor':  # |grad*act| & L2-norm (Molchanov et al., ICLR 2017)
                        v = torch.abs(self.filter_ranks[i])
                        v /= torch.sqrt(torch.sum(v * v))
                        self.filter_ranks[i] = v.cpu()
                    elif self.method_type == 'grad':  # |grad| & L2-norm (Sun et al., ICML 2017)
                        v = torch.abs(self.filter_ranks[i])
                        v /= torch.sqrt(torch.sum(v * v))
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

    def get_pruning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        # filters_to_prune contains tuples of: (layer number, filter index, its)

        # After each of the k filters are pruned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))




class PruningFineTuner:
    def __init__(self, model, dataset = 'moon', random_seed=1):
        torch.manual_seed(random_seed)

        self.dataset = dataset
        # TODO generate data based on seed on the fly, instead of loading it.
        ## generate 2d classification dataset
        # X, y = make_moons(n_samples=2000, noise=0.1)
        # X, y = make_circles(n_samples=10, noise=0.1, factor=0.3, random_state=0)
        # X = X.astype(np.float32)
        # y = y.astype(np.int64)
        # np.save('test_X', X)
        # np.save('test_y', y)

        #numpy
        X = np.load('data/' + str(self.dataset) + '_test_X.npy')
        y = np.load('data/' + str(self.dataset) + '_test_y.npy')

        #to torch
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        self.X, self.y = Variable(X), Variable(y)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        # self.train()
        # torch.save(self.model.state_dict(), 'model/' + 'model_' + str(self.dataset))
        self.model.load_state_dict(torch.load('model/' + 'model_' + str(self.dataset),\
                                    map_location='gpu' if torch.cuda.is_available() else 'cpu'))
        self.pruner = FilterPruner(self.model)

    def get_total_number_of_filters(self):
        # counts the total number of non-output dense layer filters in the network
        dense_filters = 0
        for name, module in self.model.network._modules.items():
            if isinstance(module, torch.nn.modules.linear.Linear) \
                    and not name in ['output', '6']:
                dense_filters += module.out_features
        return dense_filters

    def get_candidates_to_prune(self, num_filters_to_prune, method_type = 'lrp'):
        self.pruner.reset(method_type)

        if method_type == 'lrp':
            output = self.pruner.forward_lrp(self.X)

            T = torch.zeros_like(output)
            for ii in range(self.y.size(0)):
                T[ii, self.y[ii]] = 1.0
            self.pruner.backward_lrp(output * T)

        else:
            output = self.pruner.forward(self.X)
            loss = self.criterion(output, self.y)
            loss.backward()

        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_pruning_plan(num_filters_to_prune)

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.model.train()
        for i_epoch in range(10000):
            self.model.zero_grad()
            output = self.model(self.X)
            loss = self.criterion(output, self.y)
            loss.backward()
            optimizer.step()
            # print('Train Epoch: {}, Loss: {}'.format(i_epoch, loss.item()))

    def test(self):
        self.model.eval()
        output = self.model(self.X)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(self.y.data.view_as(pred)).cpu().sum()
        print('Test Accuracy on "{}": {}'.format(self.dataset, (float(correct/len(self.y)) * 100)))


    def prune(self, method_type = 'lrp'):
        number_of_dense = self.get_total_number_of_filters()
        filters_to_prune_per_iteration = 1000 #the number of pruned filter
        prune_targets = self.get_candidates_to_prune(filters_to_prune_per_iteration, method_type)

        layers_pruned = {}
        for layer_index, filter_index in prune_targets:
            if layer_index not in layers_pruned:
                layers_pruned[layer_index] = 0
            layers_pruned[layer_index] += 1

        print("# of dense layer filters per layer selected for pruning:", layers_pruned)
        print("Pruning filters.. ")
        model = self.model.cpu()
        for layer_index, filter_index in prune_targets:
            # print("Layer index: {}, Filter index: {}".format(layer_index, filter_index))
            model = prune_layer_toy(model, layer_index, filter_index, cuda_flag=torch.cuda.is_available())

        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        message = str(100 * float(self.get_total_number_of_filters()) / number_of_dense) + "%"
        print("Fraction of remaining dense layer filters:", str(message))
        # self.test()
        #test

    def visualize_before(self, scenario = 'train'):
        # scatter plot, dots colored by class value
        # TODO LOAD SEED-GENERATED DATASET. OR PASS AS PARAMETERS
        X = np.load('data/' + self.dataset + '_' + scenario + '_X.npy')
        y = np.load('data/' + self.dataset + '_' + scenario + '_y.npy')

        #to torch
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        self.X, self.y = Variable(X), Variable(y)
        self.model.eval()

        output = self.model(X)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).cpu().sum()
        acc = float(correct)/float(len(X)) * 100
        print('{} accuracy: {}%'.format(str(scenario), acc))

        df = DataFrame(
            dict(x=X[:, 0].cpu().numpy().squeeze(), y=self.X[:, 1].cpu().numpy().squeeze(), label=self.y.cpu().numpy().squeeze()))
        # colors = {0: 'red', 1: 'blue'}
        colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        plt.title('Pre-pruning {} accurarcy = {}%'.format(scenario, acc))
        plt.show()


    def visualize_after(self, scenario = 'train'):
        # scatter plot, dots colored by class value
        # TODO LOAD SEED-GENERATED DATASET. OR PASS AS PARAMETERS
        X = np.load('data/' + self.dataset + '_' + scenario + '_X.npy')
        y = np.load('data/' + self.dataset + '_' + scenario + '_y.npy')

        #to torch
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        self.X, self.y = Variable(X), Variable(y)
        self.model.eval()

        output = self.model(X)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).cpu().sum()
        acc = float(correct)/float(len(X)) * 100
        print('Post-pruning {} accuracy = {}%'.format(scenario, acc))

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

        # to torch
        data_t = torch.FloatTensor(data)
        if torch.cuda.is_available():
            data_t.cuda()

        # Pass data to predict method
        db_prob = self.model(data_t)

        clf = np.where(db_prob < 0.5, 0, 1)

        Z = clf[:,0].reshape(XX.shape)

        # scatter plot, dots colored by class value
        df = DataFrame(dict(x=X[:, 0].cpu().numpy().squeeze(), y=X[:, 1].cpu().numpy().squeeze(), label=pred.cpu().numpy().squeeze()))
        # colors = {0: 'red', 1: 'blue'}
        colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            # print(key)
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

        plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)
        # TODO remove hard coding of canvas limits
        plt.xlim(-1.4154078960418701, 2.3496716022491455) #moon
        plt.ylim(-0.8391216397285461, 1.4337007403373718) #moon
        # plt.xlim(-1.3220499753952026, 1.3229042291641235) #circle
        # plt.ylim(-1.3928583860397339, 1.305529236793518) #circle
        plt.title('Pre-pruning {} accurarcy = {}%'.format(scenario, acc))
        plt.show()
        # TODO remove hardoding of output file name
        fig.savefig('grad.svg', dpi=fig.dpi)




if __name__ == "__main__":
    '''
    Pruning test with toy dataset
    '''



    dataset = 'moon' #dataset: moon, circle, mult
    method_type = 'taylor' #pruning criteria: lrp, grad, taylor, weight

    if dataset == 'moon' or dataset == 'circle':
        model = Net(num_class=2)
    elif dataset == 'mult':
        model = Net(num_class=4)

    if torch.cuda.is_available():
        model = model.cuda()

    # TODO. logging.
    fine_tuner = PruningFineTuner(model, dataset = dataset) # TODO let this one generate / load all the data.
    fine_tuner.visualize_before(scenario= 'train') # I think this is not required at all
    #fine_tuner.visualize_after(scenario= 'train') # TODO load dataset here, make distinction between pre-and post eval obsolete. # TODO: measure performance independently from visualization
    fine_tuner.prune(method_type = method_type)
    fine_tuner.visualize_after(scenario= 'test')
    fine_tuner.visualize_after(scenario= 'train')
    print('Done')
