from collections import OrderedDict
import argparse

import numpy as np
import os
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
    def __init__(self, model, pruning_criterion):
        # TODO: introcude criterion via constructor.
        self.model = model
        self.pruning_criterion = pruning_criterion
        self.reset(self.pruning_criterion)

    def reset(self, pruning_criterion):
        self.filter_ranks = {}
        self.forward_hook()
        self.pruning_criterion = pruning_criterion

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

        if self.pruning_criterion == 'taylor':
            values = torch.sum((activation * grad), dim=0, keepdim=True)[0, :].data  # P. Molchanov et al., ICLR 2017
            # Normalize the rank by the filter dimensions
            values /= activation.size(0)

        elif self.pruning_criterion == 'grad':
            values = torch.sum(grad, dim=0, keepdim=True)[0, :].data  # X. Sun et al., ICML 2017
            # Normalize the rank by the filter dimensions
            values /= activation.size(0)

        elif self.pruning_criterion == 'weight':
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

            if self.pruning_criterion == 'lrp':  # average over trials - LRP case (this is not normalization !!)
                v = self.filter_ranks[i]
                v /= torch.sum(v)  # torch.sum(v) = total number of data
                self.filter_ranks[i] = v.cpu()
            else:
                if norm:  # L2-norm for global rescaling
                    if self.pruning_criterion == 'weight':  # weight & L1-norm (Li et al., ICLR 2017)
                        v = self.filter_ranks[i]
                        v /= torch.sum(v)  # L1
                        # v = v / torch.sqrt(torch.sum(v * v)) #L2
                        self.filter_ranks[i] = v.cpu()
                    elif self.pruning_criterion == 'taylor':  # |grad*act| & L2-norm (Molchanov et al., ICLR 2017)
                        v = torch.abs(self.filter_ranks[i])
                        v /= torch.sqrt(torch.sum(v * v))
                        self.filter_ranks[i] = v.cpu()
                    elif self.pruning_criterion == 'grad':  # |grad| & L2-norm (Sun et al., ICML 2017)
                        v = torch.abs(self.filter_ranks[i])
                        v /= torch.sqrt(torch.sum(v * v))
                        self.filter_ranks[i] = v.cpu()
                else:
                    if self.pruning_criterion == 'weight':  # weight
                        v = self.filter_ranks[i]
                        self.filter_ranks[i] = v.cpu()
                    elif self.pruning_criterion == 'taylor':  # |grad*act|
                        v = torch.abs(self.filter_ranks[i])
                        self.filter_ranks[i] = v.cpu()
                    elif self.pruning_criterion == 'grad':  # |grad|
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
    def __init__(self, model, dataset = 'moon',
                criterion='lrp', n_samples=5, random_seed=1,
                render='none', color_map='Dark2', log_file='./log.txt'):


        self.random_seed        = random_seed
        self.dataset            = dataset
        self.pruning_criterion  = criterion
        self.n_samples          = n_samples
        self.render             = render
        self.color_map          = color_map
        self.log_file           = log_file
        self.log_dir            = os.path.dirname(log_file)
        self.log_file           = open(self.log_file, 'ta')
        self.experiment_name    = 'dataset:{}-criterion:{}-n:{}-s:{}'.format(self.dataset, self.pruning_criterion, self.n_samples, self.random_seed)
        self.pruning_stage      = 'pre' # will be set to 'post' after pruning




        # init random seed for everything coming.
        torch.manual_seed(self.random_seed)
        # TODO generate data based on seed on the fly, instead of loading it.
        ## generate 2d classification dataset
        #

        # load data used for training the loaded model
        X_train = np.load('data/' + str(self.dataset) + '_train_X.npy')
        y_train_true = np.load('data/' + str(self.dataset) + '_train_y.npy')

        # ad-hoc generate test data instead, using the previously set random seed
        if self.dataset == 'moon':
            X_test, y_test_true = make_moons(n_samples=n_samples, noise=0.1, random_state=self.random_seed)
        elif self.dataset == 'circle':
            X_test, y_test_true = make_circles(n_samples=n_samples, noise=0.1, factor=0.3, random_state=self.random_seed)
        elif self.dataset == 'mult':
            # this is NOT the function used previously
            #X_test, y_test_true = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_repeated=0, n_redundant=0, random_state=self.random_seed)
            print("WARNING! This is a pre-recorded dataset since no-one bothered actually creating a documentation. it only works for 5 samples per class.")
            assert self.n_samples == 5*4, "Pre-recorded dataset not fit for {} != 20=5*4 samples (i.e. NOT 5 samples per class)"
            # load previously used test data used for training the loaded model
            X_test = np.load('data/mult_test_X.npy')
            y_test_true = np.load('data/mult_test_y.npy')
        else:
            raise ValueError('Unsupported Dataset name "{}"'.format(self.dataset))

        X_train         = X_train.astype(np.float32)
        y_train_true    = y_train_true.astype(np.int64)
        X_test          = X_test.astype(np.float32)
        y_test_true     = y_test_true.astype(np.int64)

        #to torch
        X_train         = torch.from_numpy(X_train)
        y_train_true    = torch.from_numpy(y_train_true)
        X_test          = torch.from_numpy(X_test)
        y_test_true     = torch.from_numpy(y_test_true)
        if torch.cuda.is_available():
            X_test          = X_test.cuda()
            y_test_true     = y_test_true.cuda()
            X_train         = X_train.cuda()
            y_train_true    = y_train_true.cuda()

        self.X_test = Variable(X_test)
        self.y_test_true = Variable(y_test_true)
        self.X_train = Variable(X_train)
        self.y_train_true = Variable(y_train_true)

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        # self.train()
        # torch.save(self.model.state_dict(), 'model/' + 'model_' + str(self.dataset))
        self.model.load_state_dict(torch.load('model/' + 'model_' + str(self.dataset),\
                                    map_location='gpu' if torch.cuda.is_available() else 'cpu'))
        self.pruner = FilterPruner(self.model, self.pruning_criterion)

    def get_total_number_of_filters(self):
        # counts the total number of non-output dense layer filters in the network
        dense_filters = 0
        for name, module in self.model.network._modules.items():
            if isinstance(module, torch.nn.modules.linear.Linear) \
                    and not name in ['output', '6']:
                dense_filters += module.out_features
        return dense_filters

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.pruner.reset(self.pruning_criterion)

        if self.pruning_criterion == 'lrp':
            output = self.pruner.forward_lrp(self.X_test)

            T = torch.zeros_like(output)
            for ii in range(self.y_test_true.size(0)):
                T[ii, self.y_test_true[ii]] = 1.0
            self.pruner.backward_lrp(output * T)

        else:
            output = self.pruner.forward(self.X_test)
            loss = self.criterion(output, self.y_test_true)
            loss.backward()

        self.pruner.normalize_ranks_per_layer()
        return self.pruner.get_pruning_plan(num_filters_to_prune)

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.model.train()
        for i_epoch in range(10000):
            self.model.zero_grad()
            output = self.model(self.X_test)
            loss = self.criterion(output, self.y_test_true)
            loss.backward()
            optimizer.step()
            # print('Train Epoch: {}, Loss: {}'.format(i_epoch, loss.item()))

    def prune(self):
        number_of_dense = self.get_total_number_of_filters()
        filters_to_prune_per_iteration = 1000 #the number of pruned filter
        prune_targets = self.get_candidates_to_prune(filters_to_prune_per_iteration)

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

        # change pruning state (important for figure output)
        self.pruning_stage = 'post'

    def evaluate_and_visualize(self, scenario = 'train'):

        eval_name = '{}-scenario:{}-stage:{}'.format(self.experiment_name, scenario, self.pruning_stage)

        if scenario == 'train':
            X       = self.X_train
            y_true  =  self.y_train_true
        elif scenario == 'test':
            X       = self.X_test
            y_true  = self.y_test_true
        else:
            raise ValueError('Unsupported scenario "{}" in {}.evaluate_and_visualize'.format(scenario, self.__class__.__name__))

        # set model to eval mode, then eval.
        self.model.eval()
        output = self.model(X)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(y_true.data.view_as(pred)).cpu().sum()
        acc = float(correct)/float(len(X)) * 100
        print('{}-pruning {} accuracy = {}%'.format(self.pruning_stage, scenario, acc))
        self.log_file.write('{} {}\n'.format(eval_name, acc))

        if self.render in ['svg', 'show']:
            # create necessary data for visualizing data and decision boundaries.
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

            # Set grid spacing parameter
            spacing = min(x_max - x_min, y_max - y_min) / 200

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
            num_classes = db_prob.shape[1]

            # classify (discretize outputs)
            clf = np.argmax(db_prob.detach().numpy(), axis=1)
            Z = clf.reshape(XX.shape)

            # prepare dataset samples (not meshgrid ones) for rendering
            X = X.cpu().numpy()
            y = pred.cpu().numpy().squeeze()


            cmap_contourf=plt.cm.get_cmap(self.color_map, num_classes) # create custom colormap matching num_classes
            cmap_scatter=plt.cm.get_cmap(self.color_map, num_classes)
            # darker colors for scatter plot
            factor = 0.80
            cmap_scatter.colors *= np.array([[factor]*3 + [1]])
            fig, ax = plt.subplots()

            plt.contourf(XX, YY, Z, alpha=.75, cmap=cmap_contourf)
            plt.scatter(x=X[:,0], y=X[:,1], c=y_true, cmap=cmap_scatter)

            plt.xlim(x_min, x_max-spacing)
            plt.ylim(y_min, y_max-spacing)
            plt.title('{}-pruning {} accurarcy with {} = {:.2f}%'.format(self.pruning_stage, scenario, self.pruning_criterion, acc))

            figname = '{}/{}.svg'.format(self.log_dir, eval_name)
            print('Saving figure to "{}"'.format(figname))
            fig.savefig(figname, dpi=fig.dpi)

            if self.render == 'show':
                plt.show()


    def close(self):
        try:
            self.log_file.close()
        except:
            pass



if __name__ == "__main__":
    '''
    Pruning test with toy dataset
    '''
    valid_datasets      = ['moon', 'circle', 'mult']
    valid_criteria      = ['lrp', 'taylor', 'grad', 'weight']
    valid_rendermodes   = ['none', 'svg', 'show']   # no visualizion, only svg output, svg+on-screen figure
    num_classes         = {'moon':2, 'circle':2, 'mult':4}

    def generate_calls():
        print('Generating parametres and shell scripts for the experiment.')
        print('One shell script per dataset x criterion combination')

        if not os.path.isdir('scripts'):
            os.mkdir('scripts')

        rendermode = 'none'
        colormap = 'Set1'
        logfile = '/output/log.txt'

        for data in valid_datasets:
            for criterion in valid_criteria:
                scriptfile = 'scripts/{}-{}.sh'.format(data, criterion)
                print('Generating {} ...'.format(scriptfile))
                with open(scriptfile, 'wt') as f:
                    f.write('#!/bin/bash\n')
                    for n in [1, 5, 10, 20, 50, 100, 200]:  # --num_samples
                        for s in range(20):                 # --seed , different repetitions of the same experiment
                            cmd = ['python',
                                   'main.py',
                                   '--rendermode {}'.format(rendermode),
                                   '--colormap {}'.format(colormap),
                                   '--logfile {}'.format(logfile),
                                   '--dataset {}'.format(data),
                                   '--criterion {}'.format(criterion),
                                   '--numsamples {}'.format(n),
                                   '--seed {}'.format(s)]
                            f.write(' '.join(cmd) + '\n')

        print('Done generating experiment scripts.')




    parser = argparse.ArgumentParser(description = 'Neural Network Pruning Toy experiment')
    parser.add_argument('--dataset',    '-d',   type=str, default='mult',         help='The toy dataset to use. Choices: {}'.format(', '.join(valid_datasets)))
    parser.add_argument('--criterion',  '-c',   type=str, default='lrp',          help='The criterion to use for pruning. Choices: {}'.format(', '.join(valid_criteria)))
    parser.add_argument('--numsamples', '-n',   type=int, default=5,              help='Number of training samples to use for computing the pruning criterion.')
    parser.add_argument('--seed',       '-s',   type=int, default=1,              help='Random seed used for (random) sample selection for pruning criterion computation.')
    parser.add_argument('--rendermode', '-r',   type=str, default='none',         help='Is result visualization desired? Choices: {}'.format(', '.join(valid_rendermodes)))
    parser.add_argument('--colormap',   '-cm',  type=str, default='Dark2',        help='The colormap to use for rendering the output figures. Must be a valid one from matplotlib.')
    parser.add_argument('--logfile',    '-l',   type=str, default='./log.txt',    help='Output log file location. Results will pe appended. File location (folder) must exist!!!')
    parser.add_argument('--generate',   '-g',   action='store_true',              help='Calls a function to generate a bunch of parameterized function calls. Recommendation. First call this tool with "-g", then execute the generated scripts.')
    args = parser.parse_args()


    # catch the "generate calls" functionality
    if args.generate:
        generate_calls()
        exit()


    # verify parametrer choices
    assert args.dataset     in valid_datasets,      'Invalid dataset choice "{}". Must be from {}'.format(args.dataset, valid_datasets)
    assert args.criterion   in valid_criteria,      'Invalid pruning criterion "{}". Must be from {}'.format(args.criterion, valid_criteria)
    assert args.numsamples  > 0,                    'Number of samples (per class) used for pruning criterion computation must be > 0, but was {}'.format(args.num_samples)
    assert args.rendermode  in valid_rendermodes,   'Invalid render mode "{}". Must be from {}'.format(args.rendermode, valid_rendermodes)
    assert args.colormap in plt.colormaps(),        'Invalid colormap choice "{}". Must be from matplotlib.pyplot.colormaps(), i.e. {}'.format(args.colormap, plt.colormaps())

    logdir = os.path.dirname(args.logfile)
    assert os.path.isdir(logdir),                   'Log file location "{}" does not exist. Will not be able to create log file and/or figures! Make sure your target folder exists (to avoid spam)'.format(logdir)

    model = Net(num_class=num_classes[args.dataset])
    if torch.cuda.is_available(): model = model.cuda()

    fine_tuner = PruningFineTuner(model,
                                  dataset=args.dataset,
                                  criterion=args.criterion,
                                  n_samples=args.numsamples*num_classes[args.dataset], # adjust sample size to number of classes
                                  random_seed=args.seed,
                                  render=args.rendermode,
                                  color_map=args.colormap,
                                  log_file=args.logfile)

    fine_tuner.evaluate_and_visualize(scenario='train')
    fine_tuner.evaluate_and_visualize(scenario='test')
    fine_tuner.prune()
    fine_tuner.evaluate_and_visualize(scenario='train')
    fine_tuner.evaluate_and_visualize(scenario='test')
    fine_tuner.close()

    print('Done for {}'.format(fine_tuner.experiment_name))

