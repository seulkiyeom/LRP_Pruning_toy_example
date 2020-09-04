from collections import OrderedDict
import argparse
import itertools
import json
import tqdm
from prettytable import PrettyTable

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from scipy.stats import kendalltau, spearmanr
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

    def backward_lrp(self, R, relevance_method='z+'):
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
        filters_to_prune, all_filters_with_score = self.lowest_ranking_filters(num_filters_to_prune)
        # filters_to_prune contains tuples of: (layer number, filter index, its score)

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

        return filters_to_prune, all_filters_with_score

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2)), data #additionally return rank over all neurons




class PruningFineTuner:
    def __init__(self, model, dataset = 'moon',
                criterion='lrp', n_samples=5, random_seed=1,
                render='none', color_map='Dark2', log_file='./log.txt', rank_analysis=False, noisy_test=0.0):


        self.random_seed        = random_seed
        self.dataset            = dataset
        self.pruning_criterion  = criterion
        self.n_samples          = n_samples
        self.render             = render
        self.color_map          = color_map
        self.rank_analysis      = rank_analysis
        self.noisy_test_sigma   = noisy_test
        self.log_file           = log_file
        self.log_dir            = os.path.dirname(log_file)
        self.log_file           = open(self.log_file, 'ta')
        self.experiment_name    = 'dataset:{}-criterion:{}-n:{}-s:{}'.format(self.dataset, self.pruning_criterion, self.n_samples, self.random_seed)
        self.pruning_stage      = 'pre' # will be set to 'post' after pruning

        # announce experimental setup
        print(self.experiment_name)




        # init random seed for everything coming.
        torch.manual_seed(self.random_seed)

        # load data used for training the loaded model
        X_train = np.load('data/' + str(self.dataset) + '_train_X.npy')
        y_train_true = np.load('data/' + str(self.dataset) + '_train_y.npy')

        X_test, y_test_true = self.generate_data(self.dataset, n_samples, self.random_seed)

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


        # in case we want to evaluate with a noisy and extra-large test set not used for pruning
        if self.noisy_test_sigma > 0:
            # fixed number of test samples per class: 500
            X_test_noisy, y_test_noisy = self.generate_data(self.dataset, 500, self.random_seed+1)
            X_test_noisy = X_test_noisy.astype(np.float32)
            y_test_noisy = y_test_noisy.astype(np.int64)

            X_test_noisy += np.random.normal(0, self.noisy_test_sigma, X_test_noisy.shape) # additive gaussian noise

            # to torch
            X_test_noisy = torch.from_numpy(X_test_noisy)
            y_test_noisy = torch.from_numpy(y_test_noisy)
            if torch.cuda.is_available():
                X_test_noisy = X_test_noisy.cuda()
                y_test_noisy = y_test_noisy.cuda()
            self.X_test_noisy = Variable(X_test_noisy)
            self.y_test_noisy = Variable(y_test_noisy)


        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        # self.train()
        # torch.save(self.model.state_dict(), 'model/' + 'model_' + str(self.dataset))
        self.model.load_state_dict(torch.load('model/' + 'model_' + str(self.dataset),\
                                    map_location='gpu' if torch.cuda.is_available() else 'cpu'))
        self.pruner = FilterPruner(self.model, self.pruning_criterion)


    def generate_data(self, dset_name, n_samples, random_seed):
        np.random.seed(random_seed)
        # ad-hoc generate test data instead, using the previously set random seed
        if self.dataset == 'moon':
            X_test, y_test_true = make_moons(n_samples=n_samples, noise=0.1, random_state=random_seed)
        elif self.dataset == 'circle':
            X_test, y_test_true = make_circles(n_samples=n_samples, noise=0.1, factor=0.3, random_state=random_seed)
        elif self.dataset == 'mult':
            D = 2  # dimensionality
            K = 4  # number of classes
            N = int(n_samples/K) #samples per class
            X = np.zeros((N * K, D))  # data matrix (each row = single example)
            y = np.zeros(N * K, dtype='uint8')  # class labels
            for j in range(K):
                 ix = range(N * j, N * (j + 1))
                 r = np.linspace(0.0, 1, N)  # radius
                 t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
                 X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
                 y[ix] = j

            X_test = X
            y_test_true = y
        else:
            raise ValueError('Unsupported Dataset name "{}"'.format(self.dataset))

        return X_test, y_test_true

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
        prune_targets, all_targets_with_score = self.get_candidates_to_prune(filters_to_prune_per_iteration)

        # generate parsable neuron index representation for later rank correlation analysis, write to log, then exit.
        if self.rank_analysis:
            # instead of actually pruning, we here are only interested in the candidates and how they are ranked for pruning.
            # we can thus exit right after writing the data.
            all_targets_with_score_array = np.array([t[2].numpy() for t in all_targets_with_score ]) # convert to matrix with columns "layer", "neuron", and "score"
            # compute rank of all network elements wrt score, since they are already unique identified in all_targets_with_score, and ordered via layer first, neuron index second, we are done here.
            pruning_order = np.argsort(all_targets_with_score_array)
            pruning_order_str = json.dumps(pruning_order.tolist()).replace(' ','')

            eval_name = '{}-scenario:{}-stage:{}'.format(self.experiment_name, 'rankselection', self.pruning_stage)
            self.log_file.write('{} {}\n'.format(eval_name, pruning_order_str))
            self.log_file.close() # manually close the file, to flush it (otherwise exit() prevents the writing)
            exit()

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
            if self.noisy_test_sigma > 0:
                X       = self.X_test_noisy
                y_true  = self.y_test_noisy
            else:
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
            x_min, x_max = self.X_train[:, 0].min() - 0.1, self.X_train[:, 0].max() + 0.1
            y_min, y_max = self.X_train[:, 1].min() - 0.1, self.X_train[:, 1].max() + 0.1

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
            plt.figure(figsize=(3.5, 3.5))
            plt.subplots_adjust(left=0.19, right=0.99, top=0.93, bottom=0.13)

            plt.contourf(XX, YY, Z, alpha=.5, cmap=cmap_contourf)
            plt.scatter(x=X[:,0], y=X[:,1], c=y_true, cmap=cmap_scatter)

            plt.xlim(x_min, x_max-spacing)
            plt.ylim(y_min, y_max-spacing)
            plt.title('{}-pruning {} accurarcy with {} = {:.2f}%'.format(self.pruning_stage, scenario, self.pruning_criterion, acc))

            figname = '{}/{}.svg'.format(self.log_dir, eval_name)
            print('Saving figure to "{}"'.format(figname))
            plt.savefig(figname)

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

    def generate_calls(args):
        print('Generating parameters and shell scripts for the experiment.')
        print('One shell script per {dataset x criterion} combination, covering all {num_samples x random_seed} variations')

        dirname_suffix = '{}'.format('-rankanalysis' if args.ranklog else ('-noisytest-{}'.format(args.noisytest) if args.noisytest else ''))
        scriptdir = 'scripts{}'.format(dirname_suffix)
        if not os.path.isdir(scriptdir):
            print('Generating {}/'.format(scriptdir))
            os.mkdir(scriptdir)

        rendermode = 'none'
        colormap = 'Accent'
        logfile = './output{}/log.txt'.format(dirname_suffix)
        logdir = os.path.dirname(logfile)
        if not os.path.isdir(logdir):
            print('Generating {}/'.format(logdir))
            os.mkdir(logdir)

        for data in valid_datasets:
            for criterion in valid_criteria:
                scriptfile = 'scripts{}/{}-{}.sh'.format(dirname_suffix, data, criterion)
                print('Generating {} ...'.format(scriptfile))
                with open(scriptfile, 'wt') as f:
                    f.write('#!/bin/bash\n')
                    #f.write('cd ..\n')
                    for n in [1, 5, 10, 20, 50, 100, 200]:  # --num_samples
                        for s in range(50):                 # --seed , different repetitions of the same experiment
                            cmd = ['python',
                                   'main.py',
                                   '--rendermode {}'.format(rendermode),
                                   '--colormap {}'.format(colormap),
                                   '--logfile {}'.format(logfile),
                                   '--dataset {}'.format(data),
                                   '--criterion {}'.format(criterion),
                                   '--numsamples {}'.format(n),
                                   '--seed {}'.format(s),
                                   '--ranklog' if args.ranklog else ('--noisytest {}'.format(args.noisytest) if args.noisytest else '')]
                            f.write(' '.join(cmd) + '\n')

        print('Done generating experiment scripts.')




    def analyze_log(args):
        # args is a argparse.Namespace object.
        assert os.path.isfile(args.logfile), 'Error! no log file to analyze at "{}"'.format(args.logfile)
        logdir = os.path.dirname(args.logfile) # draw figure next to analzed log, ie within this folder

        #some helper fxns
        def concat_helper(lists): return lists[0] + lists[1] # helper fxn due to the inavailability of unpacking inlist comprehensions
        def color_per_criterion(crit): return {'lrp':'red','weight':'black', 'grad':'green', 'taylor':'blue'}[crit]
        def correlation(data1, data2, measure='spearman'): return {'spearman':spearmanr, 'kendalltau':kendalltau}[measure](data1,data2)[0] # index 0: only collect correlation, not p-value
        def add_to_dict(result_dict, keylist, value):
            #dynamically expands a dictionary
            if len(keylist) == 1:
                if keylist[0] in result_dict:
                    result_dict[keylist[0]] += [value]
                else:
                    result_dict[keylist[0]] = [value]
            else:
                if not keylist[0] in result_dict:
                    result_dict[keylist[0]] = {}
                add_to_dict(result_dict[keylist[0]], keylist[1::], value)



        # read and parse the log.
        # sample log line:
        # dataset:circle-criterion:weight-n:2-s:0-scenario:train-stage:pre 100.0
        with open(args.logfile, 'rt') as f:
            data = f.read().split('\n')
            data = [concat_helper([[c.split(':')[-1] for c in w.split('-')] for w in l.split()]) for l in data if len(l)>0]

        # subscriptable array with field indices as below
        dset, crit, n, seed, scenario, stage, value = range(7) # field names as indices
        dset_t, crit_t, n_t, seed_t, scenario_t, stage_t, value_t = str, str, float, float, str, str, float # "natural" data types per field. final float assumes "accuracy" case.
        data = np.array(data)

        # (re)normalize sample count to "per class"
        data[:,n] = data[:,n].astype(n_t)/(2 + 2*(data[:,dset]=='mult'))

        if args.ranklog:
            # tables to produce
            #
            # over n in reference_sample_counts:
            #     - rank corellation
            #     method vs method (except for identical seed, except for weight). one table, since for computing the rank, one has to consider all neurons
            #
            #    for k in set_sizes_of_k
            #        - set intersection
            #        method vs method, for different set sizes of the first k (= least important  k, also last k = most important k) neurons/filters

            # filter out irrelevant stuff
            data = data[data[:,scenario]=='rankselection']
            seeds = np.unique(data[:,seed]) # NOTE set to or lower [:10] for debugging
            corellation_measures = ['spearman', 'kendalltau']
            set_sizes = [125, 250, 500, 1000] #set sizes for intersection computation


            print('Computing Rank Corellation and Set Intersection Scores')
            for dset_name in tqdm.tqdm(valid_datasets, desc="datasets",leave=False): # approach: progressively filter out data to mimize search times a iteration depth increases
                current_data = data[data[:,dset] == dset_name]
                assert current_data.shape[0] > 0, "Error! current_data empty after dset filtering"
                for n_samples in tqdm.tqdm(np.unique(data[:,n])[np.argsort(np.unique(data[:,n]).astype(n_t))] ,desc="sample counts",leave=False) :
                    current_data_n = current_data[current_data[:,n] == n_samples]
                    assert current_data_n.shape[0] > 0, "Error! current_data_n empty after sample filtering"
                    # gather data and compute three kinds of rank corellation and set Itersection (case four can be found further below)
                    # 1) c1 == c2: comparison across seeds with s1!=s2 (else they are identical).
                    # 2) c1 != c2 or c1 == c2: comparison across all seeds
                    # 3) c1 != c2 or c1 == c2: comparison across same seed
                    case_1_results = {} # {c1:{c2:{correlation_or_intersection_measure:[list over all the seeds with s1 != s2]}}}
                    case_2_results = {} # {c1:{c2:{correlation_or_intersection_measure:[list over all the seeds]}}}
                    case_3_results = {} # {c1:{c2:{correlation_or_intersection_measure:[list over all the seeds with s1 == s2]}}}

                    for c1, c2 in tqdm.tqdm(list(itertools.product(valid_criteria, valid_criteria)), desc='criteria combinations',leave=False):
                        c1_data = current_data_n[(current_data_n[:,crit] == c1)]
                        c2_data = current_data_n[(current_data_n[:,crit] == c2)]
                        assert c1_data.shape[0] > 0, "Error! c1_data empty after criterion filtering"
                        assert c2_data.shape[0] > 0, "Error! c2_data empty after criterion filtering"

                        for s1, s2 in tqdm.tqdm(list(itertools.product(seeds, seeds)), desc="random seed combinations",leave=False):
                            data1 = json.loads(c1_data[c1_data[:,seed]==s1][0,value])
                            data2 = json.loads(c2_data[c2_data[:,seed]==s2][0,value])
                            scorr = correlation(data1, data2, 'spearman')
                            kcorr = correlation(data1, data2, 'kendalltau')

                            tmp_set_intersections = {} # {first-<k>/last-<k>:[list over all seeds]}
                            for k in set_sizes:
                                firstk_intersection_coverage    = len(set(data1[:k]).intersection(data2[:k]))/k
                                lastk_intersection_coverage     = len(set(data1[-k:]).intersection(data2[-k:]))/k
                                tmp_set_intersections['first-{}'.format(k)] = firstk_intersection_coverage
                                tmp_set_intersections['last-{}'.format(k)] = lastk_intersection_coverage

                            if c1 == c2 and s1 != s2: # case 1
                                add_to_dict(case_1_results, [c1, c2, 'spearman'], scorr)
                                add_to_dict(case_1_results, [c1, c2, 'kendalltau'], kcorr)
                                for k in tmp_set_intersections.keys():
                                    add_to_dict(case_1_results, [c1, c2, k], tmp_set_intersections[k])

                            else: # case 2 & case 3
                                add_to_dict(case_2_results, [c1, c2, 'spearman'], scorr)
                                add_to_dict(case_2_results, [c1, c2, 'kendalltau'], kcorr)
                                for k in tmp_set_intersections.keys():
                                    add_to_dict(case_2_results, [c1, c2, k], tmp_set_intersections[k])

                                if s1 == s2: # case 3 only.
                                    add_to_dict(case_3_results, [c1, c2, 'spearman'], scorr)
                                    add_to_dict(case_3_results, [c1, c2, 'kendalltau'], kcorr)
                                    for k in tmp_set_intersections.keys():
                                        add_to_dict(case_3_results, [c1, c2, k], tmp_set_intersections[k])


                    # whole tables per measure.
                    # for each combination of dset and num_samples
                    # write out case_results here.
                    with open('{}/rank_and_set-{}.txt'.format(logdir, dset_name), 'at') as f:
                        ##
                        ## CASE 1 RESULTS
                        ##
                        header_template = '# {} n={} case 1: self-compare criteria across random seeds -> check pruning consistency'.format(dset_name, n_samples).upper()
                        header_support = '#' * len(header_template)
                        header = '\n'.join([header_support, header_template, header_support])

                        # assumption: all pairs of stuff have been computed on the same things
                        criteria = list(case_1_results.keys())
                        all_measures = list(list(list(case_1_results.values())[0].values())[0].keys())

                        f.write(header)
                        f.write('\n'*3)

                        for m in all_measures:
                            t = PrettyTable()
                            t.field_names = [m] + criteria
                            for i, c in enumerate(criteria):
                                val = np.mean(case_1_results[c][c][m])
                                std = np.std(case_1_results[c][c][m])
                                t.add_row([c] + i*[''] + ['{:.3f}+-{:.3f}'.format(val, std)] + ['']*(len(criteria)-1-i))

                            f.write(str(t))
                            f.write('\n'*2)


                        ##
                        ## CASE 2
                        ##
                        header_template = '# {} n={} case 2: cross-compare criteria across all random seeds -> check relationship between criteria'.format(dset_name, n_samples).upper()
                        header_support = '#' * len(header_template)
                        header = '\n'.join([header_support, header_template, header_support])

                        f.write(header)
                        f.write('\n'*3)

                        for m in all_measures:
                            t = PrettyTable()
                            t.field_names = [m] + criteria
                            for cr in criteria:
                                row = [cr]
                                for cc in criteria:
                                    val = np.mean(case_2_results[cr][cc][m])
                                    std = np.std(case_2_results[cr][cc][m])
                                    row += ['{:.3f}+-{:.3f}'.format(val, std)]
                                t.add_row(row)

                            f.write(str(t))
                            f.write('\n'*2)



                        ##
                        ## CASE 3
                        ##
                        header_template = '# {} n={} case 3 cross-compare criteria, same random seeds only! -> check relationship between criteria wrt same data source'.format(dset_name, n_samples).upper()
                        header_support = '#' * len(header_template)
                        header = '\n'.join([header_support, header_template, header_support])

                        f.write(header)
                        f.write('\n'*3)

                        for m in all_measures:
                            t = PrettyTable()
                            t.field_names = [m] + criteria
                            for cr in criteria:
                                row = [cr]
                                for cc in criteria:
                                    val = np.mean(case_3_results[cr][cc][m])
                                    std = np.std(case_3_results[cr][cc][m])
                                    row += ['{:.3f}+-{:.3f}'.format(val,std)]
                                t.add_row(row)

                            f.write(str(t))
                            f.write('\n'*2)



                #
                # Compute cases 4 and 5
                # in addition to above result sets
                #
                for c in tqdm.tqdm(valid_criteria, desc="global criteria consistency", leave=False):
                    # 4) and 5) comparison of neuron rank order for one method, across sample sizes
                    case_4_results = {} # {n1:{n2:{correlation_or_intersection_measure:[list over all the seeds with s1 != s2 or s1 == 2]}}}
                    case_5_results = {} # {n1:{n2:{correlation_or_intersection_measure:[list over all the seeds with s1 == 2]}}}
                    current_data_c = current_data[current_data[:,crit] == c]
                    assert current_data_c.shape[0] > 0, "Error! current_data_c empty after sample filtering"

                    for n1, n2 in tqdm.tqdm(list(itertools.product(np.unique(data[:,n]), np.unique(data[:,n]))), desc="sample set size combinations", leave=False):
                        n1_data = current_data_c[current_data_c[:,n] == n1]
                        n2_data = current_data_c[current_data_c[:,n] == n2]
                        assert n1_data.shape[0] > 0, "Error! n1_data empty after criterion filtering"
                        assert n2_data.shape[0] > 0, "Error! n2_data empty after criterion filtering"

                        for s1, s2 in tqdm.tqdm(list(itertools.product(seeds, seeds)), desc="random seed combinations", leave=False):
                            data1 = json.loads(n1_data[n1_data[:,seed]==s1][0,value])
                            data2 = json.loads(n2_data[n2_data[:,seed]==s2][0,value])
                            scorr = correlation(data1, data2, 'spearman')
                            kcorr = correlation(data1, data2, 'kendalltau')

                            tmp_set_intersections = {} # {first-<k>/last-<k>:[list over all seeds]}
                            for k in set_sizes:
                                firstk_intersection_coverage    = len(set(data1[:k]).intersection(data2[:k]))/k
                                lastk_intersection_coverage     = len(set(data1[-k:]).intersection(data2[-k:]))/k
                                tmp_set_intersections['first-{}'.format(k)] = firstk_intersection_coverage
                                tmp_set_intersections['last-{}'.format(k)] = lastk_intersection_coverage

                            add_to_dict(case_4_results, [n1, n2, 'spearman'], scorr)
                            add_to_dict(case_4_results, [n1, n2, 'kendalltau'], kcorr)
                            for k in tmp_set_intersections.keys():
                                    add_to_dict(case_4_results, [n1, n2, k], tmp_set_intersections[k])

                            if s1 == s2:
                                add_to_dict(case_5_results, [n1, n2, 'spearman'], scorr)
                                add_to_dict(case_5_results, [n1, n2, 'kendalltau'], kcorr)
                                for k in tmp_set_intersections.keys():
                                        add_to_dict(case_5_results, [n1, n2, k], tmp_set_intersections[k])


                    # write out results for  cases 4 and 5
                    # whole tables per criterion.
                    # for each combination of num_samples, over the seeds
                    # write out case_results here.
                    with open('{}/rank_and_set-{}.txt'.format(logdir, dset_name), 'at') as f:
                        ##
                        ## CASE 4 RESULTS
                        ##
                        header_template = '# {} and {} case 4: self-compare criteria across sample sizes (and random seeds) -> check pruning consistency'.format(dset_name, c).upper()
                        header_support = '#' * len(header_template)
                        header = '\n'.join([header_support, header_template, header_support])

                        # assumption: all pairs of stuff have been computed on the same things
                        criterion = c
                        all_n_samples = np.array(list(case_4_results.keys()))
                        all_n_samples = list(all_n_samples[np.argsort(all_n_samples.astype(np.float32))]) #order ascendingly
                        all_measures = list(list(list(case_4_results.values())[0].values())[0].keys())

                        f.write(header)
                        f.write('\n'*3)

                        for m in all_measures:
                            t = PrettyTable()
                            t.field_names = ['{}:{}'.format(criterion, m)] + all_n_samples
                            for nr in all_n_samples:
                                row = [nr]
                                for nc in all_n_samples:
                                    val = np.mean(case_4_results[nr][nc][m])
                                    std = np.std(case_4_results[nr][nc][m])
                                    row += ['{:.3f}+-{:.3f}'.format(val,std)]
                                t.add_row(row)

                            f.write(str(t))
                            f.write('\n'*2)


                        ##
                        ## CASE 5 RESULTS
                        ##
                        header_template = '# {} and {} case 5: self-compare criteria across sample sizes (same seed only) -> check pruning consistency'.format(dset_name, c).upper()
                        header_support = '#' * len(header_template)
                        header = '\n'.join([header_support, header_template, header_support])

                        f.write(header)
                        f.write('\n'*3)

                        for m in all_measures:
                            t = PrettyTable()
                            t.field_names = ['{}:{}'.format(criterion, m)] + all_n_samples
                            for nr in all_n_samples:
                                row = [nr]
                                for nc in all_n_samples:
                                    val = np.mean(case_5_results[nr][nc][m])
                                    std = np.std(case_5_results[nr][nc][m])
                                    row += ['{:.3f}+-{:.3f}'.format(val,std)]
                                t.add_row(row)

                            f.write(str(t))
                            f.write('\n'*2)











        else:
            # analyze accuracy post-pruning wrt n and criterion here

            #now draw some line plots
            for dset_name in np.unique(data[:,dset]):
                for scenario_name in ['train', 'test']:
                    fig = plt.figure(figsize=(3.5,3.5))
                    plt.subplots_adjust(left=0.19, right=0.99, top=0.93, bottom=0.13)
                    plt.title('{}, on {} data'.format(dset_name, scenario_name))
                    plt.xlabel('samples used to compute criteria')
                    plt.ylabel('performance after pruning in %')

                    current_data = data[(data[:,dset] == dset_name) *(data[:,scenario] == scenario_name)] # get the currently relevant data for "this" line plot
                    x = current_data[:,n].astype(n_t)
                    x_min = x.min()
                    x_max = x.max()

                    # draw baseline (original model performance, as average with standard deviation (required for test setting))
                    #y_baseline = current_data[current_data[:,stage] == 'pre'][:,acc].astype(acc_t)
                    data_baseline = current_data[current_data[:,stage] == 'pre']
                    x_baseline = data_baseline[:,n]

                    # compute average values for y per x.
                    y_baseline_avg = np.array([np.mean(data_baseline[data_baseline[:,n]==xi,value].astype(value_t)) for xi in np.unique(x_baseline)])
                    y_baseline_std = np.array([np.std(data_baseline[data_baseline[:,n]==xi,value].astype(value_t)) for xi in np.unique(x_baseline)])
                    x_baseline = np.unique(x_baseline).astype(n_t)

                    #sort wrt ascending x
                    ii = np.argsort(x_baseline)
                    x_baseline = x_baseline[ii]
                    y_baseline_avg = y_baseline_avg[ii]
                    y_baseline_std = y_baseline_std[ii]

                    plt.fill_between(x_baseline, y_baseline_avg - y_baseline_std, np.minimum(y_baseline_avg + y_baseline_std,100), color='black', alpha=0.2)
                    plt.plot(x_baseline, y_baseline_avg, '--', color='black', label='no pruning')

                    # print out some stats for the table/figure description
                    for i in range(x_baseline.size):
                        print('dataset={}, stage=pre, no pruning, n={} : {} acc = {:.2f}'.format(dset_name, x[i], scenario_name, y_baseline_avg[i]))
                    print()

                    #draw achual model performance after pruning, m'lady *heavy breathing*
                    for crit_name in np.unique(current_data[:,crit]):
                        tmp = current_data[(current_data[:,stage] == 'post') * (current_data[:,crit] == crit_name)]
                        x = tmp[:,n]

                        # compute average values for y per x.
                        y_avg = np.array([np.mean(tmp[tmp[:,n]==xi,value].astype(value_t)) for xi in np.unique(x)])
                        y_std = np.array([np.std(tmp[tmp[:,n]==xi,value].astype(value_t)) for xi in np.unique(x)])
                        x = np.unique(x).astype(n_t)

                        #sort wrt ascending x
                        ii = np.argsort(x)
                        x = x[ii]
                        y_avg = y_avg[ii]
                        y_std = y_std[ii]

                        #plot the lines
                        color = color_per_criterion(crit_name)
                        plt.fill_between(x, y_avg-y_std, np.minimum(y_avg+y_std,100), color=color, alpha=0.2)
                        plt.plot(x, y_avg, color=color, label=crit_name)
                        plt.xticks(x,[int(i) if i in [10,50,100,200] else '' for i in x], ha='right')
                        plt.gca().xaxis.grid(True)
                        plt.legend(loc='lower right')

                        # print out some stats for the table/figure description
                        for i in range(x.size):
                            print('dataset={}, stage=post, crit={}, n={} : {} acc = {:.2f}'.format(dset_name, crit_name, x[i], scenario_name, y_avg[i]))
                        print()

                    plt.xlim([x_min, x_max])
                    #save figure
                    figname = '{}/{}-{}.svg'.format(logdir, dset_name, scenario_name)
                    print('Saving result figure to {}'.format(figname))
                    plt.savefig(figname)
                    plt.show()
                    plt.close()








    ###############################
    # actual "main" part of main.
    ###############################


    parser = argparse.ArgumentParser(description = 'Neural Network Pruning Toy experiment')
    parser.add_argument('--dataset',    '-d',   type=str, default='mult',         help='The toy dataset to use. Choices: {}'.format(', '.join(valid_datasets)))
    parser.add_argument('--criterion',  '-c',   type=str, default='lrp',          help='The criterion to use for pruning. Choices: {}'.format(', '.join(valid_criteria)))
    parser.add_argument('--numsamples', '-n',   type=int, default=5,              help='Number of training samples to use for computing the pruning criterion.')
    parser.add_argument('--seed',       '-s',   type=int, default=1,              help='Random seed used for (random) sample selection for pruning criterion computation/testing.')
    parser.add_argument('--rendermode', '-r',   type=str, default='none',         help='Is result visualization desired? Choices: {}'.format(', '.join(valid_rendermodes)))
    parser.add_argument('--colormap',   '-cm',  type=str, default='Dark2',        help='The colormap to use for rendering the output figures. Must be a valid choice from matplotlib.cm .')
    parser.add_argument('--logfile',    '-l',   type=str, default='./log.txt',    help='Output log file location. Results will be appended. File location (folder) must exist!')
    parser.add_argument('--generate',   '-g',   action='store_true',              help='Calls a function to generate a bunch of parameterized function calls and prepares output locations for the scripts. Recommendation: First call this tool with "-g", then execute the generated scripts. If --generate is passed, the script will only generate the scripts and then terminate, disregarding all other settings.')
    parser.add_argument('--analyze',    '-a',   action='store_true',              help='Calls a function to analyze the previously generated log file. If --analyze is passed (but not --generate) the script will analyze the log specified via --logdir and draw some figures or write some tables.')
    parser.add_argument('--ranklog',    '-rl',  action='store_true',              help='Triggers a generation of scripts (when using -g), and an evaluation output and analysis (when using -a) for neuron rank corellations and and neuron set intersection.')
    parser.add_argument('--noisytest',  '-nt',  type=float, default=0.0,          help='The -nt parameter specifies the intensity of some EXTRA gaussian noise added to the dataset. That is, given the parameter >0, a secondary larger test set will be generated just for the purpose of testing the model (not for pruning).')
    args = parser.parse_args()


    # catch the "generate calls" functionality
    if args.generate:
        generate_calls(args)
        exit()

    # catch the "analyze logs" functionality
    if args.analyze:
        analyze_log(args)
        exit()


    # verify parameter choices
    assert args.dataset     in valid_datasets,      'Invalid dataset choice "{}". Must be from {}'.format(args.dataset, valid_datasets)
    assert args.criterion   in valid_criteria,      'Invalid pruning criterion "{}". Must be from {}'.format(args.criterion, valid_criteria)
    assert args.numsamples  > 0,                    'Number of samples (per class) used for pruning criterion computation must be > 0, but was {}'.format(args.num_samples)
    assert args.rendermode  in valid_rendermodes,   'Invalid render mode "{}". Must be from {}'.format(args.rendermode, valid_rendermodes)
    assert args.colormap in plt.colormaps(),        'Invalid colormap choice "{}". Must be from matplotlib.pyplot.colormaps(), i.e. {}'.format(args.colormap, plt.colormaps())

    logdir = os.path.dirname(args.logfile)
    assert os.path.isdir(logdir),                   'Log file location "{}" does not exist (You are here: "{}"). Will not be able to create log file and/or figures! Make sure your target folder exists (to avoid spam)'.format(logdir, os.getcwd())

    model = Net(num_class=num_classes[args.dataset])
    if torch.cuda.is_available(): model = model.cuda()

    fine_tuner = PruningFineTuner(model,
                                  dataset=args.dataset,
                                  criterion=args.criterion,
                                  n_samples=args.numsamples*num_classes[args.dataset], # adjust sample size to number of classes
                                  random_seed=args.seed,
                                  render=args.rendermode,
                                  color_map=args.colormap,
                                  log_file=args.logfile,
                                  rank_analysis=args.ranklog,
                                  noisy_test=args.noisytest)

    fine_tuner.evaluate_and_visualize(scenario='train')
    fine_tuner.evaluate_and_visualize(scenario='test')
    fine_tuner.prune()
    fine_tuner.evaluate_and_visualize(scenario='train')
    fine_tuner.evaluate_and_visualize(scenario='test')
    fine_tuner.close()

    print('Done for {}'.format(fine_tuner.experiment_name))

