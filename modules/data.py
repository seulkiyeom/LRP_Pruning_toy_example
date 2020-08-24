'''
Codes for loading the MNIST data
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import os, fnmatch
import numpy
import torch
from scipy.io import loadmat
from torchvision import datasets, transforms
import imageio


class PartDataset(torch.utils.data.Dataset):
    '''
    Partial Dataset:
        Extract the examples from the given dataset,
        starting from the offset.
        Stop if reach the length.
    '''

    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length
        super(PartDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.dataset[i + self.offset]


class MycustomVOC2007(torch.utils.data.Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
                          for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, transform=None, target_transform=None):

        self.root = os.path.expanduser(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        super(MyCustomDataset, self).__init__()

        if self.train:
            self.train_labels = []
            self.train_data = numpy.load(data_path)['arr_0.npy']
            self.train_labels = numpy.load(data_path)['arr_1.npy']
            # self.train_labels = convert_to_one_hot(self.train_labels, num_classes=2)
        else:
            self.test_labels = []
            self.test_data = numpy.load(data_path)['arr_0.npy']
            self.test_labels = numpy.load(data_path)['arr_1.npy']
            # self.test_labels = convert_to_one_hot(self.test_labels, num_classes=2)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class MyCustomDataset_scene(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        super(MyCustomDataset_scene, self).__init__()

        categories = [x[0] for x in os.walk(data_path) if x[0]][1:]
        categories = [c for c in categories]

        # # load and resize all images
        # width = 256
        # height = 256
        # for c, category in enumerate(categories):
        #     images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(category)
        #               for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
        #     count = 0
        #     for img_path in images:
        #         img = Image.open(img_path).resize((width, height), Image.BILINEAR)
        #         img.save(category + '/' + str(count) + '_resize.jpg')
        #         count += 1

        num_classes = len(categories)
        IMAGE_SIZE = 256
        IMAGE_DEPTH = 3
        NUM_TRAIN_SAMPLES = 917
        NUM_TEST_SAMPLES = 3568
        train_rate = 0.2
        test_rate = 1.0 - train_rate

        t_count = 0
        tt_count = 0

        self.train_data = numpy.zeros((NUM_TRAIN_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype='uint8')
        self.train_labels = numpy.zeros(NUM_TRAIN_SAMPLES, dtype='int64')
        self.test_data = numpy.zeros((NUM_TEST_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype='uint8')
        self.test_labels = numpy.zeros(NUM_TEST_SAMPLES, dtype='int64')

        for c, category in enumerate(categories):
            images = fnmatch.filter(os.listdir(category), '*_resize.jpg')
            n_train = numpy.ceil(images.__len__() * train_rate)
            # import random #do not process for pruning
            # random.shuffle(images)
            for i, img_path in enumerate(images):
                if i <= n_train:
                    temp = imageio.imread(category + '/' + img_path)
                    if numpy.ndim(temp) is not 3:
                        temp = numpy.stack((temp, temp, temp), axis=2)
                    self.train_data[t_count] = temp
                    self.train_labels[t_count] = c
                    t_count += 1

                else:
                    temp = imageio.imread(category + '/' + img_path)
                    if numpy.ndim(temp) is not 3:
                        temp = numpy.stack((temp, temp, temp), axis=2)
                    self.test_data[tt_count] = temp
                    self.test_labels[tt_count] = c
                    tt_count += 1

        if self.train:
            del self.test_labels, self.test_data
        else:
            del self.train_labels, self.train_data

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class MyCustomDataset_event(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        super(MyCustomDataset_event, self).__init__()

        categories = [x[0] for x in os.walk(data_path) if x[0]][1:]
        categories = [c for c in categories]

        # # load and resize all images
        # width = 256
        # height = 256
        # for c, category in enumerate(categories):
        #     images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(category)
        #               for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
        #     count = 0
        #     for img_path in images:
        #         img = Image.open(img_path).resize((width, height), Image.BILINEAR)
        #         img.save(category + '/' + str(count) + '_resize.jpg')
        #         count += 1

        num_classes = len(categories)
        IMAGE_SIZE = 256
        IMAGE_DEPTH = 3
        NUM_TRAIN_SAMPLES = 640
        NUM_TEST_SAMPLES = 939
        train_rate = 0.4
        test_rate = 1.0 - train_rate

        t_count = 0
        tt_count = 0

        self.train_data = numpy.zeros((NUM_TRAIN_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype='uint8')
        self.train_labels = numpy.zeros(NUM_TRAIN_SAMPLES, dtype='int64')
        self.test_data = numpy.zeros((NUM_TEST_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype='uint8')
        self.test_labels = numpy.zeros(NUM_TEST_SAMPLES, dtype='int64')

        for c, category in enumerate(categories):
            images = fnmatch.filter(os.listdir(category), '*_resize.jpg')
            n_train = 79 #training sample: 70 per each class
            # import random #do not process for pruning
            # random.shuffle(images)
            for i, img_path in enumerate(images):
                if i <= n_train:
                    temp = imageio.imread(category + '/' + img_path)
                    if numpy.ndim(temp) is not 3:
                        temp = numpy.stack((temp, temp, temp), axis=2)
                    self.train_data[t_count] = temp
                    self.train_labels[t_count] = c
                    t_count += 1

                else:
                    temp = imageio.imread(category + '/' + img_path)
                    if numpy.ndim(temp) is not 3:
                        temp = numpy.stack((temp, temp, temp), axis=2)
                    self.test_data[tt_count] = temp
                    self.test_labels[tt_count] = c
                    tt_count += 1

        if self.train:
            del self.test_labels, self.test_data
        else:
            del self.train_labels, self.train_data

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class MyCustomDataset_flower(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, transform=None, target_transform=None):

        self.root = os.path.expanduser(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        super(MyCustomDataset_flower, self).__init__()

        #  # phase 1)save data and its label as txt
        #
        # setid = loadmat(data_path + 'setid.mat')
        #
        # # Data index
        # idx_train = setid['trnid'][0] - 1
        # idx_test = setid['tstid'][0] - 1
        # idx_valid = setid['valid'][0] - 1
        #
        # image_labels = loadmat(data_path + 'imagelabels.mat')['labels'][0] - 1
        # import glob
        # files = sorted(glob.glob(data_path + 'jpg/*.jpg'))
        #
        # labels = numpy.array(list(zip(files, image_labels)))
        #
        # def write_set_file(fout, labels):
        #     with open(fout, 'w+') as f:
        #         for label in labels:
        #             f.write('%s %s\n' % (label[0], label[1]))
        #
        # numpy.random.seed(777)
        # idx_train = idx_train[numpy.random.permutation(len(idx_train))]
        # idx_test = idx_test[numpy.random.permutation(len(idx_test))]
        # idx_valid = idx_valid[numpy.random.permutation(len(idx_valid))]
        #
        # write_set_file('train.txt', labels[idx_train, :])
        # write_set_file('test.txt', labels[idx_test, :])
        # write_set_file('valid.txt', labels[idx_valid, :])
        # '''
        #
        #  # phase 2)save resized image
        # from PIL import Image
        # width = 256
        # height = 256
        # temp = []
        # f = open(data_path + 'train.txt', 'r').readlines()
        # for i in range(len(f)):
        #     img = Image.open(f[i].split()[0]).resize((width,height), Image.BILINEAR)
        #     img.save(data_path + 'jpg/resize_' + f[i].split()[0][22:])
        #
        # f = open(data_path + 'valid.txt', 'r').readlines()
        # for i in range(len(f)):
        #     img = Image.open(f[i].split()[0]).resize((width,height), Image.BILINEAR)
        #     img.save(data_path + 'jpg/resize_' + f[i].split()[0][22:])
        #
        # f = open(data_path + 'test.txt', 'r').readlines()
        # for i in range(len(f)):
        #     img = Image.open(f[i].split()[0]).resize((width,height), Image.BILINEAR)
        #     img.save(data_path + 'jpg/resize_' + f[i].split()[0][22:])

        label_name = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
                      'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon',
                      "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower',
                      'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower',
                      'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
                      'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist',
                      'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort',
                      'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
                      'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion',
                      'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura',
                      'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone',
                      'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus',
                      'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose',
                      'thorn apple',
                      'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis',
                      'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress',
                      'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia',
                      'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']

        IMAGE_SIZE = 256  # Random size flower image
        IMAGE_DEPTH = 3
        NUM_TRAIN_SAMPLES = 2040
        NUM_TEST_SAMPLES = 6149
        if self.train:
            self.train_data = numpy.zeros((NUM_TRAIN_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype='uint8')
            self.train_labels = numpy.zeros((NUM_TRAIN_SAMPLES), dtype='int64')
            f = open(data_path + 'train.txt', 'r').readlines()
            train_len = len(f)
            for i in range(len(f)):
                self.train_data[i] = imageio.imread(data_path + 'jpg/resize_' + f[i].split()[0][22:])
                self.train_labels[i] = int(f[i].split()[1])

            f = open(data_path + 'valid.txt', 'r').readlines()
            for i in range(len(f)):
                self.train_data[i + train_len] = imageio.imread(data_path + 'jpg/resize_' + f[i].split()[0][22:])
                self.train_labels[i + train_len] = int(f[i].split()[1])

        else:
            self.test_data = numpy.zeros((NUM_TEST_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype='uint8')
            self.test_labels = numpy.zeros((NUM_TEST_SAMPLES), dtype='int64')
            f = open(data_path + 'test.txt', 'r').readlines()
            for i in range(len(f)):
                self.test_data[i] = imageio.imread(data_path + 'jpg/resize_' + f[i].split()[0][22:])
                self.test_labels[i] = int(f[i].split()[1])

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class MyCustomDataset_caltech(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        super(MyCustomDataset_caltech, self).__init__()

        exclude = ['BACKGROUND_Google', 'Motorbikes', 'airplanes', 'Faces_easy', 'Faces']
        train_split, val_split = 0.7, 0.15

        categories = [x[0] for x in os.walk(data_path) if x[0]][1:]
        categories = [c for c in categories if
                      c not in [os.path.join(data_path, e) for e in exclude]]

        # load all images
        width = 256
        height = 256
        # for c, category in enumerate(categories):
        #     images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(category)
        #               for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
        #     count = 0
        #     for img_path in images:
        #         img = Image.open(img_path).resize((width, height), Image.BILINEAR)
        #         img.save(category + '/' + str(count) + '_resize.jpg')
        #         count += 1

        num_classes = len(categories)
        IMAGE_SIZE = 256
        IMAGE_DEPTH = 3
        NUM_TRAIN_SAMPLES = 4485
        NUM_TEST_SAMPLES = 1724
        train_rate = 0.7
        test_rate = 1.0 - train_rate
        self.train_data = numpy.zeros((NUM_TRAIN_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype='uint8')
        self.train_labels = numpy.zeros(NUM_TRAIN_SAMPLES, dtype='int64')
        self.test_data = numpy.zeros((NUM_TEST_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype='uint8')
        self.test_labels = numpy.zeros(NUM_TEST_SAMPLES, dtype='int64')

        train_switch = False
        test_switch = False
        t_count = 0
        tt_count = 0
        for c, category in enumerate(categories):
            images = fnmatch.filter(os.listdir(category), '*_resize.jpg')
            n_train = numpy.ceil(images.__len__() * train_rate)
            for i, img_path in enumerate(images):
                if i <= n_train:
                    temp = imageio.imread(category + '/' + img_path)
                    if numpy.ndim(temp) is not 3:
                        temp = numpy.stack((temp, temp, temp), axis=2)
                    self.train_data[t_count] = temp
                    self.train_labels[t_count] = c
                    t_count += 1

                else:
                    temp = imageio.imread(category + '/' + img_path)
                    if numpy.ndim(temp) is not 3:
                        temp = numpy.stack((temp, temp, temp), axis=2)
                    self.test_data[tt_count] = temp
                    self.test_labels[tt_count] = c
                    tt_count += 1

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def get_mnist(datapath='../data/mnist/', download=True):
    '''
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    '''
    # MNIST Dataset
    train_dataset = datasets.MNIST(root=datapath,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=download)

    test_dataset = datasets.MNIST(root=datapath,
                                  train=False,
                                  transform=transforms.ToTensor())
    return train_dataset, test_dataset


def get_cifar10(datapath='../data/', download=True):
    '''
    Get CIFAR10 dataset
    '''
    # MNIST Dataset
    train_dataset = datasets.CIFAR10(root=datapath,
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=download)

    test_dataset = datasets.CIFAR10(root=datapath,
                                    train=False,
                                    transform=transforms.ToTensor())
    return train_dataset, test_dataset


def get_medical_data(datapath='../data/data_HE_ts200/', download=True):
    '''
    Get Medical Data from Philipp

    Number of Data for microscopic image
        Train - 0 : 668
              - 1 : 668

              Tot : 1336

        Test  - 0 : 167
              - 1 : 167

              Tot : 334

        Total : 1670 (835 + 835)
    '''
    IMAGE_SIZE = 200
    IMAGE_DEPTH = 3
    NUM_CLASSES = 2
    NUM_TRAIN_SAMPLES = 1336
    NUM_TEST_SAMPLES = 334

    train_dataset = MyCustomDataset(data_path=datapath + 'train.npz', train=True,
                                    transform=transforms.ToTensor())
    test_dataset = MyCustomDataset(data_path=datapath + 'test.npz', train=False,
                                   transform=transforms.ToTensor())

    return train_dataset, test_dataset


def get_cub200_tl(datapath='../data/CUB_200_2011', download=True):
    '''
    Get Bird-200 dataset
    '''
    import modules.cub2011 as cub

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = cub.CUB(root=datapath, is_train=True, data_len=None)
    test_dataset = cub.CUB(root=datapath, is_train=False, data_len=None)

    return train_dataset, test_dataset


def get_cifar10_tl(datapath='../data/', download=True):
    '''
    Get CIFAR10 dataset
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Cifar-10 Dataset
    train_dataset = datasets.CIFAR10(root=datapath,
                                     train=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize
                                     ]),
                                     download=download)

    test_dataset = datasets.CIFAR10(root=datapath,
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    return train_dataset, test_dataset


def get_caltech101_tl(datapath='../data/101_ObjectCategories/', download=True):
    '''
        Get Caltech-101 dataset
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = MyCustomDataset_caltech(data_path=datapath, train=True, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    test_dataset = MyCustomDataset_caltech(data_path=datapath, train=False, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ]))

    return train_dataset, test_dataset


def get_STL10_tl(datapath='../data/', download=True):
    '''
    Get STL10 dataset for transfer learning
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # STL10 Dataset
    train_dataset = datasets.STL10(root=datapath,
                                   split='train',
                                   transform=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize
                                   ]),
                                   download=download)

    test_dataset = datasets.STL10(root=datapath,
                                  split='test',
                                  transform=transforms.Compose([
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                      normalize
                                  ]))
    return train_dataset, test_dataset


def get_LSUN_tl(datapath='../data/', download=True):
    '''
    Get Large-scale Scene Understanding (LSUN) dataset for transfer learning
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # STL10 Dataset
    # train_dataset = datasets.LSUNClass(root=datapath,
    #                                  classes='train',
    #                                  transform=transforms.Compose([
    #                                      transforms.Resize(256),
    #                                      transforms.RandomResizedCrop(224),
    #                                      transforms.RandomHorizontalFlip(),
    #                                      transforms.ToTensor(),
    #                                      normalize
    #                                  ]),
    #                                  download=download)

    train_dataset2 = datasets.LSUN(root=datapath,
                                   classes='train',
                                   transform=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize
                                   ]),
                                   download=download)

    test_dataset = datasets.LSUNClass(root=datapath,
                                      classes='test',
                                      transform=transforms.Compose([
                                          transforms.Resize(224),
                                          transforms.ToTensor(),
                                          normalize
                                      ]))
    return train_dataset, test_dataset


def get_flower102_tl(datapath='../data/oxford102/'):
    '''
    Get CIFAR10 dataset
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = MyCustomDataset_flower(data_path=datapath, train=True, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    test_dataset = MyCustomDataset_flower(data_path=datapath, train=False, transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ]))

    return train_dataset, test_dataset


def get_dataset(datapath='../data/catsNdogs/'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return datasets.ImageFolder(datapath + 'training_set',
                                transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    # transforms.RandomResizedCrop(224),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                ])), \
           datasets.ImageFolder(datapath + 'test_set',
                                transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))

    # return datasets.ImageFolder(train_path,
    #                      transforms.Compose([
    #                          transforms.Resize(224, 224),
    #                          transforms.ToTensor(),
    #                          normalize,
    #                      ])),\
    #        datasets.ImageFolder(test_path,
    #                          transforms.Compose([
    #                              transforms.Resize(224, 224),
    #                              transforms.ToTensor(),
    #                              normalize,
    #                          ])) #for explanation 다 완성되면 나중에 사용할 것

def get_event8_tl(datapath='../data/event_8/'):
    '''
    Get event 8 dataset
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    train_dataset = MyCustomDataset_event(data_path= datapath, train=True, transform=transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.Resize(256),
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ]))
    test_dataset = MyCustomDataset_event(data_path= datapath, train=False, transform=transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))

    return train_dataset, test_dataset


def get_scene15_tl(datapath='../data/15-Scene/'):
    '''
    Get scene 15 dataset
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    train_dataset = MyCustomDataset_scene(data_path= datapath, train=True, transform=transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.Resize(256),
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ]))
    test_dataset = MyCustomDataset_scene(data_path= datapath, train=False, transform=transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))

    return train_dataset, test_dataset

def get_dataset_breakhis(datapath='../data/BreaKHis/'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    Mag_size = 200  # Magnification size: 40, 100, 200, 400
    return datasets.ImageFolder(datapath + str(Mag_size) + 'X/training_set',
                                transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                ])), \
           datasets.ImageFolder(datapath + str(Mag_size) + 'X/test_set',
                                transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))

    # return datasets.ImageFolder(train_path,
    #                      transforms.Compose([
    #                          transforms.Resize(224, 224),
    #                          transforms.ToTensor(),
    #                          normalize,
    #                      ])),\
    #        datasets.ImageFolder(test_path,
    #                          transforms.Compose([
    #                              transforms.Resize(224, 224),
    #                              transforms.ToTensor(),
    #                              normalize,
    #                          ])) #for explanation 다 완성되면 나중에 사용할 것


def get_dataset_mitplaces(datapath='../data/mitplaces/'):
    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return datasets.ImageFolder(datapath + str(Mag_size) + 'X/test_set',
                                transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))


def get_artificial_dataset(nsample, ninfeature, noutfeature):
    '''
    Generate a synthetic dataset.
    '''
    data = torch.randn(nsample, ninfeature).cuda()
    target = torch.LongTensor(
        numpy.random.randint(noutfeature, size=(nsample, 1))).cuda()
    return torch.utils.data.TensorDataset(data, target)


def convert_to_one_hot(Y, num_classes=None):
    Y = numpy.eye(num_classes)[Y.reshape(-1)]
    return Y.astype(int)

class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):

        self.root = os.path.expanduser(data_path)
        self.transform = transform
        super(MyCustomDataset, self).__init__()

        self.X = numpy.load(data_path + '_X.npy')
        self.y = numpy.load(data_path + '_y.npy')
        #
        # self.X = torch.Tensor(self.X)
        # self.y = torch.LongTensor(self.y)

    def __getitem__(self, index):
        img, target = self.X[index], self.y[index]

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.X)

def get_multitoy_data(datapath='mult_', download=True):
    train_dataset = MyCustomDataset(data_path= datapath + 'train')
                                    # transform=transforms.ToPILImage())
    test_dataset = MyCustomDataset(data_path= datapath + 'test')
                                   # transform=transforms.ToPILImage())

    return train_dataset, test_dataset