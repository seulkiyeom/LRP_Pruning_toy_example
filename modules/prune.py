import torch
from torch.autograd import Variable
from torchvision import models
# import cv2
import sys
import numpy as np
import time


def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]


def prune_conv_layer(model, layer_index, filter_index, cuda_flag = False):
    ''' input parameters
    1. model: 현재 모델
    2. layer_index: 자르고자 하는 layer index
    3. filter_index: 자르고자 하는 layer의 filter index
    '''
    # _, conv = model.features._modules.items()[layer_index]
    _, conv = list(model.features._modules.items())[layer_index] #해당 layer를 우선 끄집어 온다.
    next_conv = None
    offset = 1

    while layer_index + offset < len(model.features._modules.items()): #전체 network의 layer 수보다 클때까지 while문 반복
        # res =  model.features._modules.items()[layer_index+offset]
        res = list(model.features._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d): #현재 layer를 기준으로 다음 layer가 conv layer이냐?
            next_name, next_conv = res
            break
        offset = offset + 1

    #새로운 conv_layer(new_conv)를 다시 생성시킨다.
    #output 쪽의 갯수를 하나 줄여준다.
    new_conv = \
        torch.nn.Conv2d(in_channels=conv.in_channels,
                        out_channels=conv.out_channels - 1,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=True)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    #weight는 해당 filter를 제외하고 총 갯수 - 1 를 넣어준다.
    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)

    #bias도 해당 filter number의 값을 제외하고 총 갯수 -1를 넣어준다.
    bias_numpy = conv.bias.data.cpu().numpy()
    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1:]
    new_conv.bias.data = torch.from_numpy(bias).cuda() if cuda_flag else torch.from_numpy(bias)

    #다음 conv layer도 받는 쪽의 layer 갯수를 줄여준다.
    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - 1,
                            out_channels=next_conv.out_channels,
                            kernel_size=next_conv.kernel_size,
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=True)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index, layer_index + offset],
                             [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features

    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index], \
                             [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear laye found in classifier")
        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                             [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model

def prune_layer_toy(model, layer_index, filter_index, cuda_flag = False):
    ''' input parameters
        1. model: 현재 모델
        2. layer_index: 자르고자 하는 layer index
        3. filter_index: 자르고자 하는 layer의 filter index
        '''
    # _, conv = model.features._modules.items()[layer_index]
    _, dense = list(model.network._modules.items())[layer_index]  # 해당 layer를 우선 끄집어 온다.
    next_dense = None
    offset = 1

    while layer_index + offset < len(model.network._modules.items()):  # 전체 network의 layer 수보다 클때까지 while문 반복
        res = list(model.network._modules.items())[layer_index + offset]
        if isinstance(res[1], torch.nn.modules.linear.Linear):  # 현재 layer를 기준으로 다음 layer가 dense layer이냐?
            next_name, next_dense = res
            break
        offset = offset + 1

    # 새로운 conv_layer(new_conv)를 다시 생성시킨다.
    # output 쪽의 갯수를 하나 줄여준다.
    new_dense = \
        torch.nn.Linear(in_features=dense.in_features,
                        out_features=dense.out_features - 1,
                        bias=True)

    old_weights = dense.weight.data.cpu().numpy()
    new_weights = new_dense.weight.data.cpu().numpy()

    # weight는 해당 filter를 제외하고 총 갯수 - 1 를 넣어준다.
    new_weights[: filter_index, :] = old_weights[: filter_index, :]
    new_weights[filter_index:, :] = old_weights[filter_index + 1:, :]
    new_dense.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)

    # bias도 해당 filter number의 값을 제외하고 총 갯수 -1를 넣어준다.
    bias_numpy = dense.bias.data.cpu().numpy()
    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1:]
    new_dense.bias.data = torch.from_numpy(bias).cuda() if cuda_flag else torch.from_numpy(bias)

    # 다음 conv layer도 받는 쪽의 layer 갯수를 줄여준다.
    if not next_dense is None:
        next_new_dense = \
            torch.nn.Linear(in_features=next_dense.in_features - 1,
                            out_features=next_dense.out_features,
                            bias=True)

        old_weights = next_dense.weight.data.cpu().numpy()
        new_weights = next_new_dense.weight.data.cpu().numpy()

        # for p in next_new_conv.parameters():
        #     p.requires_grad = False

        new_weights[:, : filter_index] = old_weights[:, : filter_index]
        new_weights[:, filter_index:] = old_weights[:, filter_index + 1:]
        next_new_dense.weight.data = torch.from_numpy(new_weights).cuda() if cuda_flag else torch.from_numpy(new_weights)

        next_new_dense.bias.data = next_dense.bias.data

    if not next_dense is None:
        network = torch.nn.Sequential(
            *(replace_layers(model.network, i, [layer_index, layer_index + offset],
                             [new_dense, next_new_dense]) for i, _ in enumerate(model.network)))
        del model.network
        del dense

        model.network = network
    else:
        network = torch.nn.Sequential(
            *(replace_layers(model.network, i, [layer_index],
                             [new_dense]) for i, _ in enumerate(model.network)))
        del model.network
        del dense

        model.network = network

    return model


if __name__ == '__main__':
    model = models.vgg16(pretrained=True)
    model.train()

    t0 = time.time()
    model = prune_conv_layer(model, 28, 10)
    print("The prunning took", time.time() - t0)