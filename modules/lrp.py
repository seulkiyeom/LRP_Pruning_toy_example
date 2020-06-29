import torch
import torch.nn as nn

def lrp(module, R, lrp_var=None, param=None):
    with torch.no_grad():
        if isinstance(module, torch.nn.modules.linear.Linear):  # for Linear
            return Linear(module, R, lrp_var, param)
        elif isinstance(module, torch.nn.modules.conv.Conv2d):  # for Conv
            return Convolution(module, R, lrp_var, param)
        elif isinstance(module, torch.nn.modules.activation.ReLU) or isinstance(module, torch.nn.modules.dropout.Dropout):
            return R
        elif isinstance(module, torch.nn.modules.activation.LogSoftmax):
            return module.input * R
        elif isinstance(module, torch.nn.modules.pooling.AvgPool2d) or isinstance(module,
                                                                                  torch.nn.modules.pooling.MaxPool2d):
            return Pooling(module, R, lrp_var, param)
        else:
            NameError("No function")

def gradprop_linear(weight, DY):
    return torch.mm(DY, weight)

def Linear(module, R, lrp_var=None, param=None):
    R_shape = R.shape
    if len(R_shape) != 2:
        output_shape = module.output.shape
        R = torch.reshape(R, output_shape)

    if lrp_var is None or lrp_var.lower() == 'none' or lrp_var.lower() == 'simple':
        Z = torch.nn.functional.linear(module.input, module.weight) + 1e-9
        S = R / Z
        C = torch.mm(S, module.weight)
        Rn = module.input * C
        return Rn

    elif lrp_var.lower() == 'alphabeta' or lrp_var.lower() == 'alpha':
        alpha = param
        beta = 1 - alpha

        V = module.weight
        VP = module.weight.clamp(min=0.0)
        VN = module.weight.clamp(max=0.0)

        X = module.input + 1e-9

        ZA = torch.nn.functional.linear(X, VP)
        ZB = torch.nn.functional.linear(X, VN)

        SA = alpha * R / ZA
        SB = beta * R / ZB

        Rn = X * (gradprop_linear(VP, SA) + gradprop_linear(VN, SB))

        return Rn

    elif lrp_var.lower() == 'z' or lrp_var.lower() == 'epsilon':
        V = module.weight.clamp(min=0.0)

        Z = torch.nn.functional.linear(module.input, V) + 1e-9
        S = R / Z
        C = torch.mm(S, V)
        Rn = module.input * C
        return Rn

    elif lrp_var.lower() == 'w^2' or lrp_var.lower() == 'ww':
        return None
        # return _ww_lrp()

def Pooling(module, R, lrp_var=None, param=None):
    R_shape = R.size()
    output_shape = module.output.shape
    if len(R_shape) != 4:
        R = torch.reshape(R, output_shape)
    N, NF, Hout, Wout = R.size()

    if isinstance(module, torch.nn.modules.pooling.AvgPool2d): #there is built-in avgunpool method yet!
        pool = nn.AvgPool2d(module.kernel_size, stride=module.stride, padding=module.padding,
                                           ceil_mode=module.ceil_mode,
                                           count_include_pad=module.count_include_pad)
    elif isinstance(module, torch.nn.modules.pooling.MaxPool2d):
        pool = nn.MaxPool2d(module.kernel_size, stride=module.stride, padding=module.padding,
                            dilation=module.dilation, ceil_mode=module.ceil_mode, return_indices=True)
        unpool = nn.MaxUnpool2d(module.kernel_size, stride=module.stride, padding=module.padding)

    Z, indice = pool(module.input)
    S = R / (Z + 1e-9)

    C = unpool(S, indice)
    return module.input * C

def Convolution(module, R, lrp_var=None, param=None):
    R_shape = R.size()
    output_shape = module.output.shape
    if len(R_shape) != 4:
        R = torch.reshape(R, output_shape)
    N, NF, Hout, Wout = R.size()

    if lrp_var is None or lrp_var.lower() == 'none' or lrp_var.lower() == 'simple':
        V = module.weight
        Z = torch.nn.functional.conv2d(module.input, V, stride=module.stride, padding=module.padding,
                                       dilation=module.dilation, groups=module.groups) + 1e-9
        S = R / Z
        C = torch.nn.functional.conv_transpose2d(S, V, stride=module.stride, padding=module.padding,
                                                 dilation=module.dilation, groups=module.groups)
        return module.input * C

    elif lrp_var.lower() == 'alphabeta' or lrp_var.lower() == 'alpha':
        alpha = param
        beta = 1 - alpha

        VP = module.weight.clamp(min=0.0)
        VN = module.weight.clamp(max=0.0)

        X = module.input + 1e-9

        ZA = torch.nn.functional.conv2d(X, VP, stride=module.stride, padding=module.padding,
                                       dilation=module.dilation, groups=module.groups)
        ZB = torch.nn.functional.conv2d(X, VN, stride=module.stride, padding=module.padding,
                                        dilation=module.dilation, groups=module.groups)

        SA = alpha * R / ZA
        SB = beta * R / ZB

        CP = torch.nn.functional.conv_transpose2d(SA, VP, stride=module.stride, padding=module.padding,
                                                 dilation=module.dilation, groups=module.groups)
        CN = torch.nn.functional.conv_transpose2d(SB, VN, stride=module.stride, padding=module.padding,
                                                  dilation=module.dilation, groups=module.groups)

        Rn = X * (CP + CN)

        return Rn

    elif lrp_var.lower() == 'z' or lrp_var.lower() == 'epsilon':
        V = module.weight.clamp(min=0.0)
        Z = torch.nn.functional.conv2d(module.input, V, stride=module.stride, padding=module.padding,
                                       dilation=module.dilation, groups=module.groups) + 1e-9
        S = R / Z
        if module.stride[0] > 1:
            C = torch.nn.functional.conv_transpose2d(S, V, stride=module.stride, padding= module.padding, output_padding=1,
                                           dilation=module.dilation, groups=module.groups)
        else:
            C = torch.nn.functional.conv_transpose2d(S, V, stride=module.stride, padding= module.padding,
                                           dilation=module.dilation, groups=module.groups)
        return module.input * C

    elif lrp_var.lower() == 'w^2' or lrp_var.lower() == 'ww':
        return None

    elif lrp_var.lower() == 'first':
        lowest = -1.0
        highest = 1.0

        V = module.weight
        VN = module.weight.clamp(max=0.0)
        VP = module.weight.clamp(min=0.0)
        X, L, H = module.input, module.input * 0 + lowest, module.input * 0 + highest

        ZX = torch.nn.functional.conv2d(X, V, stride=module.stride, padding=module.padding,
                                        dilation=module.dilation, groups=module.groups)
        ZL = torch.nn.functional.conv2d(L, VP, stride=module.stride, padding=module.padding,
                                        dilation=module.dilation, groups=module.groups)
        ZH = torch.nn.functional.conv2d(H, VN, stride=module.stride, padding=module.padding,
                                        dilation=module.dilation, groups=module.groups)
        Z = ZX - ZL - ZH + 1e-9
        S = R / Z

        C = torch.nn.functional.conv_transpose2d(S, V, stride=module.stride, padding=module.padding,
                                                  dilation=module.dilation, groups=module.groups)
        CP = torch.nn.functional.conv_transpose2d(S, VP, stride=module.stride, padding=module.padding,
                                                 dilation=module.dilation, groups=module.groups)
        CN = torch.nn.functional.conv_transpose2d(S, VN, stride=module.stride, padding=module.padding,
                                                  dilation=module.dilation, groups=module.groups)

        return X * C - L * CP - H * CN