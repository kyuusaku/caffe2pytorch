from utils import *
import torch
from collections import OrderedDict

def parse_param_name(p, var_name):
    name = 'conv%d_%d_%dx%d'

    a = var_name.strip().split('.')
    if len(a) == 2:
        subith_conv = int(int(a[0]) / 3) + 1
        name = name % (1, subith_conv, 3, 3)
        if int(a[0]) == 0:
            name = name + '_s2'
        if a[-1] == 'weight':
            if int(a[0]) % 3 == 0:
                data = p.conv_kernel(name)
            else:
                name = name + '/scale'
                data = p.bn_gamma(name)
    else:
        ith_conv = int(a[0]) - 8
        subith_conv = int(a[1]) + 1
        if int(a[3]) == 1:
            name = name % (ith_conv, subith_conv, 1, 1) + '_proj'
        else:
            subblock = int(int(a[4]) / 3)
            if subblock == 0:
                name = name % (ith_conv, subith_conv, 1, 1) + '_reduce'
            elif subblock == 1:
                name = name % (ith_conv, subith_conv, 3, 3)
            elif subblock == 2:
                name = name % (ith_conv, subith_conv, 1, 1) + '_increase'
        if a[-1] == 'weight':
            if int(a[4]) % 3 == 0:
                data = p.conv_kernel(name)
            else:
                name = name + '/scale'
                data = p.bn_gamma(name)
    if a[-1] == 'bias':
        name = name + '/scale'
        data = p.bn_beta(name)
    elif a[-1] == 'running_mean':
        name = name + '/bn'
        data = p.bn_mean(name)
    elif a[-1] == 'running_var':
        name = name + '/bn'
        data = p.bn_variance(name)

    return name, data

if __name__ == '__main__':
    
    prototxt = 'Places2-365-CNN.prototxt'
    caffemodel = 'Places2-365-CNN.caffemodel'
    net = load_caffe(prototxt, caffemodel)
    caffe_keys = net.params.keys()
    print(caffe_keys)
    param_provider = CaffeParamProvider(net)
    record_caffe = dict.fromkeys(caffe_keys, 0)

    import Places2_365_CNN
    model = Places2_365_CNN.resnet152_places365
    py_keys = model.state_dict().keys()
    print(py_keys)

    print('caffe:{} keys, pytoch: {} keys'.format(len(caffe_keys),len(py_keys))

    new_state_dict = OrderedDict()
    for var_name in py_keys[:]:
        name, data = parse_param_name(param_provider, var_name)
        new_state_dict[var_name] = torch.from_numpy(data).float()
        print('copy param from {} to {}'.format(name, var_name))
        record_caffe[name] = record_caffe[name] + 1

    for key, val in record_caffe.items():
        if val == 0:
            print('warning: ignoring {}'.format(key))

    model.load_state_dict(new_state_dict)
    torch.save(model.state_dict(), 'Places2_365_CNN.pth')