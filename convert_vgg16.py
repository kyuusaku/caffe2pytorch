from utils import *
import torch
from collections import OrderedDict

def parse_param_name(p, var_name):
    a = var_name.strip().split('.')
    name = a[-2]
    if a[-1] == 'weight':
        data = p.conv_kernel(name)
    if a[-1] == 'bias':
    	data = p.conv_biases(name)
    return name, data

if __name__ == '__main__':
    
    prototxt = 'VGG-Places365.prototxt'
    caffemodel = 'vgg16_places365.caffemodel'
    net = load_caffe(prototxt, caffemodel)
    caffe_keys = net.params.keys()
    print(caffe_keys)
    param_provider = CaffeParamProvider(net)
    record_caffe = dict.fromkeys(caffe_keys, 0)

    from vgg16 import VGG16
    model = VGG16()
    py_keys = model.state_dict().keys()
    print(py_keys)

    print('caffe: {} keys, pytoch: {} keys'.format(len(caffe_keys),len(py_keys)))

    new_state_dict = OrderedDict()
    for var_name in py_keys:
        name, data = parse_param_name(param_provider, var_name)
        new_state_dict[var_name] = torch.from_numpy(data).float()
        print('copy param from {} to {}'.format(name, var_name))
        record_caffe[name] = record_caffe[name] + 1

    for key, val in record_caffe.items():
        if val == 0:
            print('warning: ignoring {}'.format(key))

    model_dict = model.state_dict()
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    torch.save(model.state_dict(), 'VGG-Places365.pth')