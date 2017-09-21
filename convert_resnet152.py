from utils import *
import torch
from collections import OrderedDict

if __name__ == '__main__':
    
    prototxt = 'Places2-365-CNN.prototxt'
    caffemodel = 'Places2-365-CNN.caffemodel'
    net = load_caffe(args.prototxt, args.caffemodel)
    caffe_keys = net.
    print(caffe_keys)
    param_provider = CaffeParamProvider(net)

    import Places2_365_CNN
    model = Places2_365_CNN.resnet152_places365
    py_keys = model.state_dict().keys()
    print(py_keys)

    print('{}{}'.format())

    new_state_dict = OrderedDict()
    for var_name in keys[:]:
    	data = parse_param()
    	new_state_dict[var_name] =

    model.load_state_dict(new_state_dict)
    torch.save(model.state_dict(), 'Places2_365_CNN.pth')