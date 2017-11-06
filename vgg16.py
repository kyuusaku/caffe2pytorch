import torch.nn as nn

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]#, 'M']

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = self.make_layers()

    def make_layers(self):
        features = nn.Sequential()
        in_channels = 3
        ith_block = 1
        ith_sublayer = 1
        for v in cfg:
            if v == 'M':
                features.add_module('pool%d'%ith_block, nn.MaxPool2d(kernel_size=2, stride=2))
                ith_block += 1
                ith_sublayer = 1
            else:
                features.add_module('conv%d_%d'%(ith_block,ith_sublayer), 
                                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                features.add_module('relu%d_%d'%(ith_block,ith_sublayer), nn.ReLU(inplace=True))
                ith_sublayer += 1
                in_channels = v
        return features

    def forward(self, x):
        x = self.features(x)
        return x

    