import argparse
import os
import os.path as osp

import caffe

def load_caffe(proto, weight):
	caffe.set_mode_cpu()
    net = caffe.Net(proto, weight, caffe.TEST)
    return net

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Covert caffemodel to pytorch")
    parser.add_argument('--caffemodel',type=str,help='path to the caffemodel')
    parser.add_argument('--prototxt',type=str,help='path to validation data')
    args = parser.parse_args()

    net = load_caffe(args.prototxt, args.caffemodel)