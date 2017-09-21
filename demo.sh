export LD_PRELOAD=libmkl_rt.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl/build/lib

CAFFE_ROOT=/home/lab-qiu.suo/devel/caffe
export PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH

set -x
set -e

LOG=convert.log
python convert_resnet152.py \
2>&1 | tee -i $LOG