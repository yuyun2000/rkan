import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

#docker run -t -i --privileged -v /dev/bus/usb/:/dev/bus/usb/ -v /home/y/Desktop/:/home/y/ rknn-toolkit2:1.5.0-cp36 /bin/bash

ONNX_MODEL = 'kan.onnx'
RKNN_MODEL = 'kan.rknn'


QUANTIZE_ON = False


if __name__ == '__main__':

    # Create RKNN object
    #rknn = RKNN(verbose='Debug',verbose_file='./build.log')
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    rknn.config(target_platform='rk3568')
    #rknn.config()
    print('done')

    # Load ONNX model

    print('--> Loading model')

    ret = rknn.load_onnx(model=ONNX_MODEL)

    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3568',device_id='8429eb11f156f7',perf_debug=True)
    #ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')


    # Inference
    print('--> Running model')
    perf_detail = rknn.eval_perf()


    print('done')

    rknn.release()
