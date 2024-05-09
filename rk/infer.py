import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN


RKNN_MODEL = 'kan.rknn'
QUANTIZE_ON = False



import time
if __name__ == '__main__':


    rknn = RKNN()
    #rknn = RKNN(verbose='Debug',verbose_file='./runbuild.log')
    rknn.load_rknn(RKNN_MODEL)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3568',device_id='8429eb11f156f7')
    #ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    
    # Set inputs
    input_dict = []
    input_dict.append(np.load('./testin.npy'))


    # Inference
    print('--> Running model')

    outputs = rknn.inference(inputs=input_dict,data_format="nchw")
    print(outputs)

    rknn.release()
