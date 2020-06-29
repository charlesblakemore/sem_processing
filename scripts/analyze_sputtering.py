import sys, os, time, re

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as opti 

import sem_util as su

import peakdetect as pdet
import cv2




predat = pickle.load(open('./pre_sputtering.p', 'rb'))
postdat = pickle.load(open('./post_sputtering.p', 'rb'))



keys = list(predat.keys())

for key in keys:
    print()
    print('DEVICE ', key)

    prechannel = np.mean(predat[key][0])
    prewall1 = np.mean(predat[key][1])
    prewall2 = np.mean(predat[key][2])

    postchannel = np.mean(postdat[key][0])
    postwall1 = np.mean(postdat[key][1])
    postwall2 = np.mean(postdat[key][2])

    print('Channel: {:0.2f} -> {:0.2f}'.format(prechannel, postchannel))
    print(' Wall 1: {:0.2f} -> {:0.2f}'.format(prewall1, postwall1))
    print(' Wall 2: {:0.2f} -> {:0.2f}'.format(prewall2, postwall2))
