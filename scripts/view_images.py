import sys, os, time, re

import numpy as np
import matplotlib.pyplot as plt 

import scipy.optimize as opti 

from bead_util import find_all_fnames

import sem_util as su




img_dir = '/Users/manifestation/Stanford/beads/photos/sem/20200624_gbeads-7_5um/'
max_file = 1000

substr = '7_5um_5000x_uc'
substr = '7_5um_calibration_5000x_uc'

devlist = []

with_info = True
show_after = True


filenames, _ = find_all_fnames(img_dir, ext='.tif', substr=substr)

# filenames.sort(key = su.get_devnum)

if len(devlist):
    bad_inds = []
    for fileind, filename in enumerate(filenames):
        found = False
        for dev in devlist:
            if dev in filename:
                found = True
                break
        if not found:
            bad_inds.append(fileind)

    for ind in bad_inds[::-1]:
        filenames.pop(ind)



axes = []
for fileind, filename in enumerate(filenames[:max_file]):
    devind = su.get_devnum(filename)
    
    imgobj = su.SEMImage()
    imgobj.load(filename)

    imgobj.rough_calibrate(plot=False)
    imgobj.find_edges(vertical=True, plot=True)
    imgobj.find_edges(horizontal=True, plot=True)
    input()

    fig, ax = plt.subplots(1,1,figsize=(8,8))
    if with_info:
        ax.imshow(imgobj.full_img_arr, cmap='gray')
    else:
        ax.imshow(imgobj.img_arr, cmap='gray')
    try:
      ax.set_title('Device {:d}'.format(devind))
    except TypeError:
      ax.set_title('No Device ID')

    fig.tight_layout()

    if not show_after:
        plt.show()

if show_after:
    plt.show()
