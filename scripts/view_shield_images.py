import sys, os, time, re

import numpy as np
import matplotlib.pyplot as plt 

import scipy.optimize as opti 

from bead_util import find_all_fnames

import sem_util as su




img_dir = '/Users/manifestation/Stanford/beads/photos/sem/20191014_shield_v2_pc1_second-HF-clean/'
max_file = 1000

substr = '2000x_40deg.tif'

devlist = [#'/dev4/', \
           #'/dev7/', \
           #'/dev9/', \
           #'/dev23/', \
           #'/dev29/', \
           #'/dev37/', \
           #'/dev47/', \
           #'/dev49/', \
           #'/dev55/', \
           '/dev57/', \
           #'/dev64/', \
           #'/dev65/', \
           #'/dev67/', \
           #'/dev76/', \
          ]

with_info = True
show_after = True


filenames, _ = find_all_fnames(img_dir, ext='.tif', substr=substr)

filenames.sort(key = su.get_devnum)

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
    #imgobj.calibrate(plot=False)

    fig, ax = plt.subplots(1,1,figsize=(8,8))
    if with_info:
        ax.imshow(imgobj.full_img_arr, cmap='gray')
    else:
        ax.imshow(imgobj.img_arr, cmap='gray')
    ax.set_title('Device {:d}'.format(devind))
    fig.tight_layout()

    if not show_after:
        plt.show()

if show_after:
    plt.show()
