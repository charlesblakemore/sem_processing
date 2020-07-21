import sys, os, time, re

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as opti 

from bead_util import find_all_fnames
import sem_util as su

import peakdetect as pdet
import cv2




img_dir = '/Users/manifestation/Stanford/beads/photos/sem/20191014_shield_v2_pc1_second-HF-clean/'
# img_dir = '/Users/manifestation/Stanford/beads/photos/sem/20200109_post-sputtering/'
max_file = 20

substr = '_2000x.'

show = True

save_path = '../data/pre_sputtering_2.p'
# save_path = './post_sputtering.p'


devlist = [#'/dev2/', \
           #'/dev4/',   ###\
           #'/dev7/',   ###\
           #'/dev9/', \
           #'/dev23/', \
           #'/dev29/',   ###\
           #'/dev36/', \
           #'/dev37/', \
           '/dev43/', \
           #'/dev47/',   ###\
           #'/dev49/',   ###\
           #'/dev55/', \
           #'/dev57/', \
           '/dev63/', \
           #'/dev64/', \
           #'/dev65/', \
           #'/dev67/', \
           #'/dev76/', \
          ]






### Get the files and downselect as directed
filenames, _ = find_all_fnames(img_dir, ext='.tif', substr=substr)

### Sort the files by device index, assuming that the string 
### 'devXXX' exists somewhere in the filename
filenames.sort(key = su.get_devnum)

### Downselect
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






## Process each image individually and collect results
data = {}
channels = []
for fileind, filename in enumerate(filenames[:max_file]):
    devind = su.get_devnum(filename)
    
    imgobj = su.SEMImage()
    imgobj.load(filename)

    imgobj.rough_calibrate(plot=False, verbose=True)

    half = int( imgobj.img_arr.shape[1] * 0.5 )

    left_edges = imgobj.find_edges(yinds=(0,200), xinds=(0,half), plot=True, \
                                   vertical=True, edge_width=10, neg_thresh_fac=0.45, 
                                   pos_thresh_fac=0.5, blur_kernel=5)
    right_edges = imgobj.find_edges(yinds=(0,200), xinds=(half,-1), plot=True, \
                                    vertical=True, edge_width=10, neg_thresh_fac=0.45, 
                                    pos_thresh_fac=0.5, blur_kernel=5)

    input()

    edges = left_edges + right_edges
    baddies = []
    for edge_ind, edge in enumerate(edges):
        if not edge[1]:
            baddies.append(edge_ind)
    for baddie in baddies[::-1]:
        edges.pop(baddie)

    delta1 = edges[1][0] - edges[0][0]
    delta2 = edges[2][0] - edges[1][0]
    if len(edges) > 3:
        delta3 = edges[3][0] - edges[2][0]
        nval = 4
    else:
        delta3 = 0
        nval = 3

    if delta3 == 0:
        if delta1 > delta2:
            channel_width = delta1 * imgobj.scale_fac * 1e6
            wall_width1 = 0.0
            wall_width2 = delta2 * imgobj.scale_fac * 1e6
        elif delta1 <= delta2:
            channel_width = delta2 * imgobj.scale_fac * 1e6
            wall_width1 = delta1 * imgobj.scale_fac * 1e6
            wall_width2 = 0.0

    else:
        channel_width = delta2 * imgobj.scale_fac * 1e6
        wall_width1 = delta1 * imgobj.scale_fac * 1e6
        wall_width2 = delta3 * imgobj.scale_fac * 1e6

    channels.append(channel_width)

    print()
    print('DEV{:d}'.format(devind))
    print('    channel: {:0.1f} um'.format(channel_width))
    print('    walls: {:0.1f} um, {:0.1f} um'.format(wall_width1, wall_width2))


    if devind not in data.keys():
        data[devind] = [[], [], []]
    data[devind][0].append(channel_width)
    data[devind][1].append(wall_width1)
    data[devind][2].append(wall_width2)

    if show:
        plt.imshow(imgobj.img_arr, cmap='gray')
        for i in range(nval):
            plt.axvline(edges[i][0], ls=':', lw=2, color='r')
        plt.show()

    # print(top_peaks)
    # plt.plot(top_marginalized)
    # #plt.plot(right_marginalized)
    # plt.show()

# plt.show()

pickle.dump(data, open(save_path, 'wb'))

keys = list(data.keys())
keys.sort()

for key in keys:
    print(key, data[key])

plt.hist(channels, 60)
plt.show()