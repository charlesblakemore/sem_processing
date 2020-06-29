import sys, os, time, re, traceback

import numpy as np
import matplotlib.pyplot as plt 

import scipy.optimize as opti 

from PIL import Image
from pytesseract import image_to_string

import peakdetect as pdet

from bead_util import find_all_fnames

import cv2







def get_devnum(filename, device_prefix='dev'):
    ### Image naming convention often includes a standard device prefix
    ### like "dev" which defines a subfolder. Look for this identifier in
    ### a given filename. Function should fail if it can't find anything
    devstr = re.findall("/{:s}[0-9]+/".format(device_prefix), filename)[0]
    devnum = int( re.findall("[0-9]+", devstr)[0] )
    return devnum





class SEMImage:

    def __init__(self):
        self.path = ''

        self.full_img_arr = []
        self.img_arr = []
        self.info_bar = []
        self.scale_bar = []

        self.scale_fac = 0.0




    def load(self, filename, image_bits=16, xsize=1024, ysize=943):
        '''Load the image into the class container and separate the
           image itself, the information bar, and the scale bar.

           Default sizes are for 1024px wide images from the 
           Magellan SEM at SNSF.'''

        self.path = filename
        self.full_img_arr = np.array( Image.open(filename) )

        shape = self.full_img_arr.shape

        maxval = shape[1] * (2.0**16 - 1.0)
        inds = np.arange(shape[0])[np.sum(self.full_img_arr, axis=1) >= 0.99*maxval]

        ### Compute some cropping indices
        starty = int(inds[::-1][1])
        stopy = starty+1 + np.argmax(np.sum(self.full_img_arr[starty+1:-2,-50:], axis=1))
        startx = shape[1] - np.argmax(self.full_img_arr[starty+1,:-1][::-1]) - 2

        ### Crop that shit baby
        self.img_arr = self.full_img_arr[:starty,:]
        self.info_bar = self.full_img_arr[starty:,:]
        self.scale_bar = self.full_img_arr[starty:stopy,startx:]
        self.scale_bar = self.scale_bar[1:-1,1:-1]






    def read_scale_bar(self, plot=False, verbose=False):
        '''Try using the Tesseract OCR to automatically read the scale bar.'''
        try:
            output = image_to_string(self.scale_bar, lang='eng')
            number = re.search("[0-9]+", output)[0]
            units = re.search("[a-z]+", output)[0]
        except Exception:
            traceback.print_exc()
            number = 0
            units = 'um'

        if verbose:
            print('FOUND SCALE: ', number, units)

        if units == 'nm':
            fac = 1.0e-9
        if units == 'um':
            fac = 1.0e-6
        if units == 'mm':
            fac = 1.0e-3

        self.scale_bar_len = float(number) * fac






    def calibrate(self, scale_bar_len=0.0, verbose=True, plot=False):
        '''Using the scale bar portion of the image, find the pixel -> distance
           calibration factor and save this as a class attribute.'''

        ### Try auto-finding the scale from the bar, or query the user
        ### if the automatic procedure fails
        if not scale_bar_len:

            try:
                self.read_scale_bar(plot=plot, verbose=verbose)
                scale_bar_len = self.scale_bar_len

            except Exception:
                traceback.print_exc()
                print()
                print("COULDN'T AUTOMATICALLY DETERMINED SCALE BAR SIZE...")

                plt.imshow(self.full_img_arr, cmap='gray')
                plt.figure()
                plt.imshow(self.scale_bar, cmap='gray')
                plt.show()

                scale_bar_len = float(input('ENTER LENGTH [m]: '))

        ### Define some lengths
        full_len = self.scale_bar.shape[1]
        cut_size = int(full_len / 3.0)

        ### Crop the left and right portions of the scale bar to look for
        ### the endcaps of the bar
        first_third = self.scale_bar[:,:cut_size]  
        last_third = self.scale_bar[:,-cut_size:]

        ### Marginalize the first and last third and take the max to 
        ### to robustly find the end cap
        first_third_marginalized = np.sum(first_third, axis=0)
        first_third_ind = np.argmax(first_third_marginalized)

        last_third_marginalized = np.sum(last_third, axis=0)
        last_third_ind = full_len - np.argmax(last_third_marginalized[::-1]) - 1

        ### Compute the pixel length between these endcaps
        pixel_len = last_third_ind - first_third_ind

        ### Compute the calibration factor
        self.scale_fac = scale_bar_len / pixel_len
        if verbose:
            print('Conversion from pixels to meters', self.scale_fac)

        if plot:
            fig = plt.figure(constrained_layout=True,figsize=(7,3))
            gs = fig.add_gridspec(2, 2)
            ax00 = fig.add_subplot(gs[0,0])
            ax00.imshow(first_third, cmap='gray')
            ax00.set_title('Left endcap')
            # plt.figure()
            ax01 = fig.add_subplot(gs[0,1])
            ax01.imshow(last_third, cmap='gray')
            ax01.set_title('Right endcap')
            # plt.figure()
            ax1 = fig.add_subplot(gs[1,:])
            ax1.imshow(self.scale_bar, cmap='gray')
            ax1.axvline(first_third_ind, color='r')
            ax1.axvline(last_third_ind, color='r')
            ax1.set_title('Output of Calibration: {:d} pixels / {:0.1g} meters'\
                        .format(pixel_len, scale_bar_len))
            fig.tight_layout()
            plt.show()







    def find_vertical_edges(self, xinds=(0,-1), yinds=(0,-1), image_bits=16, \
                            edge_width=10, plot=False, verbose=False):

        ### Crop the main image according to the inputs
        cropped = self.img_arr[yinds[0]:yinds[1],xinds[0]:xinds[1]]

        ### Cast to 8-bit image values
        temp = (cropped.astype(np.uint16) * (256.0 / (2.0**image_bits)) )

        shape = temp.shape

        ### Blur the image slightly with gaussian blurring to reduce noise
        blurred = cv2.blur(temp, (3,3))

        ### Compute the horizontal gradient of each row of pixels in the cropped
        ### image and add these all up, as vertical edges usually have distinctive 
        ### intensity gradients
        deriv_sum = np.zeros(shape[1])
        for row in range(shape[0]):
            deriv_sum += np.gradient(blurred[row,:])

        ### Define some thresholds
        std = np.std(deriv_sum)
        # pos_thresh = np.max([ 2.0 * std, 0.1*np.max(deriv_sum)])
        pos_thresh = 0.15 * np.max(deriv_sum)
        # neg_thresh = np.min([-2.0 * std, 0.1*np.min(deriv_sum)])
        neg_thresh = 0.15 * np.min(deriv_sum)

        ### Find the indices above and below the defined thresholds
        inds_above = np.arange(shape[1])[deriv_sum > pos_thresh]
        inds_below = np.arange(shape[1])[deriv_sum < neg_thresh]

        pos_peaks = []
        neg_peaks = []

        ### Loop over indices above threshold and for each set of
        ### consecutive indices, compute the location of the maximum
        cinds = []
        for ind in inds_above:
            if not len(cinds):
                cinds.append(ind)
                continue
            
            if (ind - cinds[-1]) > 1 or ind == inds_above[-1]:
                max_ind = np.argmax(deriv_sum[cinds])
                pos_peaks.append([cinds[max_ind], deriv_sum[cinds[max_ind]]])
                cinds = [ind]
            else:
                cinds.append(ind)

        ### Same looping operation as above, but with indices below threshold
        cinds = []
        for ind in inds_below:
            if not len(cinds):
                cinds.append(ind)
                continue

            if (ind - cinds[-1]) > 1 or ind == inds_below[-1]:
                min_ind = np.argmin(deriv_sum[cinds])
                neg_peaks.append([cinds[min_ind], deriv_sum[cinds[min_ind]]])
                cinds = [ind]
            else:
                cinds.append(ind)

        ### Simultaneously analyze positive and negative peaks, looking for the distinctive
        ### paired peaks that arise when imaging 3-dimensional structures with scanning
        ### electron microscopy. Still accepts peaks that aren't paired, but includes a 
        ### boolean flag that is False when there is only a single positive or a single 
        ### negative peak in the gradient associated with the image
        edge_locations = []
        neg_done = []
        for pos_peak in pos_peaks:
            found_match = False
            for neg_peak_ind, neg_peak in enumerate(neg_peaks):
                if np.abs(pos_peak[0] - neg_peak[0]) <= edge_width:
                    edge_locations.append([np.average([pos_peak[0], neg_peak[0]], \
                                            weights=np.abs([pos_peak[1], neg_peak[1]])), True])
                    neg_done.append(neg_peak_ind)
                    found_match = True
                    break
            if found_match:
                continue
            edge_locations.append([pos_peak[0], False])
        for neg_peak_ind, neg_peak in enumerate(neg_peaks):
            if neg_peak_ind in neg_done:
                continue
            edge_locations.append([neg_peak[0], False])



        ### If desired, plot the gradient analysis showing all positive and negative 
        ### peaks identified, as well as the source image itself and the locations of
        ### any edges
        if plot:
            fig, axarr = plt.subplots( 2,1,figsize=(8,8), sharex=True, \
                                        gridspec_kw={'height_ratios': (3,2)} )

            axarr[0].set_title('Sum of horizontal gradients')
            axarr[0].plot(deriv_sum)
            axarr[0].set_xlim((0,shape[1]))
            for peak in pos_peaks:
                axarr[0].axvline(peak[0], ls=':', lw=3, color='r')
            for peak in neg_peaks:
                axarr[0].axvline(peak[0], ls=':', lw=3, color='b')

            axarr[1].set_title('Found edges')
            axarr[1].imshow(temp, cmap='gray')
            for edge in edge_locations:
                if edge[1]:
                    alpha = 1.0
                else:
                    alpha = 0.4
                axarr[1].axvline(edge[0], ls=':', lw=3, color='r', alpha=alpha)
            axarr[1].set_aspect('auto')

            fig.tight_layout()
            plt.show()


        for edge_ind, edge in enumerate(edge_locations):
            edge_locations[edge_ind][0] += xinds[0]

        return edge_locations




    def find_edges(self, xinds=(0,-1), yinds=(0,-1), image_bits=16):
        '''STILL DEVELOPING THIS FUNCTION. IT JUST PLOTS A BUNCH OF STUFF
           RIGHT NOW BECAUSE EDGE DETECTION IS HARD.'''
        maxval = 0.9 * (2.0**8)
        minval = 0.1 * (2.0**8)

        cropped = self.img_arr[yinds[0]:yinds[1],xinds[0]:xinds[1]]

        temp = (cropped.astype(np.uint16) * (256.0 / (2.0**image_bits)) )
        img8 = cv2.blur( temp.astype(np.uint8), (5,5) )

        # plt.imshow(img8, cmap='gray')
        #img8 = cv2.convertScaleAbs(img8, alpha=3.0, beta=-100)

        img8_eq = cv2.blur( cv2.equalizeHist(img8), (5,5))
        #img8 = cv2.blur(img8, (5,5))
        # plt.figure()
        # plt.hist(img8.flatten(), 100)
        # plt.figure()
        plt.imshow(img8, cmap='gray')
        plt.figure()
        plt.imshow(img8_eq, cmap='gray')
        # plt.show()

        plt.figure()
        plt.hist(img8_eq.flatten(), 100)
        # plt.show()

        edges = cv2.Canny(img8_eq, 125, 200)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        img8_cont = cv2.drawContours(img8_eq, contours, -1, color=150, thickness=3)

        plt.figure()
        plt.imshow(img8_cont)

        plt.figure()
        plt.imshow(edges, cmap='gray')
        plt.show()



