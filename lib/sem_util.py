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
    try:
        devstr = re.findall("/{:s}[0-9]+/".format(device_prefix), filename)[0]
        devnum = int( re.findall("[0-9]+", devstr)[0] )
        return devnum
    except Exception:
        return None





class SEMImage:

    def __init__(self):
        self.path = ''

        self.full_img_arr = []
        self.img_arr = []
        self.info_bar = []
        self.scale_bar = []

        self.scale_fac = 0.0




    def load(self, filename, bit_depth=16.0):
        '''Load the image into the class container and separate the
           image itself, the information bar, and the scale bar.

           Default sizes are for 1024px wide images from the 
           Magellan SEM at SNSF.'''

        self.path = filename
        self.full_img_arr = np.array( Image.open(filename) )
        self.bit_depth = bit_depth

        self.shape = self.full_img_arr.shape

        maxval = self.shape[1] * (2.0**bit_depth - 1.0)
        inds = np.arange(self.shape[0])[np.sum(self.full_img_arr, axis=1) >= 0.99*maxval]

        ### Compute some cropping indices by finding the white borders around the
        ### information. This works for images saved by FEI SEMs, untested on others
        ###    (LOTS OF MAGIC NUMBERS HERE, NEED TO MAKE MORE TRANSPARENT/DYNAMIC)
        starty = int(inds[::-1][1])
        stopy = starty+1 + np.argmax(np.sum(self.full_img_arr[starty+1:-2,-50:], axis=1))
        startx = self.shape[1] - np.argmax(self.full_img_arr[starty+1,:-1][::-1]) - 2

        ### Crop that shit baby
        self.img_arr = self.full_img_arr[:starty,:]
        self.info_bar = self.full_img_arr[starty:,:]
        self.scale_bar = self.full_img_arr[starty:stopy,startx:]
        self.scale_bar = self.scale_bar[1:-1,1:-1]




    def read_scale_bar(self, plot=False, verbose=False):
        '''Try using the Tesseract OCR to automatically read the scale bar.'''
        try:
            output = image_to_string(self.scale_bar[:,100:300], lang='eng')
            number = re.search("[0-9]+", output)[0]
            units = re.search("[a-z]+", output)[0]
        except Exception:
            traceback.print_exc()
            number = 0
            units = 'um'

        if verbose:
            print('FOUND SCALE: ', number, units)

        ### Figure out the units
        if units == 'nm':
            fac = 1.0e-9
        if units == 'um':
            fac = 1.0e-6
        if units == 'mm':
            fac = 1.0e-3

        self.scale_bar_len = float(number) * fac






    def rough_calibrate(self, scale_bar_len=0.0, verbose=True, plot=False):
        '''Using the scale bar portion of the image, find the pixel -> distance
           calibration factor and save this as a class attribute.'''

        ### Try auto-finding the scale from the bar, or query the user
        ### if the automatic procedure fails
        if not scale_bar_len:

            self.read_scale_bar(plot=plot, verbose=verbose)
            scale_bar_len = self.scale_bar_len

            if not scale_bar_len:
                print()
                print("COULDN'T AUTOMATICALLY DETERMINE SCALE BAR SIZE...")

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

        ### Plot the results of this procedure for debugging
        if plot:
            fig = plt.figure(constrained_layout=True,figsize=(7,3))
            gs = fig.add_gridspec(2, 2)
            ax00 = fig.add_subplot(gs[0,0])
            ax00.imshow(first_third, cmap='gray')
            ax00.set_title('Left endcap')

            ax01 = fig.add_subplot(gs[0,1])
            ax01.imshow(last_third, cmap='gray')
            ax01.set_title('Right endcap')

            ax1 = fig.add_subplot(gs[1,:])
            ax1.imshow(self.scale_bar, cmap='gray')
            ax1.axvline(first_third_ind, color='r')
            ax1.axvline(last_third_ind, color='r')
            ax1.set_title('Output of Calibration: {:d} pixels / {:0.1g} meters'\
                        .format(pixel_len, scale_bar_len))
            fig.tight_layout()
            plt.show()








    def find_edges(self, xinds=None, yinds=None, edge_width=10, plot=False, verbose=False, \
                   vertical=False, horizontal=False, blur_kernel=3, \
                   thresh_fac=0.0, pos_thresh_fac=0.25, neg_thresh_fac=0.25, \
                   image_edge_exclusion=10):
        '''Function to find vertical or horizontal edges (sharp intensity gradients)
           based on marginalizing the image. Not a super smart function, and fails
           when the images are noisy (i.e. when noise ~ edge contrast). Pre-blurring
           can help'''

        if (not horizontal) and (not vertical):
            vertical = True

        if xinds is None:
            xinds = (0, self.shape[1])
        if yinds is None:
            yinds = (0, self.shape[0])

        blur_kernel = int(blur_kernel)

        ### Cast to 8-bit image values
        self._make_8bit()
        temp = np.copy(self.img_arr_8bit[yinds[0]:yinds[1],xinds[0]:xinds[1]])

        shape = temp.shape

        ### Blur the image slightly with gaussian blurring to reduce noise
        blurred = cv2.GaussianBlur(temp, (blur_kernel,blur_kernel), 0)

        ### Compute the horizontal or vertical gradient of each row of pixels in the 
        ### cropped image and add these all up, as vertical edges usually have 
        ### distinctive intensity gradients
        if vertical:
            deriv_sum = np.zeros(shape[1])
            for row in range(shape[0]):
                deriv_sum += np.gradient(blurred[row,:])
        else:
            deriv_sum = np.zeros(shape[0])
            for col in range(shape[1]):
                deriv_sum += np.gradient(blurred[:,col])

        deriv_sum -= np.mean(deriv_sum)

        ### Define some thresholds
        if thresh_fac:
            pos_thresh_fac = np.copy(thresh_fac)
            neg_thresh_fac = np.copy(thresh_fac)
        pos_thresh = pos_thresh_fac * np.max(deriv_sum)
        neg_thresh = neg_thresh_fac * np.min(deriv_sum)

        ### Find the indices above and below the defined thresholds
        if vertical:
            all_inds = np.arange(shape[1])
        else:
            all_inds = np.arange(shape[0])
        inds_above = all_inds[deriv_sum > pos_thresh]
        inds_below = all_inds[deriv_sum < neg_thresh]

        pos_peaks = []
        neg_peaks = []

        pos_peaks = self._parse_threshold_indices(inds_above, deriv_sum, image_edge_exclusion)
        neg_peaks = self._parse_threshold_indices(inds_below, deriv_sum, image_edge_exclusion)



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
                    ### Avoid double counting closely spaced edges
                    if neg_peak_ind in neg_done:
                        continue
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
            if vertical:
                fig, axarr = plt.subplots( 2,1,figsize=(8,8), sharex=True, \
                                            gridspec_kw={'height_ratios': (3,1)} )
            else:
                fig, axarr = plt.subplots( 1,2,figsize=(10,8), sharey=True, \
                                            gridspec_kw={'width_ratios': (3,1)} )


            if vertical:
                axarr[1].plot(deriv_sum)
                axarr[1].set_title('Sum of horizontal gradients')
                axarr[1].set_xlim((0,shape[1]))
                for peak in pos_peaks:
                    axarr[1].axvline(peak[0], ls=':', lw=3, color='r')
                for peak in neg_peaks:
                    axarr[1].axvline(peak[0], ls=':', lw=3, color='b')
            else:
                axarr[1].plot(deriv_sum, range(shape[0]))
                axarr[1].set_title('Sum of vertical gradients')
                axarr[1].set_ylim((0,shape[0]))
                for peak in pos_peaks:
                    axarr[1].axhline(peak[0], ls=':', lw=3, color='r')
                for peak in neg_peaks:
                    axarr[1].axhline(peak[0], ls=':', lw=3, color='b')

            axarr[0].set_title('Found edges')
            axarr[0].imshow(temp, cmap='gray')
            for edge in edge_locations:
                if edge[1]:
                    alpha = 1.0
                else:
                    alpha = 0.4

                if vertical:
                    axarr[0].axvline(edge[0], ls=':', lw=3, color='r', alpha=alpha)
                else:
                    axarr[0].axhline(edge[0], ls=':', lw=3, color='r', alpha=alpha)

            axarr[0].set_aspect('auto')

            fig.tight_layout()
            plt.show()

        ### Adjust for the cropping so the returned edges are relative to the pixel 
        ### space of the uncropped original image
        for edge_ind, edge in enumerate(edge_locations):
            if vertical:
                offset = xinds[0]
            if horizontal:
                offset = yinds[0]

            edge_locations[edge_ind][0] += offset

        return edge_locations






    def find_circles(self, xinds=None, yinds=None, \
                     blur_kernel=3, th_block_size=15, constant=2, global_thresh=220, \
                     accumulator=1, min_dist=0, min_radius=0, max_radius=0, \
                     Hough_param1=200, Hough_param2=20, radius_adj=-3, \
                     plot=False, verbose=False):
        '''Function to find circles in an image. After some pre-processing, it applies the
           Hough transform to find circles. Because of the blurring and thresholding, the
           radii from the HoughCircles() function need to be adjusted slightly. This is
           NOT done well at this point, so it's a user-supplied argument.'''

        if min_dist == 0:
            min_dist = 0.01 * np.min(self.shape)

        if max_radius == 0:
            max_radius = np.max(self.shape)

        if xinds is None:
            xinds = (0, self.shape[1])
        if yinds is None:
            yinds = (0, self.shape[0])


        blur_kernel = int(blur_kernel)

        ### Cast to 8-bit image values
        self._make_8bit()
        temp = np.copy(self.img_arr_8bit[yinds[0]:yinds[1],xinds[0]:xinds[1]])

        ### Blur the image to reduce noise and improve the circle detection
        blur = cv2.GaussianBlur(temp, (blur_kernel, blur_kernel), 0)

        ### For the first step, apply an adaptive gaussian threshold to the blurred 
        ### image, which should help to keep this method robust against non-uniform
        ### illumination of features. Produces a binary image
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                   cv2.THRESH_BINARY_INV, th_block_size, constant)

        th_copy = np.copy(th)

        ### Re-blur the threshold image so only thick contrast gradients survive
        new_blur = cv2.GaussianBlur(th, (5, 5), 0)

        ### Apply a strict global threshold to the re-blurred image to retain the thick
        ### gradients that weren't blurred
        _, new_th = cv2.threshold(new_blur, global_thresh, 255, cv2.THRESH_BINARY)

        ### Apply the Hough Transform to the second threshold image
        circles = cv2.HoughCircles(new_th, cv2.HOUGH_GRADIENT, accumulator, \
                                   min_dist, param1=Hough_param1, param2=Hough_param2, \
                                   minRadius=min_radius, maxRadius=max_radius)


        if plot:
            if circles is not None:
                fig, ax = plt.subplots(1,1)
                fig2, ax2 = plt.subplots(1,1)
                ax.imshow(new_th, cmap='gray', zorder=1)
                ax2.imshow(temp, cmap='gray', zorder=1)
                for circle_ind, circle in enumerate(circles[0]):
                    ax.scatter([circle[0]], [circle[1]], color='g', marker='X', s=25, zorder=2)
                    ax2.scatter([circle[0]], [circle[1]], color='g', marker='X', s=25, zorder=2)
                    plot_circle = plt.Circle([circle[0], circle[1]], circle[2], \
                                             color='r', fill=False, zorder=3)
                    plot_circle2 = plt.Circle([circle[0], circle[1]], circle[2]+radius_adj, \
                                             color='r', fill=False, zorder=3)
                    ax.add_artist(plot_circle)
                    ax2.add_artist(plot_circle2)

                    circles[0][circle_ind][2] += radius_adj

                plt.show()
                input()
            else:
                fig, ax = plt.subplots(1,1)
                ax.imshow(new_th, cmap='gray', zorder=1)
                print('NO CIRCLES FOUND IN THRESHOLD IMAGE')

                plt.show()
                input()

        return circles[0]







    ### Internal methods so function definitions aren't insane. Probably should try to put more
    ### stuff down here so I look like a legit programmer


    def _make_8bit(self):
        '''Convert images arrays to 8bit values for use with openCV.'''

        self.full_img_arr_8bit = (self.full_img_arr * (256.0 / (2.0**self.bit_depth))).astype(np.uint8)
        self.img_arr_8bit = (self.img_arr * (256.0 / (2.0**self.bit_depth))).astype(np.uint8)



    def _parse_threshold_indices(self, indices, deriv_sum, exclusion):
        '''Process a list of indices giving the locations of "deriv_sum" above/below 
           a threshold value, finding and averaging the locations and values of 
           consecutive indices. Assumes indices are monotonic'''

        peaks = []

        cinds = []
        for ind in indices:

            ### Determine if the current above/below-threshold index is in 
            ### the edge exclusion zone. Because of how this loop works
            ### we only continue on the first bad condition
            bad_cond_1 = (ind < exclusion)
            bad_cond_2 = (ind > (len(deriv_sum) - exclusion))
            if bad_cond_1:
                continue

            ### Check the end condition, which is necessary to handle peaks that only 
            ### span a single index, but exist at the end of the list
            end_cond = ind == indices[-1]

            if not len(cinds):
                cinds.append(ind)
                if not end_cond:
                    continue

            if (ind - cinds[-1]) > 1 or end_cond:
                ### Append the mean location and mean value of all the most
                ### most recent set of consecutive indices (a peak)
                loc = np.mean(cinds)
                val = np.mean(deriv_sum[cinds])
                peaks.append([loc, val])

                ### Start the list over with the new, non-consecutive index.
                ### If end_cond == True, this behavior doesn't matter
                cinds = [ind]
            else:
                cinds.append(ind)


            ### Since indices are monotonic, if we reached the edge of the second 
            ### exclusion zone, there can be no more valid indices and we can break
            if bad_cond_2:
                break

        return peaks




