import numpy as np
import argparse
import cv2

import scipy.optimize as opti

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sem_util as su

from bead_util import find_all_fnames






img_dir = '/Users/manifestation/Stanford/beads/photos/sem/20200624_gbeads-7_5um/'
substr = '7_5um_5000x_uc'
# substr = '7_5um_15000x_uc'

imgs, _ = find_all_fnames(img_dir, ext='.tif', substr=substr)

# gauss_kernel = 10
gauss_kernel = 15

### Size of gaussian weighted block for adaptive thresholding. Must be odd integer
th_block_size = 31
# th_block_size = 51  
constant = 2


#calibration = np.load('./12000x_calibration.npy')
calibration = np.load('../calibrations/20200624_7_5um_calibration_5000x_uc.npy')
# calibration = np.load('../calibrations/20200624_7_5um_calibration_15000x_uc.npy')
cal_fac = calibration[6]

savepath = '../data/bead_radii_5000x.npy'
# savepath = '../data/bead_radii_15000x.npy'


plot_debug = False
plot_contour = True





####################################################################################
####################################################################################
####################################################################################

### Routines below find the outer edge of the thick contour defined by
# radius_adj = 0.5 * (gauss_kernel + th_block_size) + 5
radius_adj = th_block_size + 2





bead_radii = []
for filename in imgs:

    if '_001' in filename:
        continue

    imgobj = su.SEMImage()
    imgobj.load(filename)

    imgobj.rough_calibrate(plot=False)

    temp = imgobj.img_arr * (256.0 / (2.0**imgobj.bit_depth))
    blur = cv2.blur(temp.astype(np.uint8),(gauss_kernel,gauss_kernel))

    ret, th1 = cv2.threshold(blur,0,255,\
                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,th_block_size,constant)

    th2_copy = np.copy(th2)

    # if plot_debug:
    plt.imshow(th1, cmap='gray', alpha=1.0)

    plt.figure()
    plt.imshow(th2, cmap='gray', alpha=1.0)

    plt.show()
    input()


    # circles = cv2.HoughCircles(temp.astype(np.uint8), cv2.HOUGH_GRADIENT, 1, 150)
    # circles_blur = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 150)
    circles_blur_binary = cv2.HoughCircles(th2, cv2.HOUGH_GRADIENT, 1, 150, \
                                           param1=100, param2=50, \
                                           minRadius=80, maxRadius=200)

    # print(circles)
    # if circles is not None:
    #     fig, ax = plt.subplots(1,1)
    #     ax.imshow(temp.astype(np.uint8), cmap='gray', zorder=1)
    #     for circle in circles[0]:
    #         ax.scatter([circle[0]], [circle[1]], color='g', marker='X', s=25, zorder=2)
    #         plot_circle = plt.Circle([circle[0], circle[1]], circle[2], \
    #                                  color='r', fill=False, zorder=3)
    #         ax.add_artist(plot_circle)
    #     plt.show()
    #     input()
    # else:
    #     print('NO CIRCLES FOUND IN RAW GRAY IMAGE')
    #     input()


    # if circles_blur is not None:
    #     fig, ax = plt.subplots(1,1)
    #     ax.imshow(blur, cmap='gray', zorder=1)
    #     for circle in circles_blur[0]:
    #         ax.scatter([circle[0]], [circle[1]], color='g', marker='X', s=25, zorder=2)
    #         plot_circle = plt.Circle([circle[0], circle[1]], circle[2], \
    #                                  color='r', fill=False, zorder=3)
    #         ax.add_artist(plot_circle)
    #     plt.show()
    #     input()
    # else:
    #     print('NO CIRCLES FOUND IN BLURRED GRAY IMAGE')
    #     input()


    if circles_blur_binary is not None:
        fig, ax = plt.subplots(1,1)
        ax.imshow(th2, cmap='gray', zorder=1)
        for circle in circles_blur_binary[0]:
            ax.scatter([circle[0]], [circle[1]], color='g', marker='X', s=25, zorder=2)
            plot_circle = plt.Circle([circle[0], circle[1]], circle[2], \
                                     color='r', fill=False, zorder=3)
            ax.add_artist(plot_circle)
        plt.show()
        input()
    else:
        print('NO CIRCLES FOUND IN THRESHOLD IMAGE')
        input()


    #ax.imshow(bead, cmap='gray', alpha=0.5

    contours, hierarchy = \
            cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    big_ind = 0
    big_area = 0
    for ind, contour in enumerate(contours):

      moments = cv2.moments(contour)
      area = moments['m00']
      if area > big_area:
          big_area = area
          big_ind = ind

    derp = cv2.drawContours(th2_copy, contours[big_ind], -1, 126, 3)
    if plot_contour:
        # plt.imshow(th2, cmap='gray', alpha=0.5)
        plt.imshow(derp, cmap='gray')
        plt.show()

        input()


    contour_ind = big_ind

    center, radius = cv2.minEnclosingCircle(contours[contour_ind])

    ellipse = cv2.fitEllipse(contours[contour_ind])

    new_ellipse = ((ellipse[0][0], ellipse[0][1]), \
                   (ellipse[1][0]-radius_adj, ellipse[1][1]-radius_adj), \
                   ellipse[2])

    pixel_diam = np.mean(new_ellipse[1])

    print(0.5*pixel_diam)

    # Calibrate and subtract off metalized coating
    comp_diam = (pixel_diam * calibration[6]) - (0.1)
    comp_radius = 0.5 * comp_diam

    comp_radius_err = np.sqrt( comp_radius**2 * ((calibration[7]/calibration[6])**2 + \
                                (1.0/pixel_diam)**2) + 0.050**2 )

    print('Found radius: ', comp_radius)

    bead_radii.append([comp_radius, comp_radius_err])


    if plot_debug:

        new_img = cv2.ellipse(th2, new_ellipse, 126, 3)

        plt.imshow(new_img, cmap='gray')
        plt.title('ELLIPSE FIT')
        plt.show()

        input()


    if plot_debug:
        figure = plt.figure()
        ax = plt.subplot(111)
        ax.imshow(th2, cmap='gray')
        new_circle = plt.Circle(center, radius-1, color='r', fill=False)
        ax.add_artist(new_circle)
        ax.set_title('CIRCLE FIT')


    center_int = (np.uint8(center[0]), np.uint8(center[1]))
    radius_int = np.uint8(radius)


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.imshow(temp, cmap='gray')
    plot_ellipse = patches.Ellipse(new_ellipse[0], new_ellipse[1][0], \
                                  new_ellipse[1][1], new_ellipse[2]) 
    plot_ellipse.set_facecolor('none')
    plot_ellipse.set_edgecolor('r')
    plot_ellipse.set_linewidth(3)
    plot_ellipse.set_linestyle('--')
    ax2.add_artist(plot_ellipse)

    ax2.get_yaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    ax2.set_title('ELLIPSE FIT')

    #new_img = cv2.circle(bead, center, radius, 126, 1)
    #plt.imshow(new_img, cmap='gray')
    plt.show()

    input()

print(bead_radii)
#np.save(savepath, bead_radii)

