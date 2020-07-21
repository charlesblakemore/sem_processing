import numpy as np
import argparse
import cv2

import scipy.optimize as opti

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sem_util as su

from bead_util import find_all_fnames


plt.rcParams.update({'font.size': 14})



img_dir = '/Users/manifestation/Stanford/beads/photos/sem/20200624_gbeads-7_5um/'
substr = '7_5um_5000x_uc'
# substr = '7_5um_15000x_uc'

imgs, _ = find_all_fnames(img_dir, ext='.tif', substr=substr)

# gauss_kernel = 10
blur_kernel = 9

### Size of gaussian weighted block for adaptive thresholding. Must be odd integer
th_block_size = 15
# th_block_size = 51  
constant = 2


second_blur = [3, 5, 7, 9]


#calibration = np.load('./12000x_calibration.npy')
calibration = np.load('../calibrations/20200624_7_5um_calibration_5000x_uc.npy')
# calibration = np.load('../calibrations/20200624_7_5um_calibration_15000x_uc.npy')
cal_fac = calibration[6]

savepath = '../data/bead_radii_5000x.npy'
# savepath = '../data/bead_radii_15000x.npy'


plot_debug = False
plot_contour = False
plot_circles = False
plot_ellipses = False
plot_individual_ellipse = False





####################################################################################
####################################################################################
####################################################################################




def gauss(x, A, mu, sigma):
    return A * np.exp( -1.0 * (x - mu)**2 / (2 * sigma**2))




### Routines below find the outer edge of the thick contour defined by
# radius_adj = 0.5 * (gauss_kernel + th_block_size) + 5
radius_adj = th_block_size + 2





bead_circle_data = []
bead_ellipse_data = []

for filename in imgs:

    if 'bad' in filename:
        continue

    # if '_001' in filename:
    #     continue

    imgobj = su.SEMImage()
    imgobj.load(filename)

    imgobj.rough_calibrate(plot=False)


    circles = imgobj.find_circles(blur_kernel=blur_kernel, th_block_size=th_block_size, \
                                  constant=constant, min_dist=80, min_radius=120, max_radius=160, \
                                  Hough_param1=200, Hough_param2=20, radius_adj=-3, \
                                  plot=True)

    input()




    if plot_ellipses:
        fig, ax = plt.subplots(1,1)
        ax.imshow(imgobj.img_arr_8bit, cmap='gray')



    for circle in circles:
        # print(circle)

        bead_circle_data.append([circle[0], circle[1], circle[2]])

        left = int(circle[0] - 1.1*circle[2])
        right = int(circle[0] + 1.1*circle[2])
        top = int(circle[1] - 1.1*circle[2])
        bot = int(circle[1] + 1.1*circle[2])

        if left < 0:
            left = 0
        if top < 0:
            top = 0

        # mask = np.zeros_like(new_th2)
        ypixel, xpixel = new_th2.shape

        yvals = np.arange(ypixel)
        xvals = np.arange(xpixel)

        dists = np.sqrt(np.add.outer((yvals - circle[1])**2, (xvals - circle[0])**2))
        mask = dists <= circle[2] + 3

        # for yind, yval in enumerate(yvals):
        #     for xind, xval in enumerate(xvals):
        #         dist = np.sqrt( (xval - circle[0])**2 + (yval - circle[1])**2 )
        #         if dist <= circle[2] + 2:
        #             mask[yind,xind] = 1.0


        cropped = (new_th2 * mask)[top:bot,left:right]

        contours, hierarchy = cv2.findContours(cropped,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        big_ind = 0
        big_area = 0
        for ind, contour in enumerate(contours):

          moments = cv2.moments(contour)
          area = moments['m00']
          if area > big_area:
              big_area = area
              big_ind = ind

        ellipse = cv2.fitEllipse(contours[big_ind])
        bead_ellipse_data.append([ellipse[0][0]+left, ellipse[0][1]+top, ellipse[1][0]-6, ellipse[1][1]-6, ellipse[2]])
        # ellipse = cv2.fitEllipse(cropped)

        if plot_ellipses:
            plot_ellipse = patches.Ellipse((bead_ellipse_data[-1][0], bead_ellipse_data[-1][1]), \
                                           bead_ellipse_data[-1][2], bead_ellipse_data[-1][3], \
                                           bead_ellipse_data[-1][4]) 
            plot_ellipse.set_facecolor('none')
            plot_ellipse.set_edgecolor('r')
            plot_ellipse.set_linewidth(3)
            plot_ellipse.set_linestyle('--')
            ax.add_artist(plot_ellipse)



        if plot_individual_ellipse:
            fig2 = plt.figure()
            ax2 = fig.add_subplot(111, aspect='equal')
            ax2.imshow(cropped, cmap='gray')
            plot_ellipse2 = patches.Ellipse(ellipse[0], ellipse[1][0], \
                                           ellipse[1][1], ellipse[2]) 
            plot_ellipse2.set_facecolor('none')
            plot_ellipse2.set_edgecolor('r')
            plot_ellipse2.set_linewidth(3)
            plot_ellipse2.set_linestyle('--')
            ax2.add_artist(plot_ellipse2)

            # plt.imshow(cropped, cmap='gray')
            fig2.show()

            input()

    if plot_ellipses:
        fig.show()
        input()


bead_circle_data = np.array(bead_circle_data)
bead_ellipse_data = np.array(bead_ellipse_data)

# print(bead_circle_data)
# print(bead_ellipse_data)

# plt.figure()
# plt.hist(cal_fac*bead_circle_data[:,2], 20)

# plt.figure()
plot_x = np.linspace(2, 5, 100)

rfig, rax = plt.subplots(3,1,sharex=True,figsize=(6,6),dpi=150)
chist, bin_edge, _ = rax[0].hist(cal_fac*bead_circle_data[:,2], 30, range=(2, 5), alpha=1.0)
rax[0].set_title('Radius from HoughCircles')
bins = bin_edge[:-1] - 0.5*(bin_edge[0] - bin_edge[1])
popt, pcov = opti.curve_fit(gauss, bins, chist, p0=[np.max(chist), bins[np.argmax(chist)], 0.1])
rax[0].plot(plot_x, gauss(plot_x, *popt), lw=3, ls='--', color='r', \
            label='$r = {:0.2f} \\pm {:0.2f}~\\mu m$'.format(popt[1], popt[2]))
rax[0].legend(loc='upper left')

ehist1, _, _ = rax[1].hist(cal_fac*bead_ellipse_data[:,2]/2.0, 30, range=(2, 5), alpha=1.0)
rax[1].set_title('Ellipse Axis 1')
popt, pcov = opti.curve_fit(gauss, bins, ehist1, p0=[np.max(ehist1), bins[np.argmax(ehist1)], 0.1])
rax[1].plot(plot_x, gauss(plot_x, *popt), lw=3, ls='--', color='r', \
            label='$r = {:0.2f} \\pm {:0.2f}~\\mu m$'.format(popt[1], popt[2]))
rax[1].legend(loc='upper left')

ehist2, _, _ = rax[2].hist(cal_fac*bead_ellipse_data[:,3]/2.0, 30, range=(2, 5), alpha=1.0)
rax[2].set_title('Ellipse Axis 2')
popt, pcov = opti.curve_fit(gauss, bins, ehist2, p0=[np.max(ehist2), bins[np.argmax(ehist2)], 0.1])
rax[2].plot(plot_x, gauss(plot_x, *popt), lw=3, ls='--', color='r', \
            label='$r = {:0.2f} \\pm {:0.2f}~\\mu m $'.format(popt[1], popt[2]))
rax[2].legend(loc='upper left')
rax[2].set_xlabel('Calibrated Radius [$\\mu$m]')

rfig.tight_layout()





efig = plt.figure()
eax = efig.add_subplot(111, polar=True)
hist, bin_edge = np.histogram(bead_ellipse_data[:,4], bins=20, range=(0,180))
width = bin_edge[1] - bin_edge[0]
bins = bin_edge[:-1] + 0.5*width

fit_bins = (bins + 90.0) % 180.0

popt, pcov = opti.curve_fit(gauss, fit_bins, hist, p0=[np.max(hist), fit_bins[np.argmax(hist)], 5])

plot_x = np.linspace(0,180,100)
eax.bar(fit_bins*(np.pi/180.0), hist, width=width*(np.pi/180.0), bottom=30)
eax.plot(plot_x*(np.pi/180.0), gauss(plot_x, *popt)+30.0, lw=3, ls='--', color='r', \
            label='$\\epsilon = {:0.2f} \\pm {:0.2f} $'.format(popt[1], popt[2]))
eax.set_thetamin(0)
eax.set_thetamax(180)
# eax.set_theta_offset(np.pi)
eax.set_title('Ellipse Orientation Relative to Y-Axis')
eax.legend(loc='upper right')
efig.tight_layout()





plt.figure()
plt.title('Centers of Circle/Ellipse')
plt.scatter(bead_circle_data[:,0], bead_circle_data[:,1], label='HoughCircles', marker='X', s=25)
plt.scatter(bead_ellipse_data[:,0], bead_ellipse_data[:,1], label='Ellipse Fit', marker='P', s=25)
plt.xlabel('X-Pixel')
plt.ylabel('Y-Pixel')
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()

