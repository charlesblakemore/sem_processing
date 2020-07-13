import numpy as np
import argparse
import cv2

import scipy.optimize as opti

import matplotlib.pyplot as plt

from bead_util import find_all_fnames
import sem_util as su


gauss_kernel = 10


img_dir = '/Users/manifestation/Stanford/beads/photos/sem/20200624_gbeads-7_5um/'
substr = '7_5um_calibration_15000x_uc'

savepath = '../calibrations/20200624_{:s}.npy'.format(substr)

plot_threshold = False
plot_contours = True

average_concentric_contours = True

imgs, _ = find_all_fnames(img_dir, ext='.tif', substr=substr)







def distance(p1, p2):
    return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

def angle(p1, p2):
    return np.arctan2((p1[1]-p2[1]), (p1[0]-p2[0]))

def gauss(x, A, mu, sigma):
    return A * np.exp( -(x - mu)**2 / (2 * sigma**2))






all_dists = []
for filename in imgs:

    imgobj = su.SEMImage()
    imgobj.load(filename)

    imgobj.rough_calibrate(plot=False)

    scale_pixels_err = 1.0


    grating_pixels = 1.0e-6 / imgobj.scale_fac  # exact 1um grating
    #grating_pixels = 10.0 / derp_resolution  # approx 10um grating
    print(grating_pixels)


    temp = imgobj.img_arr * (256.0 / (2.0**imgobj.bit_depth))
    blur = cv2.blur(temp.astype(np.uint8),(gauss_kernel,gauss_kernel))

    ret, th1 = cv2.threshold(blur,0,255,\
                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    if plot_threshold:
        plt.figure()
        plt.imshow(th1, cmap='gray')
        plt.show()
    
        input()


    contours, hierarchy = \
            cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.drawContours(th1, contours, -1, 126, 1)

    pts = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            continue
        cx = float(moments['m10']/moments['m00'])
        cy = float(moments['m01']/moments['m00'])
        pts.append([cx, cy])

    npts = len(pts)
    pts = np.array(pts)



    if average_concentric_contours:
        centers = []
        for i, pt1 in enumerate(pts):
            for j, pt2 in enumerate(pts):
                if i == j:
                    continue
                dist = distance(pt1, pt2)
                if dist < 0.5 * grating_pixels:
                    centers.append( np.mean( np.array([pt1, pt2]), axis=0 ) )
        centers = np.array(centers)

    else:
        centers = np.copy(pts)


    npts = len(centers)

    if plot_contours:
        plt.figure()
        plt.imshow(img, cmap='gray', zorder=1)
        plt.scatter(centers[:,0], centers[:,1], marker='X', color='r', s=25, zorder=2)
        plt.show()

        input()


    dists = []
    dist_arr = np.zeros((npts, npts))
    for i, pt1 in enumerate(centers):
        for j, pt2 in enumerate(centers):
            dist = distance(pt1, pt2)
            dist_arr[i,j] = dist

            if dist < 0.85 * grating_pixels:
                continue
            elif dist < 1.15 * grating_pixels:
                dists.append(dist)
            elif dist < 1.6 * grating_pixels:
                dists.append(dist / np.sqrt(2))
            else:
                continue

    # plt.figure()
    # plt.hist(dist_arr.flatten(), 1000)

    # plt.figure()
    # plt.hist(dists, 20)

    # plt.show()


    all_dists += dists


mean_dist = np.mean(all_dists)
std_dist = np.std(all_dists)

std_err = std_dist / np.sqrt(len(all_dists))





p0 = [np.max(all_dists), mean_dist, std_dist]


plt.figure()
vals, bin_edge, patches = plt.hist(all_dists, bins=50)
bin_loc = bin_edge[:-1] + 0.5 * (bin_edge[1] - bin_edge[0])

plt.axvline(mean_dist, color='r')

popt, pcov = opti.curve_fit(gauss, bin_loc, vals, p0=p0)


plot_bins = np.linspace(bin_edge[0], bin_edge[-1], 200)
plot_vals = gauss(plot_bins, *popt)

plt.plot(plot_bins, plot_vals)


mean_dist_2 = popt[1]
std_err_2 = popt[2] / np.sqrt(len(all_dists))



# Compute resolution knowing 1um grating is 1.000 +- 0.005 um (NIST traceable)
resolution = 1.0 / mean_dist
resolution_err = resolution * np.sqrt((std_err/mean_dist)**2 + (0.005/1.0)**2)

resolution_2 = 1.0 / mean_dist_2
resolution_err_2 = resolution_2 * np.sqrt((std_err_2/mean_dist_2)**2 + (0.005/1.0)**2)

# Compute resolution knowing 1um grating is 9.983 +- 0.0189 um (NIST traceable)
# resolution = 9.983 / mean_dist
# resolution_err = resolution * np.sqrt((std_err/mean_dist)**2 + (0.0189/9.983)**2)

# resolution_2 = 9.983 / mean_dist
# resolution_err_2 = resolution_2 * np.sqrt((std_err_2/mean_dist_2)**2 + (0.0189/9.983)**2)


print()
print('N  : ', len(all_dists))
print()
print()

print('Raw Mean separation      : ', mean_dist)
print('Raw Std. Error on Mean   : ', std_err)
print()

print('Gauss Mean separation    : ', mean_dist_2)
print('Gauss Std. Error on Mean : ', std_err_2)
print()

print('Resolution [um/pixel]    : ', resolution)
print('Gauss Res. [um/pixel]    : ', resolution_2)






out_arr = [mean_dist, std_err, mean_dist_2, std_err_2, \
            resolution, resolution_err, resolution_2, resolution_err_2]



np.save(savepath, out_arr)





plt.show()















