import numpy as np
import argparse
import cv2

import scipy.optimize as opti

import matplotlib.pyplot as plt



imgs = ['./12000x/20190129_radii-meas_1um-grating_12000x_1.TIF', \
		'./12000x/20190129_radii-meas_1um-grating_12000x_2.TIF', \
		'./12000x/20190129_radii-meas_1um-grating_12000x_3.TIF',]
scale_size = 5.0


# imgs = ['./35000x/20190129_radii-meas_1um-grating_35000x_1.TIF', \
# 		'./35000x/20190129_radii-meas_1um-grating_35000x_2.TIF', \
# 		'./35000x/20190129_radii-meas_1um-grating_35000x_3.TIF', \
# 		'./35000x/20190129_radii-meas_1um-grating_35000x_4.TIF', \
# 		'./35000x/20190129_radii-meas_1um-grating_35000x_5.TIF']
# scale_size = 2.0


savepath = './12000x_calibration.npy'


def distance(p1, p2):
	return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

def angle(p1, p2):
	return np.arctan2((p1[1]-p2[1]), (p1[0]-p2[0]))





def plot_and_calibrate_scale_bar(scale_img, scale_size=0):

	if scale_size == 0:
		plt.figure(dpi=200)
		plt.imshow(scale_img, cmap='gray')
		plt.show()
		scale_size = float(raw_input('Scale size in microns: '))
		plot_bar = True
	else:
		plot_bar = False

	scale_1d = scale[9,:]
	for pixel_ind, pixel in enumerate(scale_1d):
		if pixel > 126:
			start_ind = pixel_ind
			break
	for pixel_ind, pixel in enumerate(scale_1d[::-1]):
		if pixel > 126:
			stop_ind = len(scale_1d) - pixel_ind - 1
			break
	scale_pixels = stop_ind - start_ind + 1
	resolution = scale_size / scale_pixels

	print 'Micron/pixel: ', resolution

	if plot_bar:
		plt.figure(dpi=200)
		plt.imshow(scale, cmap='gray')
		plt.plot([start_ind, stop_ind], [9, 9], lw=2, color='r')
		plt.show()

	return [scale_size, scale_pixels, resolution]





all_dists = []
for img in imgs:
	image = cv2.imread(img)
	output = image.copy()

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	sem = gray[:484,:]
	bar = gray[484:,:]

	#scale = bar[20:,375:625]
	scale = bar[20:,450:625]

	scale_size, scale_pixels, derp_resolution = \
				plot_and_calibrate_scale_bar(scale, scale_size=scale_size)

	scale_pixels_err = 1.0


	grating_pixels = int(1.0 / derp_resolution)  # exact 1um grating
	#grating_pixels = int(10.0 / derp_resolution)  # approx 10um grating



	blur = cv2.GaussianBlur(sem,(3,3),0)

	ret, th1 = cv2.threshold(blur,0,255,\
            		cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	plt.imshow(th1, cmap='gray')
	plt.show()



	contours, hierarchy = \
			cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	img = cv2.drawContours(th1, contours, -1, 126, 3)

	#plt.imshow(img, cmap='gray')
	#plt.show()


	pts = []
	for contour in contours:
		moments = cv2.moments(contour)
		if moments['m00'] == 0:
			continue
		cx = float(moments['m10']/moments['m00'])
		cy = float(moments['m01']/moments['m00'])
		pts.append([cx, cy])
	dists = []
	for pt1 in pts:
		for pt2 in pts:
			dist = distance(pt1, pt2)
			if dist > 1.1 * grating_pixels:
				continue
			elif dist < 0.9 * grating_pixels:
				continue
			else:
				dists.append(dist)

	all_dists += dists


mean_dist = np.mean(all_dists)
std_dist = np.std(all_dists)

std_err = std_dist / np.sqrt(len(all_dists))



def gauss(x, A, mu, sigma):
	return A * np.exp( -(x - mu)**2 / (2 * sigma**2))


# for 12000x images
p0 = [300, 26, 1]
# for 35000x images
#p0 = [300, 78, 1]


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



# Compute resolution knowing 1um grating is 1.000 + 0.005 um (NIST traceable)
resolution = 1.0 / mean_dist
resolution_err = resolution * np.sqrt((std_err/mean_dist)**2 + (0.005/1.0)**2)

resolution_2 = 1.0 / mean_dist_2
resolution_err_2 = resolution_2 * np.sqrt((std_err_2/mean_dist_2)**2 + (0.005/1.0)**2)

# Compute resolution knowing 1um grating is 1.000 + 0.005 um (NIST traceable)
# resolution = 9.983 / mean_dist
# resolution_err = resolution * np.sqrt((std_err/mean_dist)**2 + (0.0189/9.983)**2)

# resolution_2 = 9.983 / mean_dist
# resolution_err_2 = resolution_2 * np.sqrt((std_err_2/mean_dist_2)**2 + (0.0189/9.983)**2)


print
print 'N  : ', len(all_dists)
print
print

print 'Raw Mean separation      : ', mean_dist
print 'Raw Std. Error on Mean   : ', std_err
print

print 'Gauss Mean separation    : ', mean_dist_2
print 'Gauss Std. Error on Mean : ', std_err_2
print

print 'Resolution [um/pixel]    : ', resolution
print 'Gauss Res. [um/pixel]    : ', resolution_2






out_arr = [mean_dist, std_err, mean_dist_2, std_err_2, \
			resolution, resolution_err, resolution_2, resolution_err_2]



np.save(savepath, out_arr)





plt.show()















