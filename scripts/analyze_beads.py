import numpy as np
import argparse
import cv2

import scipy.optimize as opti

import matplotlib.pyplot as plt
import matplotlib.patches as patches



# imgs = ['./12000x/20190129_radii-meas_bead1_12000x_2.TIF', \
#         './12000x/20190129_radii-meas_bead2_12000x.TIF', \
#         './12000x/20190129_radii-meas_bead3_12000x.TIF',]

imgs = ['./35000x/20190129_radii-meas_bead1_35000x.TIF', \
      './35000x/20190129_radii-meas_bead2_35000x.TIF', \
      './35000x/20190129_radii-meas_bead3_35000x.TIF',]


#calibration = np.load('./12000x_calibration.npy')
calibration = np.load('./35000x_calibration.npy')


#savepath = './bead_radii_12000x.npy'
savepath = './bead_radii_35000x.npy'


plot_debug = False


bead_radii = []
for img in imgs:
    image = cv2.imread(img)
    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sem = gray[:484,:]
    bar = gray[484:,:]

    #scale = bar[20:,375:625]
    scale = bar[20:,450:625]

    blur = cv2.GaussianBlur(sem,(3,3),0)


    ret, th1 = cv2.threshold(blur,0,255,\
                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if plot_debug:
        plt.imshow(th1, cmap='gray', alpha=0.4)
        plt.show()

    # # # for 12000x
    # if 'bead1' in img:
    #     bead = th1[150:310,240:400]
    #     contour_ind = 0
    # if 'bead2' in img:
    #     bead = th1[155:315,250:410]
    #     contour_ind = 1
    # if 'bead3' in img:
    #     bead = th1[150:310,255:415]
    #     contour_ind = 0

    # for 35000x
    if 'bead1' in img:
      bead = th1[0:430,140:550]
      real_bead = sem[0:430,140:550]
    if 'bead2' in img:
      bead = th1[0:475,100:550]
      real_bead = sem[0:475,100:550]
    if 'bead3' in img:
      bead = th1[0:430,100:550]
      real_bead = sem[0:430,100:550]

    if plot_debug:
        plt.imshow(bead, cmap='gray')
        plt.show()



    #ax.imshow(bead, cmap='gray', alpha=0.5

    contours, hierarchy = \
            cv2.findContours(bead,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #derp = cv2.drawContours(bead, contours[-1], -1, 126, 1)
    #plt.imshow(derp, cmap='gray')
    #print contour_ind
    #plt.show()

    #derp = cv2.drawContours(bead, contours, -1, 126, 1)
    #plt.imshow(derp, cmap='gray')


    big_ind = 0
    big_area = 0
    for ind, contour in enumerate(contours):
      derp = cv2.drawContours(bead, contour, -1, 126, 1)
      #plt.imshow(derp, cmap='gray')
      #print contour_ind
      #plt.show()

      moments = cv2.moments(contour)
      area = moments['m00']
      if area > big_area:
          #print ind
          big_area = area
          big_ind = ind

    #print 'big: ', big_ind
    derp = cv2.drawContours(bead, contours[big_ind], -1, 126, 1)
    if plot_debug:
        plt.imshow(derp, cmap='gray')
        plt.show()


    contour_ind = big_ind

    center, radius = cv2.minEnclosingCircle(contours[contour_ind])

    ellipse = cv2.fitEllipse(contours[contour_ind])

    new_ellipse = ((ellipse[0][0], ellipse[0][1]), \
                   (ellipse[1][0]+2, ellipse[1][1]+2), \
                   ellipse[2])

    pixel_diam = np.mean(new_ellipse[1])

    print 0.5*pixel_diam

    # Calibrate and subtract off metalized coating
    comp_diam = (pixel_diam * calibration[6]) - (0.1)
    comp_radius = 0.5 * comp_diam

    comp_radius_err = np.sqrt( comp_radius**2 * ((calibration[7]/calibration[6])**2 + \
                                (1.0/pixel_diam)**2) + 0.050**2 )

    print 'Found radius: ', comp_radius


    bead_radii.append([comp_radius, comp_radius_err])



    new_img = cv2.ellipse(bead, new_ellipse, 126, 1)


    if plot_debug:
        plt.imshow(new_img, cmap='gray')
        plt.show()

    #cv2.imshow('image', new_img)


    if plot_debug:
        figure = plt.figure()
        ax = plt.subplot(111)
        ax.imshow(bead, cmap='gray')
        new_circle = plt.Circle(center, radius-1, color='r', fill=False)
        ax.add_artist(new_circle)


    center_int = (np.uint8(center[0]), np.uint8(center[1]))
    radius_int = np.uint8(radius)


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.imshow(real_bead, cmap='gray')
    plot_ellipse = patches.Ellipse(new_ellipse[0], new_ellipse[1][0], \
                                  new_ellipse[1][1], new_ellipse[2]) 
    plot_ellipse.set_facecolor('none')
    plot_ellipse.set_edgecolor('r')
    plot_ellipse.set_linewidth(3)
    plot_ellipse.set_linestyle('--')
    ax2.add_artist(plot_ellipse)

    ax2.get_yaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])

    #new_img = cv2.circle(bead, center, radius, 126, 1)
    #plt.imshow(new_img, cmap='gray')
    plt.show()

print bead_radii
#np.save(savepath, bead_radii)

