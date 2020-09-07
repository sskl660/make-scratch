import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import gridspec

config = {
    'img_src': './images/original_001.png'
}


def update(val):
    global img_gray
    global img_bin
    global sp_img
    th = int(sld_th.val)  # threshold
    bs = int((2*int(sld_bs.val))-1)  # block size
    vc = int(sld_vc.val)  # value c
    img_bin = adp_th(img_gray, th, bs, vc)
    sp_img.imshow(img_bin, cmap='gray', vmin=0, vmax=255)


def adp_th(input_img, threshold, block_size, val_c):
    tmp_img = cv2.adaptiveThreshold(input_img, threshold, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block_size, val_c)
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    # cv2.ADAPTIVE_THRESH_MEAN_C
    return tmp_img



fig = plt.figure()

img = cv2.imread(config['img_src'], cv2.IMREAD_COLOR)  # read image as BGR
h, w, c = img.shape
img_square = img[0:h, int((w-h)/2):int((w-h)/2)+h]
img_gray = cv2.cvtColor(img_square, cv2.COLOR_BGR2GRAY)  # make grayscale image for threshold operation
img_bin = adp_th(img_gray, 255, 75, 15)

gs = gridspec.GridSpec(nrows=4,  # row 몇 개
                       ncols=1,  # col 몇 개
                       height_ratios=[6, 1, 1, 1],
                       width_ratios=[1]
                      )

sp_img = plt.subplot(gs[0])
sp_img.imshow(img_bin, cmap='gray', vmin=0, vmax=255)

sp_th = plt.subplot(gs[1])
sp_bs = plt.subplot(gs[2])
sp_vc = plt.subplot(gs[3])

# set Slider object on sld_val
sld_th = Slider(sp_th, 'threshold', 0, 255, valinit=255, valfmt='%d')
sld_bs = Slider(sp_bs, 'block size (2x-1)', 1, 100, valinit=38, valfmt='%d')
# sld_bs must be odd num before threshold operation
sld_vc = Slider(sp_vc, 'value c', 1, 30, valinit=15, valfmt='%d')

# set events on each Slider object
sld_th.on_changed(update)
sld_bs.on_changed(update)
sld_vc.on_changed(update)

# plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.subplot(132), plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
# plt.subplot(411), plt.imshow(img_bin, cmap='gray', vmin=0, vmax=255)

plt.show()
