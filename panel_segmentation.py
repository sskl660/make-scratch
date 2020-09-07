import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

config = {
    'img_src': './images/original_image/',
    'threshold': 255,
    'block_size': 155,
    'val_c': 20,
    'edge_check': 15,
    'result_size': 0  # square side length of result
}


def find_edge(input_img, length, reverse):
    itr = []
    if reverse:
        itr = [length-x-1 for x in range(length)]
    else:
        itr = [x for x in range(length)]
    # find edge from line in one by one, edge_check = 15.
    for i in itr:
        cnt = 0
        for j in itr:
            if input_img[i][j] == 0:
                cnt += 1
                if cnt > config['edge_check']:
                    if reverse:
                        return length-i-1
                    else:
                        return i


if __name__ == "__main__":
    for i in range(1, 4):
        img = cv2.imread(config['img_src']  + str(i) + ".png", cv2.IMREAD_COLOR)  # read image as BGR.
        h, w, c = img.shape
        img_square = img[0:h, int((w-h)/2):int((w-h)/2)+h]  # crop out central square from img.
        # recommend changing this cropping to other technique.
        img_gray = cv2.cvtColor(img_square, cv2.COLOR_BGR2GRAY)
        # make gray-scale image for threshold operation(threshold_tester).
        img_bin = cv2.adaptiveThreshold(img_gray, config['threshold'], cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, config['block_size'], config['val_c'])

        # input_img = (img or rotated img), length = img 's height, reverse = flipping.
        top = find_edge(img_bin, h, False)
        bottom = find_edge(img_bin, h, True)
        left = find_edge(np.rot90(img_bin), h, True)
        right = find_edge(np.rot90(img_bin), h, False)

        # print("top : ", str(top))
        # print("bottom : ", str(bottom))
        # print("right : ", str(right))
        # print("left : ", str(left))
        # print("height : ", str(h-top-bottom))
        # print("width : ", str(h-left-right))

        # plt codes start
        fig = plt.figure()
        sp_img = plt.subplot(1, 1, 1)
        sp_img.imshow(cv2.cvtColor(img_square, cv2.COLOR_BGR2RGB))

        # sp_img.imshow(img_bin, cmap='gray', vmin=0, vmax=255)
        center_x = (h - left - right) / 2 + left  # plate's center_x
        center_y = (h - top - bottom) / 2 + top  # plate's center_y
        rect = patches.Rectangle(((center_x - 1000), (center_y - 1000)), 2000,
                                 2000, linewidth=1, edgecolor='b', facecolor='none')
        print(rect)

        sp_img.add_patch(rect)

        img_segmented = img_square[int(center_y - 1000):int(center_y + 1000)
        ,int(center_x - 1000):int(center_x + 1000)]

        print(img_segmented.shape)

        cv2.imwrite('./images/segmented_image/segmented_00' + str(i) + ".png", img_segmented)

    plt.show()