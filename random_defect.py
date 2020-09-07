import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import imutils
import matplotlib.patches as patches

# 해야할 것 : 라벨링 순서 및 형식 알아서 지정하기.
# 다 되었으면, 흠집 이미지 받아서 segmented 1, 2, 3순서로 데이터 만들기.

"""
overlay image must be a square shape image 
"""
config = {
    'img_src_path' : './images/segmented_image/segmented_002.png',
    'img_defect_path' : './defects/%s/%d.png',
    'data_path' : "./images/data/%d.jpg",
    'labeled_img_path' : './images/labeled_img/%d.jpg',
    'labeled_text_path' : './images/labeled_img/%d.txt',
    'repeat_num' : 1000,
    'defect_num': 4,
    'blemish_num' : 38,
    'scratch_num' : 15
}


# put defected img on background img.
def defect_overlay(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background, x , y, overlay.shape[1], overlay.shape[0]


def is_in_circle(r, c_x, c_y, x, y):
    if ((x-c_x)**2) + ((y-c_y)**2) < r**2:
        return True
    else:
        return False


def random_overlay(img, overlay):
    h, w, c = img.shape
    center_x = int(w / 2)
    center_y = int(h / 2)
    # r is the average radius of three panels(960) - 35(because of boundary treatment)
    r = 925
    oh, ow, oc = overlay.shape

    # determine defected img, Check that the four vertices fo the defect image are in the panel.
    while True:
        x = random.randrange(60, 1940)
        y = random.randrange(50, 1940)
        x1 = x - int(ow / 2)
        y1 = y - int(oh / 2)
        x2 = x + int(ow / 2)
        y2 = y - int(oh / 2)
        x3 = x - int(ow / 2)
        y3 = y + int(oh / 2)
        x4 = x + int(ow / 2)
        y4 = y + int(oh / 2)
        if is_in_circle(r, center_x, center_y, x1, y1) and is_in_circle(r, center_x, center_y, x2, y2) and \
                is_in_circle(r, center_x, center_y, x3, y3) and is_in_circle(r, center_x, center_y, x4, y4):
            break

    # overlay image random transform
    flip_yn = random.randrange(0, 2)  # flip_no = 0, flip_yes = 1
    rotation_degree = random.randrange(0, 360)
    if flip_yn == 1:
        overlay = cv2.flip(overlay, 0)
    overlay = imutils.rotate(overlay, rotation_degree)

    oh, ow, oc = overlay.shape
    # return defect_overlay(img, overlay, x - int(ow/2), y - int(oh/2))
    return defect_overlay(img, overlay, x - int(ow/2), y - int(oh/2))


if __name__ == "__main__":
    # make data set randomly.
    for img_num in range(0, config['repeat_num']):
        defect_coordinate_wh = []
        img = cv2.imread(config['img_src_path'], cv2.IMREAD_UNCHANGED)  # read image as BGR
        kind_defect = random.randrange(0, 2)
        if kind_defect == 0:
            kind_defect = "blemish"
            kind_type = 0
            defect_num = random.randrange(1, config['blemish_num'] + 1)
        else:
            kind_defect = "scratch"
            kind_type = 1
            defect_num = random.randrange(1, config['scratch_num'] + 1)
        img_defect = cv2.imread(config['img_defect_path'] % (kind_defect, defect_num) , cv2.IMREAD_UNCHANGED)

        # the number of times repeat
        img_merged, top_left_x, top_left_y, overlay_w, overlay_h = random_overlay(img, img_defect)
        center_x = (top_left_x + overlay_w) / 2
        center_y = (top_left_y + overlay_h) / 2
        defect_coordinate_wh.append([top_left_x, top_left_y, overlay_w, overlay_h, kind_type, center_x, center_y])
        for i in range(config['defect_num'] - 1):
            kind_defect = random.randrange(0, 2)
            if kind_defect == 0:
                kind_defect = "blemish"
                kind_type = 0
                defect_num = random.randrange(1, config['blemish_num'] + 1)
            else:
                kind_defect = "scratch"
                kind_type = 1
                defect_num = random.randrange(1, config['scratch_num'] + 1)
            img_defect = cv2.imread(config['img_defect_path'] % (kind_defect, defect_num), cv2.IMREAD_UNCHANGED)
            img_merged, top_left_x, top_left_y, overlay_w, overlay_h = random_overlay(img_merged, img_defect)
            center_x = top_left_x + overlay_w / 2
            center_y = top_left_y + overlay_h / 2
            defect_coordinate_wh.append([top_left_x, top_left_y, overlay_w, overlay_h, kind_type, center_x, center_y])

        # plt codes start
        fig = plt.figure()
        sp_img = plt.subplot(1, 1, 1)
        sp_img.imshow(cv2.cvtColor(img_merged, cv2.COLOR_BGR2RGB))

        # find defect_label
        for i in range(0, len(defect_coordinate_wh)):
            if defect_coordinate_wh[i][4] == 0:
                rect = patches.Rectangle((defect_coordinate_wh[i][0], defect_coordinate_wh[i][1]), defect_coordinate_wh[i][2],
                                        defect_coordinate_wh[i][3], linewidth=1, edgecolor='b', facecolor='none')
            else:
                rect = patches.Rectangle((defect_coordinate_wh[i][0], defect_coordinate_wh[i][1]),
                                         defect_coordinate_wh[i][2],
                                         defect_coordinate_wh[i][3], linewidth=1, edgecolor='r', facecolor='none')
            sp_img.add_patch(rect)

        # save_labeled_img
        fig.savefig(config['labeled_img_path'] % (img_num + 2001), format='png', dpi=300)
        # save_labeled_value
        f = open(config['labeled_text_path'] % (img_num + 2001), 'w')
        # data = "# top_left_x_value, top_left_y_value, overlay_w_value, overlay_h_value, kind_defect\n"
        # f.write(data)
        for i in range(0, len(defect_coordinate_wh)):
            for j in range(0, len(defect_coordinate_wh[i])):
                if j == 4:
                    continue
                defect_coordinate_wh[i][j] /= 2000
            data = str(defect_coordinate_wh[i][4]) + " "  # class : 0 = blemish, 1 = scratch
            data += str(defect_coordinate_wh[i][5]) + " "  # center_x
            data += str(defect_coordinate_wh[i][6]) + " "  # center_y
            data += str(defect_coordinate_wh[i][2]) + " "  # width
            data += str(defect_coordinate_wh[i][3]) + "\n"  # height
            f.write(data)

        # cv2.imwrite("./images/new_data/%d.png" % (img_num + 1), cv2.cvtColor(img_merged, cv2.COLOR_BGR2RGB))
        cv2.imwrite(config['data_path'] % (img_num + 2001), img_merged)
        print(config['data_path'] % (img_num + 2001))
        f.close()
        # plt.show()
        plt.close()