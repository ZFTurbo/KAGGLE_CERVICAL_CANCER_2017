# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *
import json


INPUT_PATH = "../input/train/"
OUTPUT_PATH = "../modified_data/train_unet/"
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
for i in range(1, 4):
    if not os.path.isdir(OUTPUT_PATH + 'Type_{}'.format(i)):
        os.mkdir(OUTPUT_PATH + 'Type_{}'.format(i))


def create_unet_training_data_from_poly():
    res = dict()
    poly_path = "../modified_data/polygons.json"
    struct = json.load(open(poly_path, "r"))
    for i in range(len(struct)):
        # print(struct[i])
        f = os.path.join(INPUT_PATH, struct[i]['filename'])
        if not os.path.isfile(f):
            print('Strange! No file: {}'.format(f))
            continue
        out_path = OUTPUT_PATH + struct[i]['filename']
        if os.path.isfile(out_path):
            print('Skip {}'.format(out_path))
            continue
        if len(struct[i]['annotations']) == 0:
            print('Absent')
            continue
        annot = struct[i]['annotations'][0]
        img = cv2.imread(f)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if annot['class'] == 'polygon':
            y_arr = annot['yn'].split(";")
            x_arr = annot['xn'].split(";")
            # print(x_arr)
            # print(y_arr)
            color = (255, 255, 255)
            contour = []
            for j in range(len(y_arr)):
                pixelW = int(round(float(x_arr[j])))
                pixelH = int(round(float(y_arr[j])))
                contour.append((pixelW, pixelH))
            ctr = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(mask, [ctr], 0, color, -1)
        else:
            print('Rectangle')
            mask[:, :] = 255

        img = cv2.resize(img.astype(np.uint8), (800, 800), interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask.astype(np.uint8), (800, 800), interpolation=cv2.INTER_LANCZOS4)
        mask[mask > 10] = 255
        mask[mask <= 10] = 0

        cv2.imwrite(out_path[:-4] + '.png', img)
        cv2.imwrite(out_path[:-4] + '_mask.png', mask)
        # show_image(img)
        # show_image(mask)

        print(f, img.shape)
