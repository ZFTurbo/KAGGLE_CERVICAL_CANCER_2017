# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *


INPUT_PATH = "../input/train/"
OUTPUT_PATH = "../modified_data/train_unet_v2/"
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
for i in range(1, 4):
    if not os.path.isdir(OUTPUT_PATH + 'Type_{}'.format(i)):
        os.mkdir(OUTPUT_PATH + 'Type_{}'.format(i))


def create_unet_mask_from_bboxes(res):
    files = glob.glob(INPUT_PATH + "*/*.jpg")
    bbox_absent = 0
    for f in files:
        type = int(f.split('\\')[-2].split('_')[1])
        id = f.split('\\')[-1]
        img = cv2.imread(f)
        mask = np.zeros(img.shape)

        out_f = f.replace('input', 'modified_data')
        out_f = out_f.replace('train', 'train_unet_v2')
        out_f = out_f.replace('.jpg', '_mask.png')

        out_f1 = f.replace('input', 'modified_data')
        out_f1 = out_f1.replace('train', 'train_unet_v2')
        out_f1 = out_f1.replace('.jpg', '.png')
        img = cv2.resize(img, (800, 800), cv2.INTER_LANCZOS4)
        cv2.imwrite(out_f1, img)

        if (type, id) in res:
            x = int(res[(type, id)][0])
            y = int(res[(type, id)][1])
            w = int(res[(type, id)][2])
            h = int(res[(type, id)][3])
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)
            # show_resized_image(img)
            # print(out_f)
            # img = cut_black_borders(img)
            mask[y:y+h, x:x+w] = 255
        else:
            print('!! BBox absent!')
            bbox_absent += 1
        mask = cv2.resize(mask, (800, 800), cv2.INTER_LANCZOS4)
        cv2.imwrite(out_f, mask)

    print('Absent: {}'.format(bbox_absent))


def read_rectangles():
    res = dict()
    for clss in range(1, 4):
        f1 = open("../modified_data/manual_segmentation_box_kaggle/Type_{}_bbox.tsv".format(clss))
        while 1:
            line = f1.readline().strip()
            if line == '':
                break
            arr = line.split(" ")
            id = arr[0].split('\\')[1].strip()
            num_boxes = int(arr[1])
            res[(clss, id)] = arr[2:]
        f1.close()
    return res
