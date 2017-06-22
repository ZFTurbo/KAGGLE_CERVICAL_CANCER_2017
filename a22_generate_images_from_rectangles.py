# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *
import struct
import imghdr

INPUT_PATH_TRAIN = "../input/train/"
INPUT_PATH_TEST = "../input/test/"
ADDITIONAL_PATH = "../input/additional/"
OUTPATH_TRAIN = "../modified_data/train_crops_kaggle_script_v1/"
if not os.path.isdir(OUTPATH_TRAIN):
    os.mkdir(OUTPATH_TRAIN)
for i in range(1, 4):
    if not os.path.isdir(OUTPATH_TRAIN + 'Type_{}'.format(i)):
        os.mkdir(OUTPATH_TRAIN + 'Type_{}'.format(i))
OUTPATH_TEST = "../modified_data/test_crops_kaggle_script_v1/"
if not os.path.isdir(OUTPATH_TEST):
    os.mkdir(OUTPATH_TEST)


def generate_images_from_rects_train():
    rect = pd.read_csv("../modified_data/rectangles_train.csv")
    for index, row in rect.iterrows():
        # print(row)
        clss = row['clss'] + 1
        fname = INPUT_PATH_TRAIN + 'Type_{}'.format(clss) + '/' + row['image_name']
        image = cv2.imread(fname)
        if image is None:
            print('Image problem {}'.format(fname))
            continue
        # show_resized_image(image)
        real_0_start = int(row['sh0_start'] * row['img_shp_0_init'] / row['img_shp_0'])
        real_1_start = int(row['sh1_start'] * row['img_shp_1_init'] / row['img_shp_1'])
        real_0_end = int(row['sh0_end'] * row['img_shp_0_init'] / row['img_shp_0'])
        real_1_end = int(row['sh1_end'] * row['img_shp_1_init'] / row['img_shp_1'])
        print(real_0_start, real_0_end, real_1_start, real_1_end)
        subimg = image[real_0_start:real_0_end, real_1_start:real_1_end]
        # show_resized_image(subimg)
        subimg = cv2.resize(subimg, (800, 800), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(OUTPATH_TRAIN + 'Type_{}'.format(clss) + '/' + row['image_name'], subimg)


def generate_images_from_rects_test():
    rect = pd.read_csv("../modified_data/rectangles_test.csv")
    for index, row in rect.iterrows():
        # print(row)
        fname = INPUT_PATH_TEST + row['image_name']
        image = cv2.imread(fname)
        if image is None:
            print('Image problem {}'.format(fname))
            continue
        # show_resized_image(image)
        real_0_start = int(row['sh0_start'] * row['img_shp_0_init'] / row['img_shp_0'])
        real_1_start = int(row['sh1_start'] * row['img_shp_1_init'] / row['img_shp_1'])
        real_0_end = int(row['sh0_end'] * row['img_shp_0_init'] / row['img_shp_0'])
        real_1_end = int(row['sh1_end'] * row['img_shp_1_init'] / row['img_shp_1'])
        print(real_0_start, real_0_end, real_1_start, real_1_end)
        subimg = image[real_0_start:real_0_end, real_1_start:real_1_end]
        # show_resized_image(subimg)
        subimg = cv2.resize(subimg, (800, 800), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(OUTPATH_TEST + row['image_name'], subimg)


