# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import math
from a00_common_functions import *


INPUT_PATH = "../input/train/"
INPUT_PATH_TEST = "../input/test/"
ADDITIONAL_PATH = "../input/additional/"
OUTPUT_PATH = "../modified_data/reduced_train/"
OUTPUT_PATH_TEST = "../modified_data/reduced_test/"
OUTPUT_PATH_ADDITIONAL = "../modified_data/reduced_additional/"
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
if not os.path.isdir(OUTPUT_PATH_TEST):
    os.mkdir(OUTPUT_PATH_TEST)
if not os.path.isdir(OUTPUT_PATH_ADDITIONAL):
    os.mkdir(OUTPUT_PATH_ADDITIONAL)
for i in range(1, 4):
    if not os.path.isdir(OUTPUT_PATH + 'Type_{}'.format(i)):
        os.mkdir(OUTPUT_PATH + 'Type_{}'.format(i))
    if not os.path.isdir(OUTPUT_PATH_ADDITIONAL + 'Type_{}'.format(i)):
        os.mkdir(OUTPUT_PATH_ADDITIONAL + 'Type_{}'.format(i))


def read_rectangles_tables():
    s1 = pd.read_csv("../modified_data/rectangles_from_unet_v1.csv")
    s2 = pd.read_csv("../modified_data/rectangles_from_unet_v2.csv")
    s2.drop(['img_shape0', 'img_shape1'], axis=1, inplace=True)
    len1 = len(s1)
    s = pd.merge(s1, s2, how='left', on=['id', 'type'], left_index=True)
    len2 = len(s)
    if len1 != len2:
        print('Some problem here!', len1, len2)
        exit()
    return s.copy()


def check_and_merge_rectangles():
    table = read_rectangles_tables()
    for index, rw in table.iterrows():
        id = rw['id']
        type = rw['type']
        rect_v2_start0 = rw['rect_v2_start0']
        rect_v2_end0 = rw['rect_v2_end0']
        rect_v2_start1 = rw['rect_v2_start1']
        rect_v2_end1 = rw['rect_v2_end1']
        rect_v1_start0 = rw['rect_v1_start0']
        rect_v1_end0 = rw['rect_v1_end0']
        rect_v1_start1 = rw['rect_v1_start1']
        rect_v1_end1 = rw['rect_v1_end1']

        if rect_v1_start0 > rect_v2_start0 and rect_v2_start0 != -1:
            # print('rect_v2_start0 error', id, type)
            # print(rect_v1_start0, rect_v2_start0)
            table.loc[(table['id'] == id) & (table['type'] == type), 'rect_v1_start0'] = rect_v2_start0
        if rect_v1_start1 > rect_v2_start1 and rect_v2_start1 != -1:
            # print('rect_v2_start1 error', id, type)
            # print(rect_v1_start1, rect_v2_start1)
            table.loc[(table['id'] == id) & (table['type'] == type), 'rect_v1_start1'] = rect_v2_start1
        if rect_v1_end0 < rect_v2_end0:
            # print('rect_v1_end0 error', id, type)
            # print(rect_v1_end0, rect_v2_end0)
            table.loc[(table['id'] == id) & (table['type'] == type), 'rect_v1_end0'] = rect_v2_end0
        if rect_v1_end1 < rect_v2_end1:
            # print('rect_v1_end1 error', id, type)
            # print(rect_v1_end1, rect_v2_end1)
            table.loc[(table['id'] == id) & (table['type'] == type), 'rect_v1_end1'] = rect_v2_end1
        if rect_v2_start0 == -1:
            print('Sub ', rect_v1_start0)
            table.loc[(table['id'] == id) & (table['type'] == type), 'rect_v2_start0'] = rect_v1_start0
        if rect_v2_end0 == -1:
            print('Sub ', rect_v1_end0)
            table.loc[(table['id'] == id) & (table['type'] == type), 'rect_v2_end0'] = rect_v1_end0
        if rect_v2_start1 == -1:
            print('Sub ', rect_v1_start1)
            table.loc[(table['id'] == id) & (table['type'] == type), 'rect_v2_start1'] = rect_v1_start1
        if rect_v2_end1 == -1:
            print('Sub ', rect_v1_end1)
            table.loc[(table['id'] == id) & (table['type'] == type), 'rect_v2_end1'] = rect_v1_end1

    table.to_csv("../modified_data/rectangles_merged.csv", index=False)


def recreate_images_based_on_rectangles():
    table = pd.read_csv("../modified_data/rectangles_merged.csv")

    for type in ['train', 'test', 'add']:
        if type == 'train':
            files = glob.glob(INPUT_PATH + '*/*.jpg')
        elif type == 'test':
            files = glob.glob(INPUT_PATH_TEST + '*.jpg')
        else:
            files = glob.glob(ADDITIONAL_PATH + '*/*.jpg')

        min_required_size = 300
        for f in files:

            print('Go for {}'.format(f))
            id = os.path.basename(f)
            if type == 'train':
                out_folder = OUTPUT_PATH + os.path.basename(os.path.dirname(f)) + '/'
            elif type == 'test':
                out_folder = OUTPUT_PATH_TEST
            else:
                out_folder = OUTPUT_PATH_ADDITIONAL + os.path.basename(os.path.dirname(f)) + '/'
            out = out_folder + id
            if os.path.isfile(out):
                print('Already exists. Skip!')
                continue

            rw = table[(table['id'] == id) & (table['type'] == type)]
            im = cv2.imread(f)

            # Get only needed subpart of image
            s0 = rw['rect_v1_start0'].values[0]
            e0 = rw['rect_v1_end0'].values[0]
            s1 = rw['rect_v1_start1'].values[0]
            e1 = rw['rect_v1_end1'].values[0]
            im = im[s0:e0, s1:e1, :]

            # Fix coordinates
            start0 = rw['rect_v2_start0'].values[0] - s0
            end0 = rw['rect_v2_end0'].values[0] - s0
            start1 = rw['rect_v2_start1'].values[0] - s1
            end1 = rw['rect_v2_end1'].values[0] - s1

            sh0_size = end0 - start0
            sh1_size = end1 - start1
            if sh0_size >= min_required_size and sh0_size >= min_required_size:
                if sh0_size < sh1_size:
                    red_koeff = sh0_size / min_required_size
                else:
                    red_koeff = sh1_size / min_required_size
                print('Reduction koefficient: {}'.format(red_koeff))

                im = cv2.resize(im, (math.ceil(im.shape[1] / red_koeff), math.ceil(im.shape[0] / red_koeff)), cv2.INTER_LANCZOS4)
                print(im.shape)
                # show_image(im)
            else:
                print('No reduction here!')

            cv2.imwrite(out, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def check_small_rectangle_get():
    table = pd.read_csv("../modified_data/rectangles_merged.csv")
    for type in ['train']:
        if type == 'train':
            files = glob.glob(INPUT_PATH + '*/*.jpg')
        elif type == 'test':
            files = glob.glob(INPUT_PATH_TEST + '*.jpg')
        else:
            files = glob.glob(ADDITIONAL_PATH + '*/*.jpg')


        for f in files[:10]:
            print('Go for {}'.format(f))
            id = os.path.basename(f)
            rw = table[(table['id'] == id) & (table['type'] == type)]
            path_reduced = OUTPUT_PATH + os.path.basename(os.path.dirname(f)) + '/' + id
            im_orig = cv2.imread(f)
            im_red = cv2.imread(path_reduced)

            s0 = rw['rect_v1_start0'].values[0]
            e0 = rw['rect_v1_end0'].values[0]
            s1 = rw['rect_v1_start1'].values[0]
            e1 = rw['rect_v1_end1'].values[0]

            start0 = rw['rect_v2_start0'].values[0]
            end0 = rw['rect_v2_end0'].values[0]
            start1 = rw['rect_v2_start1'].values[0]
            end1 = rw['rect_v2_end1'].values[0]

            # Initial
            im_rect = cv2.rectangle(im_orig, (start1, start0), (end1, end0), (0, 0, 255), thickness=5)
            show_resized_image(im_rect)

            # Cropped
            red_coeff = (e0 - s0) / im_red.shape[0]
            print(red_coeff)

            start0_red = round((start0 - s0) / red_coeff)
            end0_red = round((end0 - s0) / red_coeff)
            start1_red = round((start1 - s1) / red_coeff)
            end1_red = round((end1 - s1) / red_coeff)

            im_rect = cv2.rectangle(im_red, (int(start1_red), int(start0_red)), (int(end1_red), int(end0_red)), (0, 0, 255), thickness=5)
            show_image(im_rect)

    return


