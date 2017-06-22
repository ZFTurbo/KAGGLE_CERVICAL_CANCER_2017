# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from multiprocessing import Process, Manager
import numpy as np
import random
import cv2
import os

random.seed(2016)
np.random.seed(2016)


def get_bounding_boxes_positions(image, rw):
    s0 = rw['rect_v1_start0'].values[0]
    e0 = rw['rect_v1_end0'].values[0]
    s1 = rw['rect_v1_start1'].values[0]
    e1 = rw['rect_v1_end1'].values[0]

    start0 = rw['rect_v2_start0'].values[0]
    end0 = rw['rect_v2_end0'].values[0]
    start1 = rw['rect_v2_start1'].values[0]
    end1 = rw['rect_v2_end1'].values[0]

    red_coeff = (e0 - s0) / image.shape[0]
    start0_red = round((start0 - s0) / red_coeff)
    end0_red = round((end0 - s0) / red_coeff)
    start1_red = round((start1 - s1) / red_coeff)
    end1_red = round((end1 - s1) / red_coeff)

    return start0_red, end0_red, start1_red, end1_red


def return_random_crop(img, row):
    perc = 0.1
    start0_max, end0_max, start1_max, end1_max = get_bounding_boxes_positions(img, row)

    if start0_max <= 0:
        start0 = random.randint(0, int(img.shape[0] * perc))
    else:
        start0 = random.randint(0, start0_max)

    if end0_max >= img.shape[0]:
        end0 = img.shape[0] - random.randint(0, int(img.shape[0] * perc))
    else:
        end0 = random.randint(end0_max, img.shape[0])

    if start1_max <= 0:
        start1 = random.randint(0, int(img.shape[1] * perc))
    else:
        start1 = random.randint(0, start1_max)

    if end1_max >= img.shape[1]:
        end1 = img.shape[1] - random.randint(0, int(img.shape[1] * perc))
    else:
        end1 = random.randint(end1_max, img.shape[1])

    # print(img.shape)
    # print(start0, end0, start1, end1)
    # img = cv2.rectangle(img, (int(start1_max), int(start0_max)), (int(end1_max), int(end0_max)), (0, 0, 255), thickness=5)

    return img[start0:end0, start1:end1]


def return_random_perspective(img, row):
    perc = 0.1
    cols = img.shape[1]
    rows = img.shape[0]
    start0_max, end0_max, start1_max, end1_max = get_bounding_boxes_positions(img, row)

    if start1_max <= 0:
        p1 = random.randint(0, int(img.shape[1] * perc))
    else:
        p1 = random.randint(0, start1_max)

    if start0_max <= 0:
        p2 = random.randint(0, int(img.shape[0] * perc))
    else:
        p2 = random.randint(0, start0_max)

    if end1_max >= img.shape[1]:
        p3 = img.shape[1] - random.randint(0, int(img.shape[1] * perc))
    else:
        p3 = random.randint(end1_max, img.shape[1])

    if start0_max <= 0:
        p4 = random.randint(0, int(img.shape[0] * perc))
    else:
        p4 = random.randint(0, start0_max)

    if start1_max <= 0:
        p5 = random.randint(0, int(img.shape[1] * perc))
    else:
        p5 = random.randint(0, start1_max)

    if end0_max >= img.shape[0]:
        p6 = img.shape[0] - random.randint(0, int(img.shape[0] * perc))
    else:
        p6 = random.randint(end0_max, img.shape[0])

    if end1_max >= img.shape[1]:
        p7 = img.shape[1] - random.randint(0, int(img.shape[1] * perc))
    else:
        p7 = random.randint(end1_max, img.shape[1])

    if end0_max >= img.shape[0]:
        p8 = img.shape[0] - random.randint(0, int(img.shape[0] * perc))
    else:
        p8 = random.randint(end0_max, img.shape[0])

    pts1 = np.float32([[p1, p2], [p3, p4], [p5, p6], [p7, p8]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # img = cv2.rectangle(img, (int(start1_max), int(start0_max)), (int(end1_max), int(end0_max)), (0, 0, 255), thickness=5)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    # show_resized_image(dst)
    return dst


def random_rotate(image):
    cols = image.shape[1]
    rows = image.shape[0]
    mean_color = np.mean(image, axis=(0, 1))

    angle = random.uniform(0, 90)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    if random.randint(0, 1) == 1:
        dst = cv2.warpAffine(image, M, (cols, rows), borderValue=mean_color, borderMode=cv2.BORDER_REFLECT)
    else:
        dst = cv2.warpAffine(image, M, (cols, rows), borderValue=mean_color)
    return dst


def lightning_change(image):
    intencity = 20
    image = image.astype(np.int16)
    for i in range(3):
        r = random.randint(-intencity, intencity)
        image[:, :, i][(255-r > image[:, :, i]) & (image[:, :, i] >= -r)] += r
    image = image.astype(np.uint8)
    return image


def blur_image(image):
    if random.randint(0, 10) == 0:
        intencity = random.randint(1, 5)
        image = cv2.blur(image, (intencity, intencity))
    return image


def random_augment_image(image, row):
    # start0_max, end0_max, start1_max, end1_max = get_bounding_boxes_positions(image, row)
    # image = cv2.rectangle(image, (int(start1_max), int(start0_max)), (int(end1_max), int(end0_max)), (0, 0, 255), thickness=5)
    if random.randint(0, 1) == 0:
        image = return_random_crop(image, row)
    else:
        image = return_random_perspective(image, row)
    image = random_rotate(image)

    # all possible mirroring and flips (in total there are only 8 possible configurations)
    mirror = random.randint(0, 1)
    if mirror != 0:
        image = image[::-1, :, :]
    angle = random.randint(0, 3)
    if angle != 0:
        image = np.rot90(image, k=angle)

    image = lightning_change(image)
    image = blur_image(image)

    return image


def get_row_from_table(f, rectangle_table):
    id = os.path.basename(f)
    if 'reduced_train' in f:
        return rectangle_table[(rectangle_table['id'] == id) & (rectangle_table['type'] == 'train')]
    elif 'reduced_test' in f:
        return rectangle_table[(rectangle_table['id'] == id) & (rectangle_table['type'] == 'test')]
    return rectangle_table[(rectangle_table['id'] == id) & (rectangle_table['type'] == 'add')]


def get_augmented_image_list_single_part(proc_num, files, augment_number_per_image, input_shape, rectangle_table, return_dict):
    image_list = []
    # print('Start process: {}. Images {}'.format(proc_num, len(files)*augment_number_per_image))
    for ti in range(len(files)):
        image = cv2.imread(files[ti])
        rw = get_row_from_table(files[ti], rectangle_table)
        for j in range(augment_number_per_image):
            if j > 0:
                im1 = random_augment_image(image.copy(), rw)
            else:
                im1 = image.copy()
            im1 = cv2.resize(im1, input_shape, cv2.INTER_LINEAR)
            image_list.append(im1)
    return_dict[proc_num] = np.array(image_list)
    # print('Finished process {}'.format(proc_num))


def get_augmented_image_list(files, augment_number_per_image, input_shape, rectangle_table):
    # Split on parts
    threads = 6
    files_split = []
    step = len(files) // threads

    files_split.append(files[:step])
    for i in range(1, threads-1):
        files_split.append(files[i*step:(i+1)*step])
    files_split.append(files[(threads-1)*step:])

    manager = Manager()
    return_dict = manager.dict()
    p = dict()
    for i in range(threads):
        p[i] = Process(target=get_augmented_image_list_single_part, args=(i, files_split[i], augment_number_per_image, input_shape, rectangle_table, return_dict))
        p[i].start()
    for i in range(threads):
        p[i].join()
    # print('Return dictionary: ', len(return_dict), return_dict.keys())

    concat_list = []
    for i in range(threads):
        concat_list.append(return_dict[i])
    image_list = np.concatenate(concat_list)
    return np.array(image_list)

