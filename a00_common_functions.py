# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import time
import glob
import cv2
import matplotlib.pyplot as plt
import random
import gzip
from sklearn.metrics import log_loss


def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=6))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def check_score(subm_file):
    real_answ = "../modified_data/answers_stage1.csv"
    real = pd.read_csv(real_answ)
    pred = pd.read_csv(subm_file)
    real['s'] = 0
    real.loc[real['Type_1'] > 0, 's'] = 0
    real.loc[real['Type_2'] > 0, 's'] = 1
    real.loc[real['Type_3'] > 0, 's'] = 2
    pred = pd.merge(pred, real[['image_name', 's']], on=['image_name'], left_index=True)
    score = log_loss(pred['s'], pred[['Type_1', 'Type_2', 'Type_3']].as_matrix())
    return score