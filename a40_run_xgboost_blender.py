# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, matthews_corrcoef
import xgboost as xgb
import random
import pickle
import os
from operator import itemgetter
import zipfile
import time
import shutil
import copy
from scipy.sparse import csr_matrix, hstack
import gc
import glob
from operator import itemgetter
import heapq
from a00_common_functions import *
random.seed(2016)
np.random.seed(2016)

INPUT_PATH = "../input/"
OUTPUT_PATH = "../subm/"


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))


def save_features_importance(out_file, features, imp):
    out = open(out_file, "w")
    out.write(str(features) + '\n\n')
    out.write(str(imp) + '\n\n')
    out.close()


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def run_kfold_create_models(nfolds, train, features, random_state=44):

    model_list = []
    total_score = 0
    iter_num = 500
    for z in range(iter_num):
        print("Iteration: {}".format(z))
        eta = random.uniform(0.1, 0.35)
        max_depth = random.randint(1, 3)
        subsample = random.uniform(0.5, 0.95)
        colsample_bytree = random.uniform(0.5, 0.95)
        start_time = time.time()
        target = 'target'

        print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth,
                                                                                                    subsample,
                                                                                                    colsample_bytree))
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "booster": "gbtree",
            "eval_metric": "mlogloss",
            "eta": eta,
            "tree_method": 'exact',
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "seed": random_state,
            # 'gpu_id': 0,
            # 'updater': 'grow_gpu',
            # "nthreads": 6,
        }
        num_boost_round = 2000
        early_stopping_rounds = 12


        ti = []
        yfull_train = dict()
        kf = KFold(len(train.index), n_folds=nfolds, shuffle=True, random_state=random_state + z)
        num_fold = 0
        for train_index, test_index in kf:
            params['seed'] = random_state + num_fold
            num_fold += 1
            print('Start fold {} from {}'.format(num_fold, nfolds))
            X_train = train[features].loc[train_index]
            X_valid = train[features].loc[test_index]
            y_train = train[target].loc[train_index]
            y_valid = train[target].loc[test_index]
            # print('Length train:', X_train.shape[0], y_train.shape[0])
            # print('Length valid:', X_valid.shape[0], y_valid.shape[0])

            dtrain = xgb.DMatrix(X_train, y_train)
            dvalid = xgb.DMatrix(X_valid, y_valid)

            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

            # print("Validating...")
            yhat = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)
            yval = y_valid.tolist()

            # Each time store portion of predicted data in train predicted values
            ti.append(test_index)
            for i in range(len(test_index)):
                yfull_train[test_index[i]] = yhat[i]

            best_score = log_loss(yval, yhat)
            # print(best_score)

            # print("Saving model...")
            model_path = os.path.join('models', 'model_eta_' + str(eta) + '_md_'
                                      + str(max_depth)
                                      + '_iter_' + str(gbm.best_iteration)
                                      + '_kfold_' + str(num_fold) + '_z_' + str(z)
                                      + '_score_' + str(best_score) + '.bin')
            # gbm.save_model(model_path)
            imp = get_importance(gbm, features)
            print('Importance: ', imp)
            model_list.append((gbm, gbm.best_iteration))

        # Copy dict to list
        train_res = []
        for i in sorted(yfull_train.keys()):
            train_res.append(yfull_train[i])

        best_score = log_loss(train[target].tolist(), train_res)
        total_score += best_score
        print('Score fold: {}'.format(best_score))

    total_score /= iter_num
    print('Overall score: {}'.format(total_score))
    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return model_list, total_score


def kfold_predict_on_test_and_create_submission(model_list, score, test, features):
    print("Predict test set...")
    ids = test['image_name'].values
    test_prediction = []
    matrix = xgb.DMatrix(test[features])
    for m in model_list:
        # print("Loading {}...".format(m))
        gbm = m[0]
        predicted = gbm.predict(matrix, ntree_limit=m[1]+1)
        test_prediction.append(predicted)

    preds = np.mean(test_prediction, axis=0)
    now = datetime.datetime.now()
    sub_file = OUTPUT_PATH + 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('image_name,Type_1,Type_2,Type_3\n')
    for i in range(len(preds)):
        f.write(str(ids[i]))
        for j in range(len(preds[i])):
            f.write(',' + str(preds[i][j]))
        f.write('\n')
    f.close()

    # Check score (only debug purpose)
    if 1:
        print('Score on LB: {}'.format(check_score(sub_file)))

    # Copy code
    shutil.copy2(__file__, sub_file + ".py")
    return sub_file


def get_features(table):
    features = list(table.columns.values)
    remove_list = ['image_name', 'target']
    for i in remove_list:
        if i in features:
            features.remove(i)
    return features


def get_train_table():
    files = glob.glob(INPUT_PATH + "train/*/*.jpg")
    lst = []
    for f in files:
        bn = os.path.basename(f)
        clss = int(os.path.basename(os.path.dirname(f)).split('_')[1]) - 1
        lst.append([bn, clss])
    df = pd.DataFrame(lst, columns=['image_name', 'target'])
    return df


def read_data_table_train(valid_tables):
    train = get_train_table()
    l1 = len(train)
    i = 0
    for t_path in valid_tables:
        data = pd.read_csv(t_path)
        data.rename(columns={'Type_1': 'Type_1_num_{}'.format(i),
                             'Type_2': 'Type_2_num_{}'.format(i),
                             'Type_3': 'Type_3_num_{}'.format(i),
                             }, inplace=True)
        train = pd.merge(train, data, how='left', on='image_name', left_index=True)
        i += 1


    '''
    resolutions = pd.read_csv("../modified_data/resolutions_and_color_features_1.csv")
    resolutions = resolutions[resolutions['type'] == 'train']
    resolutions.drop(['type'], axis=1, inplace=True)
    train = pd.merge(train, resolutions, how='left', on='image_name', left_index=True)
    '''

    train.reset_index(drop=True, inplace=True)
    l2 = len(train)
    if l1 != l2:
        print('Error with merge!')
        exit()

    features = get_features(train)
    print(train['target'].describe())
    return train, features


def read_data_table_test(test_tables):
    test = pd.read_csv('../input/sample_submission.csv')[['image_name']]
    i = 0
    for t_path in test_tables:
        data = pd.read_csv(t_path)
        data.rename(columns={'Type_1': 'Type_1_num_{}'.format(i),
                             'Type_2': 'Type_2_num_{}'.format(i),
                             'Type_3': 'Type_3_num_{}'.format(i),
                             }, inplace=True)
        test = pd.merge(test, data, how='left', on='image_name', left_index=True)
        i += 1

    '''
    resolutions = pd.read_csv("../modified_data/resolutions_and_color_features_1.csv")
    resolutions = resolutions[resolutions['type'] == 'test']
    resolutions.drop(['type'], axis=1, inplace=True)
    test = pd.merge(test, resolutions, how='left', on='image_name', left_index=True)
    '''

    return test
