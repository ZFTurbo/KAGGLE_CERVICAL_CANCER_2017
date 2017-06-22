# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
from a25_unet_training_v1_on_my_segmentation import run_cross_validation_create_models_unet1, \
    predict_on_train_and_additional_images_unet1, \
    predict_on_test_images_unet1, \
    create_csv_file_from_predictions_unet1
from a25_unet_training_v2_on_boxes import run_cross_validation_create_models_unet2, \
    predict_on_train_and_additional_images_unet2, \
    predict_on_test_images_unet2, \
    create_csv_file_from_predictions_unet2
from a26_merge_rectangles_and_reduce_images import check_and_merge_rectangles, recreate_images_based_on_rectangles
from a30_pretrained_nets_pipeline_with_additional_data import run_cross_validation_create_models, run_validation, \
    run_test, read_rectangles_table
from a40_run_xgboost_blender import read_data_table_train, run_kfold_create_models, read_data_table_test, \
    kfold_predict_on_test_and_create_submission
import shutil


# Set 1 in case you want to retrain all CNN models from scratch (very long process)
RETRAIN_ALL_MODELS = 1
# Possible values 0 or 1. 0 - with additional data used for training, 1 - without additional data.
# It defines 2 submissions on leaderboard!
SUBMISSION_VERSION = 1

INPUT_PATH = '../input/'
TEST_STAGE2_FOLDER_PATH = '../input/test2/'
TEST_FOLDER_PATH = '../input/test/'
OUTPUT_PATH = '../subm/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


def copy_first_stage_test_to_train():
    if 1:
        file_with_labels = "../modified_data/answers_stage1.csv"
        tbl = pd.read_csv(file_with_labels)
        for index, row in tbl.iterrows():
            img = row['image_name']
            id = 2000 + int(img[:-4])
            in_path = INPUT_PATH + 'test/' + img
            if row['Type_1'] == 1:
                out_path = INPUT_PATH + 'train/Type_1/' + str(id) + '.jpg'
            elif row['Type_2'] == 1:
                out_path = INPUT_PATH + 'train/Type_2/' + str(id) + '.jpg'
            elif row['Type_3'] == 1:
                out_path = INPUT_PATH + 'train/Type_3/' + str(id) + '.jpg'
            else:
                print('Some problem here!')
                exit()
            shutil.move(in_path, out_path)

    # Move stage 2 data on place of stage 1
    shutil.rmtree(TEST_FOLDER_PATH)
    shutil.move(TEST_STAGE2_FOLDER_PATH, TEST_FOLDER_PATH)


if __name__ == '__main__':
    num_folds = 5
    cnn_list = ['VGG16', 'VGG19', 'RESNET50', 'INCEPTION_V3', 'SQUEEZE_NET', 'DENSENET_161', 'DENSENET_121']
    copy_first_stage_test_to_train()

    if RETRAIN_ALL_MODELS:
        score = run_cross_validation_create_models_unet1(num_folds)
        print('UNET v1 score: ', score)
        score = run_cross_validation_create_models_unet2(num_folds)
        print('UNET v2 score: ', score)

    if 1:
        predict_on_train_and_additional_images_unet1(num_folds)
        predict_on_test_images_unet1(num_folds)
        create_csv_file_from_predictions_unet1()
        predict_on_train_and_additional_images_unet2(num_folds)
        predict_on_test_images_unet2(num_folds)
        create_csv_file_from_predictions_unet2()

        check_and_merge_rectangles()
        recreate_images_based_on_rectangles()

    if RETRAIN_ALL_MODELS:
        for cnn in cnn_list:
            read_rectangles_table()
            score = run_cross_validation_create_models(cnn, num_folds, SUBMISSION_VERSION)
            print('Score for {}: {} (Submission version: {})'.format(cnn, score, SUBMISSION_VERSION))

    valid_files = []
    test_files = []
    if 1:
        for cnn in cnn_list:
            score2, valid_subm_file = run_validation(cnn, num_folds, SUBMISSION_VERSION)
            subm_file = OUTPUT_PATH + 'subm_{}_LS_{}_test.csv'.format(cnn, score2)
            run_test(cnn, num_folds, subm_file, SUBMISSION_VERSION)
            valid_files.append(valid_subm_file)
            test_files.append(subm_file)

    print('Valid files:', valid_files)
    print('Test files:', test_files)
    train, features = read_data_table_train(valid_files)
    print('Features [{}]: {}'.format(len(features), features))
    model_list, best_score = run_kfold_create_models(5, train, features)
    test = read_data_table_test(test_files)
    subm_file = kfold_predict_on_test_and_create_submission(model_list, best_score, test, features)
    print('Submission file stored in {}'.format(subm_file))

