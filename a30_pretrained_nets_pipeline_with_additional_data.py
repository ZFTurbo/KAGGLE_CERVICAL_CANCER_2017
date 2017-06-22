# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import platform
import datetime
import shutil
from a00_common_functions import *
from a02_zoo import *
from a03_augmentation import *

random.seed(2016)
np.random.seed(2016)


CLASSES_NUMBER = 3
PATIENCE = 30
NB_EPOCH = 250
MAX_IMAGES_FOR_INFERENCE = 12000 # Increase if you have much of memory
RESTORE_FROM_LAST_CHECKPOINT = 0
TABLE_WITH_RECTANGLES = None
IMAGE_ARRAY = None

# Use cropped with both UNETs
INPUT_PATH = "../modified_data/reduced_train/"
INPUT_PATH_TEST = "../modified_data/reduced_test/"
INPUT_PATH_ADD = "../modified_data/reduced_additional/"


MODELS_PATH = '../models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
OUTPUT_PATH = "../subm/"
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
CODE_COPY_FOLDER = "../models/code/"
if not os.path.isdir(CODE_COPY_FOLDER):
    os.mkdir(CODE_COPY_FOLDER)
HISTORY_FOLDER_PATH = "../models/history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)


def get_row_from_table(f):
    id = os.path.basename(f)
    if 'reduced_train' in f:
        return TABLE_WITH_RECTANGLES[(TABLE_WITH_RECTANGLES['id'] == id) & (TABLE_WITH_RECTANGLES['type'] == 'train')]
    elif 'reduced_test' in f:
        return TABLE_WITH_RECTANGLES[(TABLE_WITH_RECTANGLES['id'] == id) & (TABLE_WITH_RECTANGLES['type'] == 'test')]
    return TABLE_WITH_RECTANGLES[(TABLE_WITH_RECTANGLES['id'] == id) & (TABLE_WITH_RECTANGLES['type'] == 'add')]


def batch_generator_train(cnn, files, batch_size, additional_files, augment):

    if additional_files is not None:
        if cnn == 'DENSENET_161':
            batch_additional = 2
        else:
            batch_additional = 10
        batch_normal = batch_size - batch_additional
        random.shuffle(additional_files)
    else:
        batch_normal = batch_size

    random.shuffle(files)
    number_of_batches = np.floor(len(files) / batch_normal)
    counter = 0

    while True:
        if additional_files is not None:
            batch_files = files[batch_normal*counter:batch_normal*(counter+1)] + additional_files[batch_additional*counter:batch_additional*(counter+1)]
        else:
            batch_files = files[batch_normal * counter:batch_normal * (counter + 1)]
        # print(batch_size*counter)
        image_list = []
        mask_list = []
        for f in batch_files:
            image = cv2.imread(f)
            clss = int(os.path.basename(os.path.dirname(f)).split('_')[1]) - 1
            if augment:
                rw = get_row_from_table(f)
                # print(rw, f)
                image = random_augment_image(image, rw)
            image = cv2.resize(image, get_input_shape(cnn), cv2.INTER_LANCZOS4)

            mask = [0] * CLASSES_NUMBER
            mask[clss] = 1

            image_list.append(image.astype(np.float32))
            mask_list.append(mask)
        counter += 1
        image_list = np.array(image_list)
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_input_overall(cnn, image_list)
        mask_list = np.array(mask_list)

        yield image_list, mask_list
        if counter >= number_of_batches:
            random.shuffle(files)
            if additional_files is not None:
                random.shuffle(additional_files)
            counter = 0


def train_single_model(cnn, num_fold, train_index, test_index, files, additional_files, submission_version):
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    print('Creating and compiling model [{}]...'.format(cnn))
    # model = get_pretrained_model(CNN_TYPE, CLASSES_NUMBER, OPTIM_TYPE, LEARNING_RATE)
    model = get_pretrained_model(cnn, CLASSES_NUMBER)

    final_model_path = MODELS_PATH + '{}_subm_type_{}_fold_{}.h5'.format(cnn, submission_version, num_fold)
    if os.path.isfile(final_model_path) and (RESTORE_FROM_LAST_CHECKPOINT or 1):
        print('Model already created. Skip')
        return 0
    cache_model_path = MODELS_PATH + '{}_subm_type_{}_temp_fold_{}.h5'.format(cnn, submission_version, num_fold)
    if os.path.isfile(cache_model_path) and RESTORE_FROM_LAST_CHECKPOINT:
        print('Load model from last point: ', cache_model_path)
        model.load_weights(cache_model_path)
    else:
        print('Start training from begining')

    print('Fitting model...')
    train_images_list = []
    valid_images_list = []
    for i in train_index:
        train_images_list.append(files[i])
    for i in test_index:
        valid_images_list.append(files[i])

    batch_size = get_batch_size(cnn)
    print('Batch size: {}'.format(batch_size))

    samples_train_per_epoch = 1000
    samples_valid_per_epoch = 1000

    print('Samples train: {}, Samples valid: {}'.format(samples_train_per_epoch, samples_valid_per_epoch))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    list_of_add = []
    if submission_version == 0:
        list_of_add = additional_files

    history = model.fit_generator(generator=batch_generator_train(cnn, train_images_list, batch_size, list_of_add, True),
                  nb_epoch=NB_EPOCH,
                  samples_per_epoch=samples_train_per_epoch,
                  validation_data=batch_generator_train(cnn, valid_images_list, batch_size, [], True),
                  nb_val_samples=samples_valid_per_epoch,
                  verbose=2, max_q_size=1000,
                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}.csv'.format(cnn, num_fold, min_loss, get_learning_rate(cnn), now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    return min_loss


def run_cross_validation_create_models(cnn, nfolds, submission_version):
    from sklearn.cross_validation import KFold
    files = glob.glob(INPUT_PATH + "*/*.jpg")
    additional_files = glob.glob(INPUT_PATH_ADD + "*/*.jpg")
    kf = KFold(len(files), n_folds=nfolds, shuffle=True, random_state=get_random_state(cnn))
    num_fold = 0
    sum_score = 0
    print('Len of additional files: {}'.format(len(additional_files)))
    for train_index, test_index in kf:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(train_index))
        print('Split valid: ', len(test_index))

        score = train_single_model(cnn, num_fold, train_index, test_index, files, additional_files, submission_version)
        sum_score += score

    print('Avg loss: {}'.format(sum_score/nfolds))


def get_raw_predictions_for_images(cnn, model, files_to_process, augment_number_per_image):
    predictions_list = []
    batch_len = MAX_IMAGES_FOR_INFERENCE // augment_number_per_image
    current_position = 0
    while current_position < len(files_to_process):
        if current_position + batch_len < len(files_to_process):
            part_files = files_to_process[current_position:current_position + batch_len]
        else:
            part_files = files_to_process[current_position:]
        image_list = get_augmented_image_list(part_files, augment_number_per_image, get_input_shape(cnn), TABLE_WITH_RECTANGLES)
        print('Test shape: ', str(image_list.shape))
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_input_overall(cnn, image_list)
        predictions_list.append(model.predict(image_list, batch_size=32, verbose=2))
        current_position += batch_len
    predictions = np.concatenate(predictions_list)
    if len(predictions) != len(files_to_process) * augment_number_per_image:
        print('Some error here on augmentation!')
        exit()
    return predictions


def run_validation(cnn, nfolds, submission_version):
    from sklearn.cross_validation import KFold
    from sklearn.metrics import log_loss
    from keras.models import load_model

    read_rectangles_table()

    print('Start validation...')
    augment_number_per_image = 20

    true_labels = []
    files = glob.glob(INPUT_PATH + "*/*.jpg")
    for f in files:
        clss = int(os.path.basename(os.path.dirname(f)).split('_')[1])
        true_labels.append(clss-1)

    kf = KFold(len(files), n_folds=nfolds, shuffle=True, random_state=get_random_state(cnn))
    num_fold = 0
    full_predictions = [0]*len(files)
    for train_index, test_index in kf:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(train_index))
        print('Split valid: ', len(test_index))

        final_model_path = MODELS_PATH + '{}_subm_type_{}_fold_{}.h5'.format(cnn, submission_version, num_fold)
        if "DENSENET" not in cnn:
            model = load_model(final_model_path)
        else:
            model = get_pretrained_model(cnn, CLASSES_NUMBER)
            model.load_weights(final_model_path)

        total_image_to_process = len(test_index)*augment_number_per_image
        print('Total images to process: {}'.format(total_image_to_process))

        if num_fold < 4 and 0:
            print('READING FROM CACHE!')
            partial_pred = load_from_file("./cache/cache_valid_{}.pklz".format(num_fold))
            for i in range(len(test_index)):
                full_predictions[test_index[i]] = partial_pred[i]
            continue

        files_to_process = []
        for ti in test_index:
            files_to_process.append(files[ti])
        predictions = get_raw_predictions_for_images(cnn, model, files_to_process, augment_number_per_image)

        partial_true = [0] * len(test_index)
        partial_pred = [0] * len(test_index)
        for i in range(len(test_index)):
            pred_avg = np.zeros(len(predictions[i]))
            cur_index = i * augment_number_per_image
            for j in range(augment_number_per_image):
                pred_avg += np.array(predictions[cur_index + j])
            pred_avg /= augment_number_per_image
            full_predictions[test_index[i]] = pred_avg
            partial_true[i] = true_labels[test_index[i]]
            partial_pred[i] = pred_avg
            # print('Pred: {} True: {}'.format(predictions[i], true_labels[test_index[i]]))

        save_in_file(partial_pred, "./cache/cache_valid_{}.pklz".format(num_fold))
        score_fold = log_loss(partial_true, partial_pred)
        print('Loss fold {}: {}'.format(num_fold, score_fold))
        shutil.copy2(final_model_path, MODELS_PATH + '{}_fold_{}_score_{}.h5'.format(cnn, num_fold, score_fold))
        shutil.copy2(__file__, CODE_COPY_FOLDER + '{}_fold_{}_score_{}.py'.format(cnn, num_fold, score_fold))

    score = log_loss(true_labels, full_predictions)
    print('Loss: {}'.format(score))

    # Save to validation submission list
    valid_subm_file = OUTPUT_PATH + 'subm_{}_LS_{}_validation.csv'.format(cnn, score)
    out = open(valid_subm_file, "w")
    out.write("image_name,Type_1,Type_2,Type_3\n")
    for i in range(len(files)):
        pred_avg = full_predictions[i]
        bn = os.path.basename(files[i])
        bn = bn.replace(".png", ".jpg")
        out.write(bn + ',' + str(pred_avg[0]) + ',' + str(pred_avg[1]) + ',' + str(pred_avg[2]) + '\n')
    out.close()

    return score, valid_subm_file


def get_mean(pred):
    mn = np.zeros(len(pred[0]))
    for i in range(len(pred)):
        mn += np.array(pred[i])
    return mn/len(pred)


def get_best_model(cnn, num_fold, model_type, submission_version):
    from keras.models import load_model

    final_model_path = MODELS_PATH + '{}_subm_type_{}_fold_{}.h5'.format(cnn, submission_version, num_fold)
    if model_type == 1:
        # Use with best score
        files = glob.glob(MODELS_PATH + '{}_fold_{}_score_*.h5'.format(cnn, num_fold + 1))
        best_score = 1000000000
        best_model_path = final_model_path
        for f in files:
            scr = float(f.split("_score_")[1].split(".")[0])
            if scr < best_score:
                best_score = scr
                best_model_path = f
    else:
        # Use only latest
        best_model_path = MODELS_PATH + '{}_subm_type_{}_fold_{}.h5'.format(cnn, submission_version, num_fold)

    if "DENSENET" not in cnn:
        model = load_model(best_model_path)
    else:
        model = get_pretrained_model(cnn, CLASSES_NUMBER)
        model.load_weights(best_model_path)
    return model, best_model_path


def run_test(cnn, nfolds, subm_file, submission_version):
    start_time = time.time()
    print('Start test prediction...')
    augment_number_per_image = 40
    use_best_score_1_or_latest_0 = 0

    files = glob.glob(INPUT_PATH_TEST + "*.jpg")
    print('Number of test files: ', str(len(files)))
    predicts = np.zeros((len(files), nfolds, CLASSES_NUMBER))

    for num_fold in range(nfolds):
        print('Start Fold number {} from {}'.format(num_fold+1, nfolds))

        model, model_path = get_best_model(cnn, num_fold+1, use_best_score_1_or_latest_0, submission_version)
        print('Using model: {}'.format(model_path))

        total_image_to_process = len(files) * augment_number_per_image
        print('Total images to process: {}'.format(total_image_to_process))

        files_to_process = files.copy()
        predictions = get_raw_predictions_for_images(cnn, model, files_to_process, augment_number_per_image)

        for i in range(len(files)):
            pred_avg = np.zeros(len(predictions[i]))
            cur_index = i * augment_number_per_image
            for j in range(augment_number_per_image):
                pred_avg += np.array(predictions[cur_index + j])
            pred_avg /= augment_number_per_image
            predicts[i, num_fold, :] = pred_avg
        save_in_file(predicts[:, num_fold], "./cache/cache_test_{}.pklz".format(num_fold))


    # Write submission
    out = open(subm_file, "w")
    out.write("image_name,Type_1,Type_2,Type_3\n")
    for i in range(len(files)):
        pred_avg = get_mean(predicts[i])
        bn = os.path.basename(files[i])
        bn = bn.replace(".png", ".jpg")
        out.write(bn + ',' + str(pred_avg[0]) + ',' + str(pred_avg[1]) + ',' + str(pred_avg[2]) + '\n')

    out.close()
    # print('LB score: ', check_score(subm_file))
    print("Elapsed time for test: %s seconds" % (time.time() - start_time))


def read_rectangles_table():
    global TABLE_WITH_RECTANGLES
    TABLE_WITH_RECTANGLES = pd.read_csv("../modified_data/rectangles_merged.csv")
