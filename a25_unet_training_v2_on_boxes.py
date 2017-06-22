# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import platform
import datetime
from a00_common_functions import *
from a02_zf_unet_model import *
from sklearn.model_selection import KFold

random.seed(2016)
np.random.seed(2016)

MASK_THRESHOLD_POINT = 100
TEST_FILES_PATH = "../input/test/"
TRAIN_FILES_PATH = "../input/train/"
ADDITIONAL_FILES_PATH = "../input/additional/"
INPUT_PATH = "../modified_data/train_unet_v2/"
PREDICTION_STORAGE_PATH = "../modified_data/unet_processed_v2/"

TEST_RESULT_PATH = "../modified_data/cropped_test_v2/"
TRAIN_RESULT_PATH = "../modified_data/cropped_train_v2/"
ADDITIONAL_RESULT_PATH = "../modified_data/cropped_additional_v2/"
if not os.path.isdir(TEST_RESULT_PATH):
    os.mkdir(TEST_RESULT_PATH)
if not os.path.isdir(TRAIN_RESULT_PATH):
    os.mkdir(TRAIN_RESULT_PATH)
if not os.path.isdir(ADDITIONAL_RESULT_PATH):
    os.mkdir(ADDITIONAL_RESULT_PATH)

if not os.path.isdir(PREDICTION_STORAGE_PATH):
    os.mkdir(PREDICTION_STORAGE_PATH)
if not os.path.isdir(PREDICTION_STORAGE_PATH + 'test/'):
    os.mkdir(PREDICTION_STORAGE_PATH + 'test/')
if not os.path.isdir(PREDICTION_STORAGE_PATH + 'train/'):
    os.mkdir(PREDICTION_STORAGE_PATH + 'train/')
if not os.path.isdir(PREDICTION_STORAGE_PATH + 'additional/'):
    os.mkdir(PREDICTION_STORAGE_PATH + 'additional/')

MODELS_PATH = '../models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
HISTORY_FOLDER_PATH = "../models/history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)


def return_random_crop(img, mask):
    perc = 0.1
    start0 = random.randint(0, int(img.shape[0]*perc))
    end0 = img.shape[0] - random.randint(0, int(img.shape[0] * perc))
    start1 = random.randint(0, int(img.shape[1] * perc))
    end1 = img.shape[1] - random.randint(0, int(img.shape[1] * perc))
    # print(img.shape)
    # print(start0, end0, start1, end1)
    return img[start0:end0, start1:end1], mask[start0:end0, start1:end1]


def random_rotate(image, mask):
    cols = image.shape[1]
    rows = image.shape[0]
    mean_color = np.mean(image, axis=(0, 1))

    angle = random.uniform(0, 90)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderValue=mean_color)
    dst_msk = cv2.warpAffine(mask, M, (cols, rows))
    return dst, dst_msk


def lightning_change(image):
    intencity = 20
    image = image.astype(np.int16)
    for i in range(3):
        r = random.randint(-intencity, intencity)
        image[:, :, i][(255-r > image[:, :, i]) & (image[:, :, i] >= -r)] += r
    image = image.astype(np.uint8)
    return image


def random_augment_image_for_unet(image, mask):
    image, mask = random_rotate(image, mask)
    image, mask = return_random_crop(image, mask)

    # all possible mirroring and flips
    # (in total there are only 8 possible configurations out of 16)
    mirror = random.randint(0, 1)
    if mirror == 1:
        # flipud
        image = image[::-1, :, :]
        mask = mask[::-1, :]
    angle = random.randint(0, 3)
    if angle != 0:
        image = np.rot90(image, k=angle)
        mask = np.rot90(mask, k=angle)

    image = lightning_change(image)
    return image, mask


def batch_generator_train(files, batch_size, augment=False):

    batch_normal = batch_size
    random.shuffle(files)
    number_of_batches = np.floor(len(files) / batch_normal)
    counter = 0

    while True:
        batch_files = files[batch_normal * counter:batch_normal * (counter + 1)]
        image_list = []
        mask_list = []
        for f in batch_files:
            real_image_path = f[:-9] + '.png'
            image = cv2.imread(real_image_path)
            mask = cv2.imread(f, 0)
            # print(f)
            # print(f[:-4] + '_mask.png')
            # print(image.shape)
            # print(mask.shape)
            if augment:
                image, mask = random_augment_image_for_unet(image, mask)
            image = cv2.resize(image, (224, 224), cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (224, 224), cv2.INTER_LANCZOS4)
            # show_image(image)
            # show_image(mask)
            image_list.append(image.astype(np.float32))
            mask_list.append([mask])
        counter += 1
        image_list = np.array(image_list)
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_batch(image_list)
        mask_list = np.array(mask_list).astype(np.float32)
        mask_list /= 255.0
        # print(image_list.shape)
        # print(mask_list.shape)
        yield image_list, mask_list
        if counter >= number_of_batches:
            random.shuffle(files)
            counter = 0


def train_single_model(num_fold, train_index, test_index, files):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
    from keras.optimizers import Adam, SGD

    print('Creating and compiling UNET...')
    restore = 0
    patience = 20
    epochs = 200
    optim_type = 'Adam'
    learning_rate = 0.0005
    cnn_type = 'UNET_V2'
    model = ZF_UNET_224()
    model.load_weights("../weights/zf_unet_224.h5")

    final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
    if os.path.isfile(final_model_path):
        print('Model already exists for fold {}. Skipping !!!!!!!!!!'.format(final_model_path))
        return 0.0

    cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, num_fold)
    if os.path.isfile(cache_model_path) and restore:
        print('Load model from last point: ', cache_model_path)
        model.load_weights(cache_model_path)
    else:
        print('Start training from begining')

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

    print('Fitting model...')
    train_images_list = []
    valid_images_list = []
    for i in train_index:
        train_images_list.append(files[i])
    for i in test_index:
        valid_images_list.append(files[i])

    batch_size = 16
    print('Batch size: {}'.format(batch_size))
    # print('Learning rate: {} Optimizer: {}'.format(LEARNING_RATE, OPTIM_TYPE))
    samples_train_per_epoch = batch_size * 80
    # samples_valid_per_epoch = (batch_size + 1) * (len(test_index) // batch_size)
    samples_valid_per_epoch = batch_size * 80
    print('Samples train: {}, Samples valid: {}'.format(samples_train_per_epoch, samples_valid_per_epoch))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(generator=batch_generator_train(train_images_list, batch_size, True),
                  nb_epoch=epochs,
                  samples_per_epoch=samples_train_per_epoch,
                  validation_data=batch_generator_train(valid_images_list, batch_size, True),
                  nb_val_samples=samples_valid_per_epoch,
                  verbose=2, max_q_size=1000,
                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}.csv'.format(cnn_type, num_fold, min_loss, learning_rate, now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    return min_loss


def run_cross_validation_create_models_unet2(nfolds=5):
    from sklearn.model_selection import KFold
    files_full = glob.glob(INPUT_PATH + "*/*.png")
    files = []
    for f in files_full:
        if '_mask' not in f:
            continue
        files.append(f)

    kf = KFold(n_splits=nfolds, shuffle=True, random_state=66)
    num_fold = 0
    sum_score = 0
    for train_index, test_index in kf.split(range(len(files))):
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(train_index))
        print('Split valid: ', len(test_index))
        if num_fold != 2:
            continue
        score = train_single_model(num_fold, train_index, test_index, files)
        sum_score += score

    print('Avg loss: {}'.format(sum_score/nfolds))


def predict_for_train_images(num_fold, test_index, files):
    cnn_type = 'UNET_V2'
    model = ZF_UNET_224()
    final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
    model.load_weights(final_model_path)
    print('Fitting model...')

    file_list = []
    image_list = []
    for i in test_index:
        storage_path = PREDICTION_STORAGE_PATH + 'train/' + os.path.basename(os.path.dirname(files[i])) + '/'
        if not os.path.isdir(storage_path):
            os.mkdir(storage_path)
        image = cv2.imread(files[i])
        image = cv2.resize(image, (224, 224), cv2.INTER_LANCZOS4)
        cv2.imwrite(storage_path + os.path.basename(files[i]), image)
        image_list.append(image.astype(np.float32))
        file_list.append(files[i])
    image_list = np.array(image_list)
    image_list = image_list.transpose((0, 3, 1, 2))
    image_list = preprocess_batch(image_list)
    predictions = model.predict(image_list)
    print(predictions.shape)
    for i in range(predictions.shape[0]):
        storage_path = PREDICTION_STORAGE_PATH + 'train/' + os.path.basename(os.path.dirname(file_list[i])) + '/'
        np.save(storage_path + os.path.basename(file_list[i])[:-4] + '.npy', predictions[i])
        arr = np.round(predictions[i][0]*255).astype(np.uint8)
        cv2.imwrite(storage_path + os.path.basename(file_list[i])[:-4] + '_mask.png', arr)


def predict_on_train_and_additional_images_unet2(nfolds=5):
    cnn_type = 'UNET_V2'

    if 1:
        # Predict on images which was used for training (exclude possible leakage)
        files_full = glob.glob(INPUT_PATH + "*/*.png")
        files = []
        for f in files_full:
            if '_mask' in f:
                continue
            files.append(f)

        kf = KFold(n_splits=nfolds, shuffle=True, random_state=66)
        num_fold = 0
        for train_index, test_index in kf.split(range(len(files))):
            num_fold += 1
            print('Start KFold number {} from {}'.format(num_fold, nfolds))
            print('Split valid: ', len(test_index))
            predict_for_train_images(num_fold, test_index, files)

        # Predict on other train images
        all_files = glob.glob(TRAIN_FILES_PATH + "*/*.jpg")
        files = []
        for i in range(len(all_files)):
            storage_path = PREDICTION_STORAGE_PATH + 'train/' + os.path.basename(os.path.dirname(all_files[i])) + '/'
            if not os.path.isdir(storage_path):
                os.mkdir(storage_path)
            npy_path = storage_path + os.path.basename(all_files[i][:-4]) + '.npy'
            if not os.path.isfile(npy_path):
                files.append(all_files[i])

        print('All files: ', len(all_files))
        print('Not predicted files: ', len(files))

    if 1:
        image_list = []
        for i in range(len(files)):
            storage_path = PREDICTION_STORAGE_PATH + 'train/' + os.path.basename(os.path.dirname(files[i])) + '/'
            image = cv2.imread(files[i])
            image = cv2.resize(image, (224, 224), cv2.INTER_LANCZOS4)
            cv2.imwrite(storage_path + os.path.basename(files[i]), image)
            image_list.append(image.astype(np.float32))
        image_list = np.array(image_list)
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_batch(image_list)

        predictions = []
        for num_fold in range(1, nfolds + 1):

            model = ZF_UNET_224()
            final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
            print('Load {}'.format(final_model_path))
            model.load_weights(final_model_path)
            predictions.append(model.predict(image_list))
        predictions = np.mean(np.array(predictions), axis=0)
        print(predictions.shape)
        for i in range(predictions.shape[0]):
            storage_path = PREDICTION_STORAGE_PATH + 'train/' + os.path.basename(os.path.dirname(files[i])) + '/'
            np.save(storage_path + os.path.basename(files[i])[:-4] + '.npy', predictions[i])
            arr = np.round(predictions[i][0] * 255).astype(np.uint8)
            cv2.imwrite(storage_path + os.path.basename(files[i])[:-4] + '_mask.png', arr)

    if 1:
        # Predict on other additional images
        all_files = glob.glob(ADDITIONAL_FILES_PATH + "*/*.jpg")
        files = []
        for i in range(len(all_files)):
            storage_path = PREDICTION_STORAGE_PATH + 'additional/' + os.path.basename(os.path.dirname(all_files[i])) + '/'
            if not os.path.isdir(storage_path):
                os.mkdir(storage_path)
            npy_path = storage_path + os.path.basename(all_files[i][:-4]) + '.png.npy'
            if not os.path.isfile(npy_path):
                files.append(all_files[i])

        print('All additional files: ', len(all_files))
        print('Not predicted additional files: ', len(files))

        image_list = []
        for i in range(len(files)):
            storage_path = PREDICTION_STORAGE_PATH + 'additional/' + os.path.basename(os.path.dirname(files[i])) + '/'
            image = cv2.imread(files[i])
            image = cv2.resize(image, (224, 224), cv2.INTER_LANCZOS4)
            cv2.imwrite(storage_path + os.path.basename(files[i]), image)
            image_list.append(image.astype(np.float32))
        image_list = np.array(image_list)
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_batch(image_list)

        predictions = []
        for num_fold in range(1, nfolds + 1):
            model = ZF_UNET_224()
            final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
            print('Load {}'.format(final_model_path))
            model.load_weights(final_model_path)
            predictions.append(model.predict(image_list))
        predictions = np.mean(np.array(predictions), axis=0)
        print(predictions.shape)
        for i in range(predictions.shape[0]):
            storage_path = PREDICTION_STORAGE_PATH + 'additional/' + os.path.basename(os.path.dirname(files[i])) + '/'
            np.save(storage_path + os.path.basename(files[i])[:-4] + '.npy', predictions[i])
            arr = np.round(predictions[i][0] * 255).astype(np.uint8)
            cv2.imwrite(storage_path + os.path.basename(files[i])[:-4] + '_mask.png', arr)


def predict_on_test_images_unet2(nfolds):
    cnn_type = 'UNET_V2'
    # Predict test
    test_files = glob.glob(TEST_FILES_PATH + "/*.jpg")
    image_list = []
    for i in range(len(test_files)):
        image = cv2.imread(test_files[i])
        image = cv2.resize(image, (224, 224), cv2.INTER_LANCZOS4)
        cv2.imwrite(PREDICTION_STORAGE_PATH + 'test/' + os.path.basename(test_files[i]), image)
        image_list.append(image.astype(np.float32))
    image_list = np.array(image_list)
    image_list = image_list.transpose((0, 3, 1, 2))
    image_list = preprocess_batch(image_list)

    predictions = []
    for num_fold in range(1, nfolds + 1):
        model = ZF_UNET_224()
        final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
        print('Load {}'.format(final_model_path))
        model.load_weights(final_model_path)
        predictions.append(model.predict(image_list))
    predictions = np.mean(np.array(predictions), axis=0)
    print(predictions.shape)
    for i in range(predictions.shape[0]):
        np.save(PREDICTION_STORAGE_PATH + 'test/' + os.path.basename(test_files[i])[:-4] + '.npy', predictions[i])
        arr = np.round(predictions[i][0] * 255).astype(np.uint8)
        cv2.imwrite(PREDICTION_STORAGE_PATH + 'test/' + os.path.basename(test_files[i])[:-4] + '_mask.png', arr)



def save_cropped_image(orig_path, mask_path, out_path):
    prediction = np.load(mask_path)[0]
    mask = np.round(prediction * 255).astype(np.uint8)
    ret, thresh = cv2.threshold(mask, MASK_THRESHOLD_POINT, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    if len(contours) == 0:
        # Fallback on first version
        print('Fallback on version 1', orig_path, mask_path, out_path)
        mask_path = mask_path.replace("_v2", "")
        if not os.path.isfile(mask_path):
            mask_path = mask_path.replace(".npy", ".jpg.npy")
        prediction = np.load(mask_path)[0]
        mask = np.round(prediction * 255).astype(np.uint8)
        ret, thresh = cv2.threshold(mask, MASK_THRESHOLD_POINT, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    if len(contours) == 0:
        # Fallback on first version
        print('Still error here!!!', orig_path, mask_path, out_path)
        exit()
    if len(contours) > 1:
        max_area = -1
        pred_max_area = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            # print('Area',  area)
            if area > max_area:
                cnt = contours[i]
                pred_max_area = max_area
                max_area = area

        if pred_max_area > 0:
            if max_area / pred_max_area < 3:
                print('Critical contours! ', orig_path)
                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    print('Area', area)
    else:
        cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    big_image = cv2.imread(orig_path)
    sh0_start = round(big_image.shape[0] * y / 224)
    sh0_end = round(big_image.shape[0] * (y + h) / 224)
    sh1_start = round(big_image.shape[1] * x / 224)
    sh1_end = round(big_image.shape[1] * (x + w) / 224)
    sub_img = big_image[sh0_start:sh0_end, sh1_start:sh1_end, :]
    # show_resized_image(big_image)
    # show_resized_image(sub_img)
    sub_img = cv2.resize(sub_img, (800, 800), cv2.INTER_LANCZOS4)
    cv2.imwrite(out_path, sub_img)
    reduction = 1 - w*h/(224*224)
    # print('Reduction of image area: {}'.format(reduction))
    return reduction


def create_crops_from_predictions_unet2():
    if 0:
        # Test part
        avg_reduction = 0.0
        files = glob.glob(TEST_FILES_PATH + "/*.jpg")
        for i in range(len(files)):
            orig_path = files[i]
            mask_path = PREDICTION_STORAGE_PATH + 'test/' + os.path.basename(files[i])[:-4] + '.npy'
            if not os.path.isfile(mask_path):
                print('Some error', mask_path)
                exit()
            out_path = TEST_RESULT_PATH + os.path.basename(files[i])[:-4] + '.png'
            red = save_cropped_image(orig_path, mask_path, out_path)
            avg_reduction += red
        avg_reduction /= len(files)
        print('Average reduction for test: {}'.format(avg_reduction))

    # Train part
    if 0:
        avg_reduction = 0.0
        files = glob.glob(TRAIN_FILES_PATH + "*/*.jpg")
        for i in range(len(files)):
            orig_path = files[i]
            d1 = os.path.basename(os.path.dirname(files[i])) + '/'
            mask_path = PREDICTION_STORAGE_PATH + 'train/' + d1 + os.path.basename(files[i])[:-4] + '.npy'
            if not os.path.isfile(mask_path):
                print('Some error', mask_path)
                exit()
            if not os.path.isdir(TRAIN_RESULT_PATH + d1):
                os.mkdir(TRAIN_RESULT_PATH + d1)
            out_path = TRAIN_RESULT_PATH + d1 + os.path.basename(files[i])[:-4] + '.png'
            red = save_cropped_image(orig_path, mask_path, out_path)
            avg_reduction += red
        avg_reduction /= len(files)
        print('Average reduction for train: {}'.format(avg_reduction))


    # Additional part
    avg_reduction = 0.0
    files = glob.glob(ADDITIONAL_FILES_PATH + "*/*.jpg")
    for i in range(len(files)):
        orig_path = files[i]
        d1 = os.path.basename(os.path.dirname(files[i])) + '/'
        mask_path = PREDICTION_STORAGE_PATH + 'additional/' + d1 + os.path.basename(files[i])[:-4] + '.npy'
        if not os.path.isfile(mask_path):
            print('Some error', mask_path)
            exit()
        if not os.path.isdir(ADDITIONAL_RESULT_PATH + d1):
            os.mkdir(ADDITIONAL_RESULT_PATH + d1)
        out_path = ADDITIONAL_RESULT_PATH + d1 + os.path.basename(files[i])[:-4] + '.png'
        red = save_cropped_image(orig_path, mask_path, out_path)
        avg_reduction += red
    avg_reduction /= len(files)
    print('Average reduction for additional: {}'.format(avg_reduction))



def get_image_rect_sizes(orig_path, mask_path):
    big_image = cv2.imread(orig_path)
    prediction = np.load(mask_path)[0]
    mask = np.round(prediction * 255).astype(np.uint8)
    ret, thresh = cv2.threshold(mask, MASK_THRESHOLD_POINT, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    if len(contours) == 0:
        return big_image.shape[0], big_image.shape[1], -1, -1, -1, -1

    cnt = contours[0]
    if len(contours) > 1:
        max_area = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                cnt = contours[i]
                max_area = area

    x, y, w, h = cv2.boundingRect(cnt)
    sh0_start = round(big_image.shape[0] * y / 224)
    sh0_end = round(big_image.shape[0] * (y + h) / 224)
    sh1_start = round(big_image.shape[1] * x / 224)
    sh1_end = round(big_image.shape[1] * (x + w) / 224)
    return big_image.shape[0], big_image.shape[1], sh0_start, sh0_end, sh1_start, sh1_end


def create_csv_file_from_predictions_unet2():
    out = open("../modified_data/rectangles_from_unet_v2.csv", "w")
    out.write("id,type,img_shape0,img_shape1,rect_v2_start0,rect_v2_end0,rect_v2_start1,rect_v2_end1\n")

    # Test part
    if 1:
        print('Go for test part...')
        files = glob.glob(TEST_FILES_PATH + "/*.jpg")
        for i in range(len(files)):
            orig_path = files[i]
            mask_path = PREDICTION_STORAGE_PATH + 'test/' + os.path.basename(files[i])[:-4] + '.npy'
            if not os.path.isfile(mask_path):
                print('Some error', mask_path)
                exit()
            img_shape0, imag_shape1, rect_start0, rect_end0, rect_start1, rect_end1 = get_image_rect_sizes(orig_path, mask_path)
            out.write(os.path.basename(files[i])[:-4] + '.jpg')
            out.write(',' + 'test')
            out.write(',' + str(img_shape0))
            out.write(',' + str(imag_shape1))
            out.write(',' + str(rect_start0))
            out.write(',' + str(rect_end0))
            out.write(',' + str(rect_start1))
            out.write(',' + str(rect_end1))
            out.write('\n')

    # Train part
    if 1:
        print('Go for train part...')
        files = glob.glob(TRAIN_FILES_PATH + "*/*.jpg")
        for i in range(len(files)):
            orig_path = files[i]
            d1 = os.path.basename(os.path.dirname(files[i])) + '/'
            mask_path = PREDICTION_STORAGE_PATH + 'train/' + d1 + os.path.basename(files[i])[:-4] + '.npy'
            if not os.path.isfile(mask_path):
                print('Some error', mask_path)
                exit()
            if not os.path.isdir(TRAIN_RESULT_PATH + d1):
                os.mkdir(TRAIN_RESULT_PATH + d1)
            img_shape0, imag_shape1, rect_start0, rect_end0, rect_start1, rect_end1 = get_image_rect_sizes(orig_path,
                                                                                                           mask_path)
            out.write(os.path.basename(files[i])[:-4] + '.jpg')
            out.write(',' + 'train')
            out.write(',' + str(img_shape0))
            out.write(',' + str(imag_shape1))
            out.write(',' + str(rect_start0))
            out.write(',' + str(rect_end0))
            out.write(',' + str(rect_start1))
            out.write(',' + str(rect_end1))
            out.write('\n')

    # Additional part
    if 1:
        print('Go for additional part...')
        files = glob.glob(ADDITIONAL_FILES_PATH + "*/*.jpg")
        for i in range(len(files)):
            orig_path = files[i]
            d1 = os.path.basename(os.path.dirname(files[i])) + '/'
            mask_path = PREDICTION_STORAGE_PATH + 'additional/' + d1 + os.path.basename(files[i])[:-4] + '.npy'
            if not os.path.isfile(mask_path):
                print('Some error', mask_path)
                exit()
            if not os.path.isdir(ADDITIONAL_RESULT_PATH + d1):
                os.mkdir(ADDITIONAL_RESULT_PATH + d1)
            img_shape0, imag_shape1, rect_start0, rect_end0, rect_start1, rect_end1 = get_image_rect_sizes(orig_path,
                                                                                                           mask_path)
            out.write(os.path.basename(files[i])[:-4] + '.jpg')
            out.write(',' + 'add')
            out.write(',' + str(img_shape0))
            out.write(',' + str(imag_shape1))
            out.write(',' + str(rect_start0))
            out.write(',' + str(rect_end0))
            out.write(',' + str(rect_start1))
            out.write(',' + str(rect_end1))
            out.write('\n')

    out.close()

# Avg reduction ~0.80 of area