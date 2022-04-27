
from __future__ import print_function

import os

from constants import *

import pandas as pd 
from data import preprocessor, load_val_data
from unet import get_unet_ds, get_unet, dice_coef
from utils import EnsembleDenseCRF, unnormalize
from denseCRF import average_arbitrator
import numpy as np
import cv2
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_test_data(data_path, masks_path):
    """
    Generate training data numpy arrays and save them into the project path
    """
    image_rows = 420
    image_cols = 580

    images = os.listdir(data_path)
    total = len(images)
    if os.path.exists(global_path + 'imgs_test.npy') and os.path.exists(global_path + 'imgs_mask_test.npy'):
        return images

    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    i = 0
    file_list = []
    for image_name in images:
        print(i)
        img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
        img = np.array([img])
        img = np.rollaxis(img, 0,3)
        imgs[i] = img
        img_mask = cv2.imread(os.path.join(masks_path, image_name.replace('.jpg', '_segmentation.jpg')), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
        img_mask = np.array([img_mask])
        img_mask = np.rollaxis(img_mask, 0,3)
        imgs_mask[i] = img_mask
        i += 1
        file_list.append(image_name)

    np.save(global_path + 'imgs_test.npy', imgs)
    np.save(global_path + 'imgs_mask_test.npy', imgs_mask)
    # print(file_list)
    return file_list


def saveResult_prediction_ds(save_path, file_List, npyfile):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for j in range(len(npyfile)):
        for i, item in enumerate(npyfile[j]):
            img = item[..., 0]
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            img = cv2.resize(img, (img_cols, img_rows))
            cv2.imwrite(os.path.join(save_path, file_List[i].replace('.jpg', f'_out{j}.jpg')), img * 255)


def saveResult_prediction(save_path, file_List, npyfile, gt_file=None, ori_file=None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i, (item, filename) in enumerate(zip(npyfile, file_List)):
        img = item[..., 0]
        if ori_file is not None:
            ori_img = unnormalize(ori_file[i])
            cv2.imwrite(os.path.join(save_path, filename.replace('.jpg', '_ori.jpg')), ori_img)
            if False and post_process :
                crf = EnsembleDenseCRF(MCRF_cfgs)
                img = crf(ori_img, img, output_bases=False)
        cv2.imwrite(os.path.join(save_path, filename.replace('.jpg', '_out.jpg')), img * 255.)
        if gt_file is not None:
            cv2.imwrite(os.path.join(save_path, filename.replace('.jpg', '_gt.jpg')), gt_file[i] * 255.)



def load_test_data(data_path, masks_path):
    """
    Load training data from project path
    :return: [X_test, y_test] numpy arrays containing the training data and their respective masks.
    """
    print("\nLoading test data...\n")
    X_test = np.load(data_path)
    y_test = np.load(masks_path)

    X_test = preprocessor(X_test)
    y_test = preprocessor(y_test)

    X_test = X_test.astype('float32')

    mean = np.mean(X_test)  # mean for data centering
    std = np.std(X_test)  # std for data normalization

    X_test -= mean
    X_test /= std

    y_test = y_test.astype('float32')
    y_test /= 255.  # scale masks to [0, 1]
    return X_test, y_test


def predict(data, model, iteration, labels=None):
    if uncertain_type == "":
        if iteration == 0:
            weights_path = global_path + f'{exp}/[initial_weights_name].hdf5'
        else:
            weights_path = global_path + f"{exp}/active_model" + str(iteration) + ".h5"
            print(weights_path)
        model.load_weights(weights_path)

        if "ds" in exp:
            prediction = model.predict(data, verbose=0)[0]
        else:
            prediction = model.predict(data, verbose=0)
    else:
        predictions = []
        dice2p = OrderedDict()
        for t in range(T):
            if iteration == 0:
                weights_path = global_path + f"{exp}/[initial_weights_name].hdf5"
            else:
                weights_path = global_path + f"{exp}/active_model{iteration}_{t}.h5"
            model.load_weights(weights_path)
            print("Predicting using", weights_path)
            if "ds" in exp:
                prediction = model.predict(data, verbose=0)[0]
            else:
                prediction = model.predict(data, verbose=0)
            if labels is not None:
                dice2p[float(dice_coef(labels, prediction).numpy())] = np.array(prediction >= .5).astype(prediction.dtype)
        print("Base model dices:", list(dice2p.keys()))
        prediction = average_arbitrator(list(dice2p.values()))
    return prediction


def get_test_dc(scores):
    if "ds" not in exp:
        if USE_BCE:
            return scores[4]
    return scores[1]


def model_test(file_list, model_func=get_unet):
    test_data, label_data = load_val_data("test")
    print(test_data.shape)
    df = pd.DataFrame(np.arange(1, nb_iterations+1), columns=['index'])
    df['meanDC'] = ''
    for iteration in range(0, nb_iterations+1):
        if iteration == 0:
            df.loc[iteration, ['index']] = 'init'
            weights_path = global_path + f'{exp}/[initial_weights_name].hdf5'
        else:
            weights_path = global_path + f"{exp}/active_model" + str(iteration) + ".h5"
        model = model_func(False)
        model.load_weights(weights_path)
        predictions = predict(test_data, model, iteration, label_data)

        target = [label_data] * 3 if "ds" in exp else label_data
        scores = model.evaluate(test_data, target, batch_size=batch_size, verbose=0)
        df.loc[iteration, ['meanDC']] = get_test_dc(scores)
        print(iteration, 'DC', df.loc[iteration, ['meanDC']])
        saveResult_prediction(global_path + f'{exp}_prediction/'+ str(iteration)+ '/', file_list[nb_train:],
                              predictions, label_data, test_data)
        del predictions

    df.to_csv(global_path + f'{exp}_DCMean_dsal.csv')

file_list = os.listdir(data_path)
if not os.path.exists(global_path + f"{exp}_prediction"):
    os.makedirs(global_path + f"{exp}_prediction")
    print("Path created: ", global_path + f"{exp}_prediction")
else:
    print("Testing for", exp)


model_test(file_list, get_unet_ds if "ds" in exp else get_unet)
