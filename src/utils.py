from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
from denseCRF import EnsembleDenseCRF
from constants import *
from collections import OrderedDict
import pandas as pd
import heapq
from data import unnormalize
import time
from datetime import datetime


def range_transform(sample):
    """
    Range normalization for 255 range of values
    :param sample: numpy array for normalize
    :return: normalize numpy array
    """
    if np.max(sample) == 1:
        sample = sample * 255

    m = 255 / (np.max(sample) - np.min(sample))
    n = 255 - m * np.max(sample)
    return (m * sample + n) / 255


def predict(data, model):
    """
    Data prediction for a given model
    :param data: input data to predict.
    :param model: unet model.
    :return: predictions.
    """
    return model.predict(data, verbose=0)


def print_and_log(string: str, logfile: str):
    print(string)
    log(string, 0, logfile)


def compute_uncertain(sample, prediction, model, pred_func=predict):
    """
    Computes uncertainty map for a given sample and its prediction for a given model, based on the
    number of step predictions defined in constants file.
    :param sample: input sample.
    :param prediction: input sample prediction.
    :param model: unet model with Dropout layers.
    :return: uncertainty map.
    """
    X = np.zeros([1, img_rows, img_cols])

    for t in range(nb_step_predictions):
        prediction = pred_func(sample, model).reshape([1, img_rows, img_cols])
        X = np.concatenate((X, prediction))

    X = np.delete(X, [0], 0)

    if apply_edt:
        # apply distance transform normalization.
        var = np.var(X, axis=0)
        transform = range_transform(edt(prediction))
        return np.sum(var * transform)

    else:
        return np.sum(np.var(X, axis=0))


def interval(data, start, end):
    """
    Returns the index of data within range values from start to end.
    :param data: numpy array of data.
    :param start: starting value.
    :param end: ending value.
    :return: numpy array of data index.
    """
    p = np.where(data >= start)[0]
    return p[np.where(data[p] < end)[0]]


def get_pseudo_index(uncertain, nb_pseudo):
    """
    Gives the index of the most certain data, to make the pseudo annotations.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_pseudo: Total of pseudo samples.
    :return: Numpy array of index.
    """
    h = np.histogram(uncertain, 80)

    pseudo = interval(uncertain, h[1][np.argmax(h[0])], h[1][np.argmax(h[0]) + 1])
    np.random.shuffle(pseudo)
    return pseudo[0:nb_pseudo]


def random_index(uncertain, nb_random):
    """
    Gives the index of the random selection to be manually annotated.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_random: Total of random samples.
    :return: Numpy array of index.
    """
    histo = np.histogram(uncertain, 80)
    # TODO: automatic selection of random range
    index = interval(uncertain, histo[1][np.argmax(histo[0]) + 6], histo[1][len(histo[0]) - 33])
    np.random.shuffle(index)
    return index[0:nb_random]


def no_detections_index(uncertain, nb_no_detections):
    """
    Gives the index of the no detected samples to be manually annotated.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_no_detections: Total of no detected samples.
    :return: Numpy array of index.
    """
    return np.argsort(uncertain)[0:nb_no_detections]


def compute_dice_coef_ds(y_pred1,y_pred2,y_pred3):
    """
    Computes the Dice-Coefficient of a prediction given its ground truth.
    :param y_true: Ground truth.
    :param y_pred: Prediction.
    :return: Dice-Coefficient value.
    """
    dc1=compute_dice_coef(y_pred1, y_pred2)
    dc2=compute_dice_coef(y_pred1, y_pred3)
    return (dc1+dc2)/2


def most_uncertain_index(uncertain, nb_most_uncertain, rate):
    """
     Gives the index of the most uncertain samples to be manually annotated.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_most_uncertain: Total of most uncertain samples.
    :param rate: Hash threshold to define the most uncertain area. Bin of uncertainty histogram.
    TODO: automatic selection of rate.
    :return: Numpy array of index.
    """
    data = np.array([]).astype('int')

    histo = np.histogram(uncertain, 80)

    p = np.arange(len(histo[0]) - rate, len(histo[0]))  # index of last bins above the rate
    pr = np.argsort(histo[0][p])  # p index ascendant sorted
    cnt = 0
    pos = 0
    index = np.array([]).astype('int')

    while cnt < nb_most_uncertain and pos < len(pr):
        sbin = histo[0][p[pr[pos]]]

        index = np.append(index, p[pr[pos]])
        cnt = cnt + sbin
        pos = pos + 1

    for i in range(0, pos):
        data = np.concatenate((data, interval(uncertain, histo[1][index[i]], histo[1][index[i] + 1])))

    np.random.shuffle(data)
    return data[0:nb_most_uncertain]


def get_oracle_index(uncertain, nb_no_detections, nb_random, nb_most_uncertain, rate):
    """
    Gives the index of the unlabeled data to annotated at specific DSAL iteration, based on their uncertainty.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_no_detections: Total of no detected samples.
    :param nb_random: Total of random samples.
    :param nb_most_uncertain: Total of most uncertain samples.
    :param rate: Hash threshold to define the most uncertain area. Bin of uncertainty histogram.
    :return: Numpy array of index.
    """
    return np.concatenate((no_detections_index(uncertain, nb_no_detections),
                           random_index(uncertain, nb_random),
                           most_uncertain_index(uncertain, nb_most_uncertain, rate)))


def compute_dice_coef(y_true, y_pred):
    """
    Computes the Dice-Coefficient of a prediction given its ground truth.
    :param y_true: Ground truth.
    :param y_pred: Prediction.
    :return: Dice-Coefficient value.
    """
    smooth = 1.  # smoothing value to deal zero denominators.
    # y_pred = y_pred[:, :,np.newaxis]
    y_true_f = y_true.reshape([-1, 1])
    y_pred_f = y_pred.reshape([-1, 1])
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def shannon_entropy(y_preds):
    """
    Shannon Entropy for 2d binary prediction mask
    :param y_preds: a list of 2d prediction masks from different models [tensor([H, W, 1])]
    :return: Shannon Entropy score
    """
    avged = np.mean(np.stack(y_preds), axis=0).reshape([img_rows, img_cols])
    onehot = np.zeros((img_rows, img_cols, 2), dtype=avged.dtype)
    onehot[..., 0] = avged
    onehot[..., 1] = 1 - avged
    pre_log = onehot.copy()
    pre_log[pre_log == 0] = 1e-10
    pixel_wise_entropy = np.nansum(onehot * np.log(pre_log), axis=2)
    return - np.nansum(pixel_wise_entropy)


def jenson_shannon_entropy(y_preds):
    """
    Jenson-Shannon Entropy for 2d binary prediction mask
    :param y_preds: a list of 2d prediction masks from different models [tensor([H, W, 1])]
    :return: Shannon Entropy score
    """
    return shannon_entropy(y_preds) - np.nanmean([shannon_entropy([y_pred]) for y_pred in y_preds])


def mean_confidence_uncertainty(y_pred):
    # y_pred: [H, W, 1]
    return np.mean(np.abs(y_pred - .5))


def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError(f"Shape mismatch: im1 and im2 must have the same shape, "
                         f"but got im1.shape: {im1.shape} adn im2.shape: {im2.shape}")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


entropies = {
    "ensemble_S": shannon_entropy,
    "ensemble_JS": jenson_shannon_entropy
}


def compute_train_sets_ds(X_train, y_train, labeled_index, unlabeled_index, weights_path, iteration, nb_next_sample,
                          logfile, nb_pseudo=0, crf_configs=None, pre_ensemble=False):
    """
    Performs the DSAL labeling step, giving the available training data for each iteration.
    :param X_train: Overall training data.
    :param y_train: Overall training labels. Including the unlabeled samples to simulate the oracle annotations.
    :param labeled_index: Indices of labeled samples.
    :param unlabeled_index: Indices of unlabeled samples.
    :param weights: pre-trained unet weights.
    :param iteration: Currently DSAL iteration.
    :param nb_pseudo: number of unlabeled samples selected to calculate pseudo labels for next round

    :return: X_labeled_train: Update of labeled training data, adding the manual and pseudo annotations.
    :return: y_labeled_train: Update of labeled training labels, adding the manual and pseudo annotations.
    :return: labeled_index: Update of labeled indices, adding the manual annotations.
    :return: unlabeled_index: Update of labeled indices, removing the manual annotations.

    """
    from unet import get_unet_ds
    print_and_log("\nActive iteration " + str(iteration), logfile)
    print("-" * 50 + "\n")

    # predictions
    modelPredictions = get_unet_ds(dropout=False)
    modelPredictions.load_weights(weights_path)
    print_and_log("Computing log predictions ...\n", logfile)
    predictions = modelPredictions.predict(X_train[unlabeled_index], verbose=0)

    df = pd.DataFrame(unlabeled_index, columns=['unlabeled_index'])
    df['meanIntersec'], df['R-DSC'], df['M-DSC'], df['L-DSC'] = '', '', '', ''
    print_and_log("Computing train sets ...", logfile)
    for index in range(0, len(unlabeled_index)):
        if index % 100 == 0:
            print_and_log("completed: " + str(index) + "/" + str(len(unlabeled_index)), logfile)
        sample_prediction1 = cv2.threshold(predictions[0][index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
        sample_prediction2 = cv2.threshold(predictions[1][index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
        sample_prediction3 = cv2.threshold(predictions[2][index], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
        if index % 10 == 0:
            cv2.imwrite(f"{global_path}{exp}_prediction/{unlabeled_index[index]}_sample_prediction1.jpg", sample_prediction1 * 255)
            cv2.imwrite(f"{global_path}{exp}_prediction/{unlabeled_index[index]}_sample_prediction2.jpg", sample_prediction2 * 255)
            cv2.imwrite(f"{global_path}{exp}_prediction/{unlabeled_index[index]}_sample_prediction3.jpg", sample_prediction3 * 255)
        df.loc[index, ['meanIntersec']] = compute_dice_coef_ds(sample_prediction1,sample_prediction2, sample_prediction3)
        df.loc[index, ["R-DSC"]] = compute_dice_coef(y_train[unlabeled_index[index]], sample_prediction1)
        df.loc[index, ["M-DSC"]] = compute_dice_coef(sample_prediction1, sample_prediction2)
        df.loc[index, ["L-DSC"]] = compute_dice_coef(sample_prediction1, sample_prediction3)

    df.to_csv(global_path + f'{exp}_ranks/maskMeans_act'+str(iteration) + '.csv')
    sort_df_des = df.sort_values('meanIntersec', ascending=DS_ASCEND)
    sort_df_des.to_csv(global_path + f'{exp}_ranks/maskMeans_act'+str(iteration) + '_sort.csv')
    if random_inc:
        print_and_log("randomly select oracle indices", logfile)
        transIndex = list(set(np.random.choice(unlabeled_index, nb_next_sample)))
    else:
        transIndex = sort_df_des['unlabeled_index'][0:nb_next_sample]

    pseudo_rank = []
    # Post processing of UNet raw prediction using DenseCRF
    if nb_pseudo > 0 and iteration >= pseudo_epoch:
        if random_pseudo:
            pseudo_rank = list(set(np.random.choice(list(set(unlabeled_index) - set(transIndex)),
                                                    min(nb_pseudo, len(df) - nb_next_sample))))
        elif CONF_THRES:
            unlabeled_confidences = list(map(lambda x: mean_confidence_uncertainty(predictions[0][list(unlabeled_index).index(x)]),
                                             sort_df_des["unlabeled_index"]))
            df['unlabeled_confidences'] = unlabeled_confidences
            df.to_csv(global_path + f'{exp}_ranks/maskMeans_act'+str(iteration) + '.csv')
            freq, ranges = np.histogram(unlabeled_confidences, bins=BINS)
            print(f"Selecting PL among whose confidence is above {ranges[-2]}, aka {freq[-1]/len(unlabeled_confidences)} of PLs")
            filtered_unlabeled_index = [int(ui) for uc, ui in zip(unlabeled_confidences, sort_df_des["unlabeled_index"])
                                        if uc >= ranges[-2]]
            if len(filtered_unlabeled_index) < nb_pseudo:
                filling = nb_pseudo - len(filtered_unlabeled_index)
                filtered_unlabeled_index += list(map(int, filter(lambda x: x not in filtered_unlabeled_index,
                                                                 sort_df_des["unlabeled_index"])))[-filling:]
                print("Not enough PLs, filled to", len(filtered_unlabeled_index))
            pseudo_rank = filtered_unlabeled_index[-min(nb_pseudo, len(df) - nb_next_sample):]
        else:
            pseudo_rank = sort_df_des["unlabeled_index"][-min(nb_pseudo, len(df) - nb_next_sample):].to_list()

        pseudos = np.zeros((len(pseudo_rank), img_rows, img_cols, 1))
        df = pd.DataFrame(pseudo_rank, columns=['pseudo_rank'])
        if crf_configs is not None and post_process:
            df['discrep'] = ''
            pseudos_raw = np.zeros((len(pseudo_rank), len(crf_configs), img_rows, img_cols))
            print_and_log(f"Post processing {len(pseudo_rank)} predictions with denseCRF", logfile)
            ensCRF = EnsembleDenseCRF(crf_configs, arbitrator=lambda maps: np.mean(np.stack(maps), axis=0))
            pred_dices, post_dices, base_ls_dice_avg = [], [], []
            used = 0
            for i, pi in enumerate(pseudo_rank):
                pi_in_unlb = list(unlabeled_index).index(pi)
                print_and_log(f"Post processing {pi}-th data in unlabeled_index", logfile)
                np.save(f"{global_path}/{exp}_prediction/{pi}_ori", X_train[pi])
                np.save(f"{global_path}/{exp}_prediction/{pi}_pre",  predictions[0][pi_in_unlb])
                np.save(f"{global_path}/{exp}_prediction/{pi}_gt", y_train[pi])
                pred_dices.append(compute_dice_coef(y_train[pi], predictions[0][pi_in_unlb]))
                if pre_ensemble:
                    post = ensCRF(X_train[pi], [cv2.threshold(scale[pi_in_unlb], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
                                                for scale in predictions[:CRF_SCALES]], output_bases=False)
                else:
                    post, pls = ensCRF(X_train[pi], [cv2.threshold(scale[pi_in_unlb], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
                                                     for scale in predictions[:CRF_SCALES]], output_bases=True)
                    pseudos_raw[i] = pls
                    base_ls_dice_avg.append([compute_dice_coef(y_train[pi], m) for m in pls])

                disc = compute_dice_coef(predictions[0][pi_in_unlb], post)
                df.loc[i, 'discrep'] = disc
                if disc > USE_CRF_THRES:
                    pseudos[i] = post[..., None]
                    used += 1
                else:
                    # Discard unreliable pseudo labels with too large changes after post processing
                    pseudos[i] = predictions[0][pi_in_unlb]
                post_dices.append(compute_dice_coef(y_train[pi], post[..., None]))
            temp = np.mean(np.array(base_ls_dice_avg), axis=0)
            print_and_log(f"Before Dice Score: {sum(pred_dices) / len(pred_dices)}\t"
                          f"After Dice Score: {sum(post_dices) / len(post_dices)}\t"
                          f"Mean of CRFs Dice Score: {np.mean(temp)}\t"
                          f"Used {used}/{len(pseudo_rank)} of refined PLs", logfile)
            print_and_log(f"Best CRF ({max(temp)}): {crf_configs[list(temp).index(max(temp))]}", logfile)
            np.save(global_path + f"{exp}_logs/pseudo_raw" + str(iteration), pseudos_raw)
        else:
            df['meanIntersec'], df['R-DSC'], df['M-DSC'], df['L-DSC'] = '', '', '', ''
            for i, pi in enumerate(pseudo_rank):
                # Check if the most certain, aka most consistent samples are actually consistently poor
                pi_in_unlb = list(unlabeled_index).index(pi)
                sample_prediction1 = cv2.threshold(predictions[0][pi_in_unlb], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
                sample_prediction2 = cv2.threshold(predictions[1][pi_in_unlb], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
                sample_prediction3 = cv2.threshold(predictions[2][pi_in_unlb], 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
                cv2.imwrite(f"{global_path}{exp}_prediction/{unlabeled_index[pi_in_unlb]}_pseudo_prediction1.jpg",
                            sample_prediction1 * 255)
                cv2.imwrite(f"{global_path}{exp}_prediction/{unlabeled_index[pi_in_unlb]}_pseudo_prediction2.jpg",
                            sample_prediction2 * 255)
                cv2.imwrite(f"{global_path}{exp}_prediction/{unlabeled_index[pi_in_unlb]}_pseudo_prediction3.jpg",
                            sample_prediction3 * 255)
                df.loc[i, ['meanIntersec']] = compute_dice_coef_ds(sample_prediction1, sample_prediction2, sample_prediction3)
                df.loc[i, ["R-DSC"]] = compute_dice_coef(y_train[unlabeled_index[pi_in_unlb]], sample_prediction1)
                df.loc[i, ["M-DSC"]] = compute_dice_coef(y_train[unlabeled_index[pi_in_unlb]], sample_prediction2)
                df.loc[i, ["L-DSC"]] = compute_dice_coef(y_train[unlabeled_index[pi_in_unlb]], sample_prediction3)
                # pseudos[i] = np.array(np.mean(np.stack([np.array(scale[pi_in_unlb] >= .5).astype(scale.dtype) for scale in predictions]), axis=0) >= 0.5).astype(y_train.dtype)
                pseudos[i] = sample_prediction1[..., None]
        df.to_csv(global_path + f'{exp}_ranks/maskMeans_act' + str(iteration) + 'pseudos.csv')

    print_and_log(f'labeled_index list before: {len(labeled_index)}', logfile)
    print_and_log(f'unlabeled_index list before: {len(unlabeled_index)}', logfile)
    labeled_index = np.concatenate((labeled_index, transIndex)).astype(int)
    X_labeled_train = np.concatenate((X_train[labeled_index], X_train[np.array(pseudo_rank).astype(int)]))
    X_labeled_train = X_labeled_train.reshape([len(labeled_index) + len(pseudo_rank), img_rows, img_cols, 1])
    if len(pseudo_rank):
        print(len(pseudo_rank), pseudos.shape)
        y_labeled_train = np.concatenate((y_train[labeled_index], pseudos)).reshape([len(labeled_index) + len(pseudo_rank), img_rows, img_cols, 1])
    else:
        y_labeled_train = np.concatenate((y_train[labeled_index])).reshape([len(labeled_index), img_rows, img_cols, 1])
    print_and_log(f'X_labeled_train: {X_labeled_train.shape}', logfile)
    print_and_log(f'y_labeled_train: {y_labeled_train.shape}', logfile)
    # new_unlabeled_index = np.delete(unlabeled_index, transIndex, 0)
    unlabeled_index = list(filter(lambda x: x not in list(transIndex), unlabeled_index))
    print_and_log(f'labeled_index list after: {len(labeled_index)}', logfile)
    print_and_log(f'unlabeled_index list after: {len(unlabeled_index)}', logfile)

    del modelPredictions, predictions
    return X_labeled_train, y_labeled_train, labeled_index, unlabeled_index


def trainGenerator(x, y, seed=1, out_n=3):
    from keras.preprocessing.image import ImageDataGenerator
    image_generator = ImageDataGenerator(**aug_dict).flow(x, batch_size=batch_size, seed=seed)
    mask_generator = ImageDataGenerator(**aug_dict).flow(y, batch_size=batch_size, seed=seed)
    for img, mask in zip(image_generator, mask_generator):
        yield img, [mask] * out_n if out_n > 1 else mask


def log(history, step, log_file):
    """
    Writes the training history to the log file.
    :param history: Training history. Dictionary with training and validation scores.
    :param step: Training step
    :param log_file: Log file.
    """
    if isinstance(history, str):
        with open(log_file, "a+") as f:
            f.write(f"{datetime.fromtimestamp(time.time())}\t{history}\n")
    else:
        keys = history.history.keys()
        with open(log_file, "a+") as f:
            f.write(f"{datetime.fromtimestamp(time.time())}\t" + "\t".join(tuple(keys)) + "\n")
            for i in range(0, len(history.history["loss"])):
                f.write(f"{step}\t" + "\t".join(tuple(str(history.history[k][i]) for k in keys)) + "\n")


def create_paths():
    """
    Creates all the output paths.
    """
    path_ranks = global_path + f"{exp}_ranks/"
    path_logs = global_path + f"{exp}_logs/"
    path_plots = global_path + f"{exp}_plots/"
    path_models = global_path + f"{exp}/"
    path_predictions = global_path + f"{exp}_prediction"

    if not os.path.exists(path_ranks):
        os.makedirs(path_ranks)
        print("Path created: ", path_ranks)

    if not os.path.exists(path_logs):
        os.makedirs(path_logs)
        print("Path created: ", path_logs)

    if not os.path.exists(path_plots):
        os.makedirs(path_plots)
        print("Path created: ", path_plots)

    if not os.path.exists(path_models):
        os.makedirs(path_models)
        print("Path created: ", path_models)

    if not os.path.exists(path_predictions):
        os.makedirs(path_predictions)
        print("Path created: ", path_predictions)


def savehistory(history, savename):
    hist_df = pd.DataFrame(history.history)
    with open(savename, mode='w') as f:
        hist_df.to_csv(f)
