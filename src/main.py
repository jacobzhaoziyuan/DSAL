from __future__ import print_function

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from data import load_train_data, load_val_data
from data import create_train_data, create_val_data
from utils import *
from unet import *
import os, numpy as np
from shutil import copyfile

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("Running experiment", exp)
create_paths()
copyfile(os.path.join(os.path.split(os.path.realpath(__file__))[0], "constants.py"), global_path + f"{exp}/constants.py")
log_file = global_path + f"{exp}_logs/log_file.txt"
create_train_data()
create_val_data("val")
create_val_data("test")

# data definition
X_train, y_train = load_train_data()
if "ISIC" in global_path:
    X_train, y_train = X_train[:nb_train], y_train[:nb_train]
    X_val, y_val = X_train[nb_train:], y_train[nb_train:]
else:
    X_val, y_val = load_val_data("test")

labeled_index = np.arange(0, nb_labeled)
#assert len(labeled_index) == 10
unlabeled_index = np.array(list(filter(lambda x: x not in labeled_index, range(nb_train))))

#(1) Initialize model
model = get_unet_ds(dropout=True)  #
# model.load_weights(initial_weights_path)
mon = {"monitor": "conv2d_19_dice_coef", "mode": "max"}


if resume_on < 1:
    es = EarlyStopping(patience=50, **mon)
    lrs = ReduceLROnPlateau(verbose=1, cooldown=10, min_lr=lr / 100, patience=40, factor=0.1, **mon)
    model_checkpoint = ModelCheckpoint(initial_weights_path, save_best_only=True, verbose=1, **mon)
    log(f"Starting training of base model", 0, log_file)
    callbacks = [model_checkpoint] + ([] if "ISIC" in global_path else [lrs]) + ([es] if "es" in exp else [])
    if apply_augmentation:
        train_gen = trainGenerator(X_train[labeled_index], y_train[labeled_index])
        history = model.fit_generator(train_gen, steps_per_epoch=len(X_train[labeled_index]) // batch_size,
                                      callbacks=callbacks, verbose=1, validation_data=(X_val, [y_val] * 3),
                                      epochs=nb_initial_epochs)

        savehistory(history, global_path + f'{exp}_logs/init_train_aug_history.csv')
    else:
        history = model.fit(X_train[labeled_index], [y_train[labeled_index] for _ in range(3)], batch_size=batch_size,
                            nb_epoch=nb_initial_epochs, validation_data=(X_val, [y_val] * 3), verbose=1, shuffle=True,
                            callbacks=callbacks)
        savehistory(history, global_path + f'{exp}_logs/init_train_history.csv')
    log(history, 0, log_file)
    resume_on = 1
else:
    model.load_weights(initial_weights_path)

# Active loop
log("Active Loop", resume_on, log_file)

for iteration in range(resume_on, nb_iterations + 1):

    if iteration == 1:
        weights_path = initial_weights_path
    else:
        weights_path = global_path + f"{exp}/active_model" + str(iteration - 1) + ".h5"
    print(weights_path)
    # (2) Labeling
    crf_configs = MCRF_cfgs if post_process else None
    if iteration > 1 and os.path.exists(f"{global_path}{exp}_ranks/labeled_indices_activ{iteration - 1}.npy"):
        labeled_index = np.load(f"{global_path}{exp}_ranks/labefled_indices_activ{iteration - 1}.npy")
        unlabeled_index = np.load(f"{global_path}{exp}_ranks/unlabeled_indices_activ{iteration - 1}.npy")
    log(f"Compute trianing set of AL's {iteration} iteration", iteration, log_file)
    X_labeled_train, y_labeled_train, labeled_index, unlabeled_index = compute_train_sets_ds(X_train, y_train, labeled_index,
                                                                                             unlabeled_index, weights_path,
                                                                                             iteration, nb_next_sample, log_file,
                                                                                             crf_configs=crf_configs,
                                                                                             nb_pseudo=nb_pseudo_initial + pseudo_rate * (iteration - pseudo_epoch))
    log(f"Complete Computing", iteration, log_file)
    np.save(f"{global_path}{exp}_ranks/labeled_indices_activ{iteration}", labeled_index)
    np.save(f"{global_path}{exp}_ranks/unlabeled_indices_activ{iteration}", unlabeled_index)
    # (3) Training
    model.load_weights(weights_path)
    model_checkpoint = ModelCheckpoint(global_path + f"{exp}/active_model" + str(iteration) + ".h5", save_best_only=True,
                                       verbose=1, **mon)
    lrs = ReduceLROnPlateau(verbose=1, cooldown=10, min_lr=lr / 100, patience=25, factor=0.1, **mon)
    es = EarlyStopping(patience=30, **mon)
    # model.load_weights(weights)
    log(f"Starting training of AL's {iteration} iteration", iteration, log_file)
    callbacks = [model_checkpoint] + ([] if "ISIC" in global_path else [lrs]) + ([es] if "es" in exp else [])
    if apply_augmentation:
        # data_gen = data_generator(MultiOutputImageDataGenerator)
        train_gen = trainGenerator(X_labeled_train, y_labeled_train)
        history = model.fit_generator(train_gen, epochs=nb_active_epochs, verbose=1, steps_per_epoch=len(X_labeled_train) // batch_size,
                                      callbacks=callbacks, validation_data=(X_val, [y_val] * 3))
        savename = global_path + f'{exp}_logs/active_train_itera'+ str(iteration) + '_aug_history.csv'
    else:
        history = model.fit(X_labeled_train, [y_labeled_train] * 3, batch_size=batch_size, epochs=nb_active_epochs,
                            validation_data=(X_val, [y_val] * 3), verbose=1, shuffle=True, callbacks=callbacks)
        savename = global_path + f'{exp}_logs/active_train_itera'+ str(iteration) + '_history.csv'

    log(history, iteration, log_file)
    savehistory(history, savename)
    model.save(final_weights_path)
    model.save(global_path + f"{exp}/active_model_last{iteration}.h5")
