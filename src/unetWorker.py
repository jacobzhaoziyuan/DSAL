import os
from constants import *
from utils import log, savehistory
from multiprocessing import Process
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)


class Worker(Process):
    def __init__(self, t, X_train, y_train, weights, iteration, mon, X_val=None, y_val=None, log_file=None):
        Process.__init__(self, name="ModelTrainer")
        self.t = t
        self.X_train = X_train
        self.y_train = y_train
        self.weights = weights
        self.iteration = iteration
        self.mon = mon
        self.X_val = X_val
        self.y_val = y_val
        self.log_file = log_file

    def run(self):
        from utils import trainGenerator
        from unet import get_unet, get_unet_ds
        model_func = get_unet_ds if "ds" in exp else get_unet
        from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.t)
        print(f"[ENSEMBLE] Training {self.t}-th base model")
        model = model_func(True, mp=True)
        model.load_weights(self.weights)
        model_checkpoint = ModelCheckpoint(global_path + f"{exp}/active_model{self.iteration}_{self.t}.h5", save_best_only=True,
                                           verbose=1, **self.mon)
        lrs = ReduceLROnPlateau(verbose=1, cooldown=10, min_lr=lr / 100, patience=40, factor=0.1, **self.mon)
        target = self.y_train if model_func == get_unet else [self.y_train for _ in range(3)]
        val_target = self.y_val if model_func == get_unet else [self.y_val for _ in range(3)]
        es = EarlyStopping(patience=50, **self.mon)
        if apply_augmentation:
            history = model.fit_generator(trainGenerator(self.X_train, self.y_train), epochs=nb_active_epochs,
                                          steps_per_epoch=len(self.X_train) // batch_size,
                                          callbacks=[model_checkpoint, lrs] + ([es] if "es" in exp else []),
                                          validation_data=(self.X_val, [self.y_val] * 3))
            savename = global_path + f'{exp}_logs/active_train_itera' + str(self.iteration) + f'_aug_history{self.t}.csv'
        else:
            history = model.fit(self.X_train, target, batch_size=batch_size, epochs=nb_active_epochs, shuffle=True,
                                callbacks=[model_checkpoint, lrs] + ([es] if "es" in exp else []),
                                validation_data=(self.X_val, val_target) if self.X_val is not None else None)
            savename = global_path + f'{exp}_logs/active_train_itera{self.iteration}_history{self.t}.csv'
        if self.log_file:
            log(f"{self.t}-th base model:\n", self.iteration, self.log_file)
            log(history, self.iteration, self.log_file)
        savehistory(history, savename)
        model.save(global_path + f"{exp}/active_model_last_{self.iteration}_{self.t}.h5")


class Scheduler:
    def __init__(self, T, X_train, y_train, weights, iteration, mon, X_val=None, y_val=None, log_file=None):
        self.workers = [Worker(t, X_train, y_train, weights, iteration, mon, X_val, y_val, log_file) for t in range(T)]

    def start(self):
        print("Train Ensemble Models in parallel...")
        for worker in self.workers:
            worker.start()

        for worker in self.workers:
            worker.join()

        print("Done parallel training.")