from __future__ import print_function
from constants import *
smooth = 1.


def dice_coef(y_true, y_pred):
    from keras import backend as K
    K.common.image_dim_ordering()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_pred_f = K.cast(K.greater_equal(y_pred_f, 0.5), dtype=K.floatx())
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


#Override Dropout. Make it able at test time.
def call(self, inputs, training=None):
    from keras import backend as K
    K.common.image_dim_ordering()
    if 0. < self.rate < 1.:
        noise_shape = self._get_noise_shape(inputs)
        dropped_inputs = lambda: K.dropout(inputs, self.rate, noise_shape, seed=self.seed)
        if training:
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        else:
            return K.in_test_phase(inputs, dropped_inputs, training=None)
    return inputs


def get_unet(dropout, input_size = (img_rows,img_cols, 1), kernel_initializer="he_normal"):
    from keras.layers import Input, concatenate, Conv2D, MaxPool2D, UpSampling2D, Dropout
    from keras.models import Model
    from keras.optimizers import Adam
    from keras import backend as K
    K.common.image_dim_ordering()
    Dropout.call = call

    # inputs = Input(1, img_rows, img_cols)
    # inputs = Input(img_rows, img_cols, 1)
    inputs = Input(input_size, name="input_1")
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_1")(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_2")(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2), name="max_pooling2d_1")(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_3")(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_4")(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2), name="max_pooling2d_2")(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_5")(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_6")(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2), name="max_pooling2d_3")(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_7")(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_8")(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2), name="max_pooling2d_4")(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_9")(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_10")(conv5)
    conv5 = Dropout(0.5, name="dropout_1")(conv5, training=True if dropout else False)

    up6 = concatenate([UpSampling2D(size=(2, 2), name="up_sampling2d_1")(conv5), conv4], axis=3, name="concatenate_1")

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_11")(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_12")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2), name="up_sampling2d_2")(conv6), conv3], axis=3, name="concatenate_2")

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_13")(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_14")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2), name="up_sampling2d_3")(conv7), conv2], axis=3, name="concatenate_3")

    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_15")(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_16")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2), name="up_sampling2d_4")(conv8), conv1],axis=3, name="concatenate_4")

    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_17")(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_18")(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=kernel_initializer, name="conv2d_19")(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=lr), loss=lambda x, y: -dice_coef(x, y), metrics=[dice_coef])

    return model


def get_unet_ds(dropout, input_size=(img_rows, img_cols, 1), kernel_initializer="he_normal", interm=False, mp=False):
    """
    :param interm: deprecated, should be False, don't change it
    :return: keras model
    """
    from keras.layers import Input, Conv2D, MaxPool2D, Dropout, concatenate, UpSampling2D
    from keras.models import Model
    from keras.optimizers import Adam
    from keras import backend as K
    K.common.image_dim_ordering()
    Dropout.call = call

    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_1")(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_2")(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2), name="max_pooling2d_1")(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_3")(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_4")(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2), name="max_pooling2d_2")(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_5")(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_6")(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2), name="max_pooling2d_3")(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_7")(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_8")(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2), name="max_pooling2d_4")(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_9")(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_10")(conv5)
    conv5 = Dropout(0.5, name="dropout_1")(conv5, training=True if dropout else False)

    up6 = concatenate([UpSampling2D(size=(2, 2), name="up_sampling2d_1")(conv5), conv4], axis=3)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_11")(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_12")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2), name="up_sampling2d_2")(conv6), conv3], axis=3)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_13")(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_14")(conv7)

    convT1 = UpSampling2D(size=(4, 4), name="up_sampling2d_T1")(conv7)
    # print("convT1 shape:", convT1.shape)
    out1 = Conv2D(1, (1, 1), activation='sigmoid', name="conv2d_T1")(convT1)

    up8 = concatenate([UpSampling2D(size=(2, 2), name="up_sampling2d_3")(conv7), conv2], axis=3)

    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_15")(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_16")(conv8)

    convT2 = UpSampling2D(size=(2,2), name="up_sampling2d_T2")(conv8)
    # print("convT2 shape:", convT2.shape)
    out2 = Conv2D(1, (1, 1), activation='sigmoid', name="conv2d_T2")(convT2)

    up9 = concatenate([UpSampling2D(size=(2, 2), name="up_sampling2d_4")(conv8), conv1], axis=3)

    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_17")(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer, name="conv2d_18")(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name="conv2d_19")(conv9)

    model = Model(inputs=[inputs], outputs=[conv10, out1, out2])
    model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy" if USE_BCE else lambda x, y: -dice_coef(x, y),
                  loss_weights=loss_weights, metrics=[dice_coef] + (["accuracy"] if USE_BCE else []))
    if interm:
        model_interm = Model(inputs=[inputs], outputs=[conv7, conv8])
        return model, model_interm
    return model


if __name__ == '__main__':
    model=get_unet_ds(True)
    model.summary()
