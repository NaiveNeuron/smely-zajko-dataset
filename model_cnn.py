from __future__ import print_function
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.core import Activation, Dense, Flatten, Dropout, MaxoutDense
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
import model_utils
import numpy as np
import utils
import h5py
import os


np.random.seed(1337)  # for reproducibility

batch_size = 32
nb_epoch = 100
window = (20, 20)
stride = 5
regression = False
mean_train = 87.062

h5filename = 'data_classification_normalized_middle_pixel.h5'

# eigenvalues & eigenvectors for plzen dataset
einval = np.array([0.91650828, 0.0733283, 0.04866549])
vec = np.array([[-0.5448285, 0.82209843, 0.16527574],
                [-0.60238846, -0.24659869, -0.75915561],
                [-0.58334386, -0.51316981, 0.6295766]])

if os.path.isfile(h5filename) is False:
    dataset_train = utils.augmented_dataset_from_folder('./dataset/train',
                                                        einval, vec, resize=None, mask_ext='_mask.jpg')
    train_data = model_utils.prepare_pixelized_dataset(dataset_train, window,
                                                       stride=stride,
                                                       image_by_image=True,
                                                       regression=regression)
    X_train, y_train = model_utils.reshape_dataset(train_data, window,
                                                   regression=regression,
                                                   y_applied_function=lambda x: x)
    
    dataset_test = utils.augmented_dataset_from_folder('./dataset/test',
                                                       einval, vec, resize=None, mask_ext='_mask.jpg')
    
    test_data = model_utils.prepare_pixelized_dataset(dataset_test, window,
                                                      stride=stride,
                                                      image_by_image=True,
                                                      regression=regression)
    X_test, y_test = model_utils.reshape_dataset(test_data, window,
                                                 regression=regression,
                                                 y_applied_function=lambda x: x)
    X_train = (X_train - mean_train) / 255.0
    # y_train = y_train / 255.0
    X_test = (X_test - mean_train) / 255.0
    # y_test = y_test / 255.0
    h5f = h5py.File(h5filename, 'w')
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('y_test', data=y_test)
    h5f.close()
else:
    h5f = h5py.File(h5filename, 'r')
    X_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]
    h5f.close()


print(np.min(X_train), np.max(X_train))
print("Dataset loaded X shape: {}, y shape: {}".format(X_train.shape,
                                                       y_train.shape))
# utils.show_dataset_samples(X_train, y_train)

if regression is True:
    model = Sequential()
    model.add(Convolution2D(32, 10, 10, border_mode='same',
                            input_shape=(3, window[0], window[1])))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 10, 10, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(3200))
    model.add(Activation('relu'))
    model.add(MaxoutDense(1000, nb_feature=2))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(400))
    model.add(Activation('relu'))
    adam = Adam(lr=0.01, decay=0.0005)
    model.compile(loss='mean_squared_error', optimizer=adam)
else:
    model = Sequential()
    model.add(Convolution2D(10, 5, 5, border_mode='same',
                            input_shape=(3, window[0], window[1])))
    model.add(Activation('relu'))
    model.add(Convolution2D(10, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(10, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(10, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    adam = Adam(lr=0.01, decay=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=['accuracy'])
    

# checkpoint
filepath="weights-improvement-epoch-{epoch:02d}-val_acc-{val_acc:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                             verbose=1, save_best_only=True)
earlystop = EarlyStopping(monitor='val_acc', patience=5, verbose=0)
callbacks_list = [checkpoint, earlystop]

history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_split=0.1, callbacks=callbacks_list)

# if regression is True:
#     model_utils.plot_history(history, [['loss', 'val_loss']], [['o', 'o']])
# else:
#     model_utils.plot_history(history, [['loss', 'val_loss'],
#                                        ['acc', 'val_acc']],
#                                       [['o', 'o'],
#                                        ['-o', '-go']])
# 
# weights = model.layers[0].get_weights()[0]
# model_utils.show_weights(weights)

model.save('cnn_class_normalized.h5')

if regression is True:
    loss = model.evaluate(X_test, y_test, verbose=1)
    print("\nloss: {:.4}".format(loss))
else:
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print("\nloss: {:.4}, acc: {:.4}".format(loss, acc))
