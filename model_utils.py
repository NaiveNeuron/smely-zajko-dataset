from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt


def prepare_pixelized_dataset(dataset, window_x=5, window_y=5,
                              y_applied_function=np.asarray):
    Xs = []
    ys = []
    for datum in dataset:
        img = datum['img']
        mask = datum['mask']
        for selection, mask_selection in prepare_pixelized_image(img, mask):
            Xs.append(selection)
            ys.append(mask_selection)
    return np.array(Xs), y_applied_function(ys)


def prepare_pixelized_image(img, mask=None, window_x=5, window_y=5):
    h, w, ch = img.shape
    for j in range(w//window_y - 1):
        for i in range(h//window_x - 1):
            selection = img[i*window_x:(i+1)*window_x,
                            j*window_y:(j+1)*window_y] / 255.0
            selection = selection.reshape(-1)
            mask_selection = None
            if mask is not None:
                mask_selection = mask[i*window_x:(i+1)*window_x,
                                      j*window_y:(j+1)*window_y] / 255.0
                mask_selection = int(mask_selection.mean() > 0.5)
            yield selection, mask_selection


def plot_history(history):
    plt.plot(history.history['loss'], 'o', label='loss')
    plt.plot(history.history['val_loss'], '-go', label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def train_and_eval(model, X_train, y_train, X_test, y_test,
                   batch_size=128, nb_epoch=30,
                   verbose=0, score_name='MSE'):

    print('Training architecture: {}'.format(model.arch))
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch, verbose=verbose,
                        validation_split=0.1)
    print('Finished training.')

    print('\nComputing test score.')
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test {}: {}'.format(score_name, score))
    return history
