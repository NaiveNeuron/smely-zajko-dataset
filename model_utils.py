from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np


def prepare_pixelized_dataset(dataset, window_x=5, window_y=5,
                              y_applied_function=np.asarray):
    Xs = []
    ys = []
    for datum in dataset:
        img = datum['img']
        mask = datum['mask']
        Xs.append(prepare_pixelized_image(img, window_x, window_y))
        if mask is not None:
            ms = prepare_pixelized_image(mask, window_x, window_y)
            ys.append((ms.mean(axis=1) > 0.5).astype(np.uint))
    Xs = np.asarray(Xs)
    ys = np.squeeze(np.asarray(ys)).reshape(-1)
    return Xs.reshape(-1, Xs.shape[-1]), y_applied_function(ys)


def prepare_pixelized_image(img, window_x=5, window_y=5):
    h, w = img.shape[:2]
    num_windows_x = w//window_x
    num_windows_y = h//window_y
    if img.ndim == 3:
        res = img.reshape(num_windows_y, window_x, -1, window_y, img.shape[-1])
        res = res.swapaxes(1, 2).reshape(-1, window_x, window_y, img.shape[-1])
    else:
        res = img.reshape(num_windows_y, window_x, -1, window_y)
        res = res.swapaxes(1, 2).reshape(-1, window_x, window_y)
    return res.reshape(num_windows_x * num_windows_y, -1) / 255.0


def plot_history(history):
    plt.plot(history.history['loss'], 'o', label='loss')
    plt.plot(history.history['val_loss'], '-go', label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def train_and_eval(model, X_train, y_train, X_test, y_test,
                   batch_size=128, nb_epoch=30,
                   verbose=0, score_name='MSE', score_format='{}'):

    print('Training architecture: {}'.format(model.arch))
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch, verbose=verbose,
                        validation_split=0.1)
    print('Finished training.')

    print('\nComputing test score.')
    score = model.evaluate(X_test, y_test, verbose=0)
    msg = 'Test {}: ' + score_format
    print(msg.format(score_name, score))
    return history
