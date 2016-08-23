from __future__ import print_function

import math
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
            ys.append((ms.mean(axis=1) > 0.5))
    Xs = np.asarray(Xs)
    ys = np.squeeze(np.asarray(ys, dtype='int32')).reshape(-1)
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


def plot_history(history, plots=[['loss', 'val_loss']], format=[['o', 'o']]):
    for i, metrics in enumerate(plots):
        for j, metric in enumerate(metrics):
            plt.plot(history.history[metric], format[i][j], label=metric)
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend()
        if i != len(plots)-1:
            plt.figure()
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


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.
    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid

    Courtesy of cs231n
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(math.ceil(math.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in xrange(grid_size):
        x0, x1 = 0, W
        for x in xrange(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def show_weights(weights):
    grid = visualize_grid(weights.transpose(0, 2, 3, 1))
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.imshow(grid.astype('uint8'))
    plt.axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.show()
