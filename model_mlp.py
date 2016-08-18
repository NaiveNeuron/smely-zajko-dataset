from __future__ import print_function

import utils
import model_utils
import models

import numpy as np
np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_epoch = 30

dataset = utils.dataset_from_folder('./plzen/train')
X_train, y_train = model_utils.prepare_pixelized_dataset(dataset)

dataset = utils.dataset_from_folder('./plzen/test')
X_test, y_test = model_utils.prepare_pixelized_dataset(dataset)


print('\n')
model = models.mlp(n_input=75, architecture=[(8, 'sigmoid'), (1, 'sigmoid')])
model_utils.train_and_eval(model, X_train, y_train, X_test, y_test,
                           batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(12, 'sigmoid'), (1, 'sigmoid')])
model_utils.train_and_eval(model, X_train, y_train, X_test, y_test,
                           batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(20, 'sigmoid'), (1, 'sigmoid')])
model_utils.train_and_eval(model, X_train, y_train, X_test, y_test,
                           batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(6, 'sigmoid'),
                                             (6, 'sigmoid'),
                                             (1, 'sigmoid')])
model_utils.train_and_eval(model, X_train, y_train, X_test, y_test,
                           batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(10, 'sigmoid'),
                                             (10, 'sigmoid'),
                                             (1, 'sigmoid')])
model_utils.train_and_eval(model, X_train, y_train, X_test, y_test,
                           batch_size=batch_size, nb_epoch=nb_epoch)
