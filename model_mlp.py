from __future__ import print_function
import utils
import model_utils
import models
import numpy as np


np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_epoch = 30
window = (5, 5)
stride = 5

# eigenvalues & eigenvectors for plzen dataset
einval = np.array([0.91650828, 0.0733283, 0.04866549])
vec = np.array([[-0.5448285, 0.82209843, 0.16527574],
                [-0.60238846, -0.24659869, -0.75915561],
                [-0.58334386, -0.51316981, 0.6295766]])

dataset = utils.augmented_dataset_from_folder('./plzen/train', einval,
                                              vec, resize=None)
train_data = model_utils.prepare_pixelized_dataset(dataset, window,
                                                   stride=stride,
                                                   regression=False,
                                                   image_by_image=True)
X_train, y_train = model_utils.reshape_dataset(train_data, window,
                                               regression=False,
                                               y_applied_function=np.asarray)
X_train = np.reshape(X_train, (X_train.shape[0], -1))

dataset = utils.augmented_dataset_from_folder('./plzen/test', einval,
                                              vec, resize=None)
test_data = model_utils.prepare_pixelized_dataset(dataset, window,
                                                  stride=stride,
                                                  regression=False,
                                                  image_by_image=True)
X_test, y_test = model_utils.reshape_dataset(test_data, window,
                                             regression=False,
                                             y_applied_function=np.asarray)
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print("Done loading dataset X: {}, y: {}".format(X_train.shape, y_train.shape))


print('\n')
model = models.mlp(n_input=75, architecture=[(8, 'relu'), (1, 'sigmoid')])
model_utils.train_and_eval(model, X_train, y_train, X_test, y_test,
                           batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(12, 'relu'), (1, 'sigmoid')])
model_utils.train_and_eval(model, X_train, y_train, X_test, y_test,
                           batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(20, 'relu'), (1, 'sigmoid')])
model_utils.train_and_eval(model, X_train, y_train, X_test, y_test,
                           batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(6, 'relu'),
                                             (6, 'relu'),
                                             (1, 'sigmoid')])
model_utils.train_and_eval(model, X_train, y_train, X_test, y_test,
                           batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(10, 'relu'),
                                             (10, 'relu'),
                                             (1, 'sigmoid')])
model_utils.train_and_eval(model, X_train, y_train, X_test, y_test,
                           batch_size=batch_size, nb_epoch=nb_epoch)
