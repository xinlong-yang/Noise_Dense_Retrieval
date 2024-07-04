from array import array
import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
import torchvision
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import random
from PIL import ImageFilter

def multiclass_noisify(y, P, random_state=0, isonehot=False):
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0] 
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1])) 
    assert (P >= 0.0).all() 
    if isonehot == False: 
        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]
    else:
        m = y.shape[0]
        new_y = np.zeros(m,dtype=int)
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = np.argmax(y[idx])
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]
    return new_y


def generate_noise_matrix(noise_type, closeset_noise_ratio, openset_noise_ratio=0.8, nb_classes=10):
    assert closeset_noise_ratio > 0.0, 'noise rate must be greater than 0.0'
    assert 0.0 <= openset_noise_ratio < 1.0, 'the ratio of out-of-distribution class must be within [0.0, 1.0)'
    closeset_nb_classes = int(nb_classes * (1 - openset_noise_ratio)) 
    if noise_type == 'symmetric':
        P = np.ones((nb_classes, nb_classes))
        P = (closeset_noise_ratio / (closeset_nb_classes - 1)) * P
        for i in range(closeset_nb_classes):
            P[i, i] = 1.0 - closeset_noise_ratio
        for i in range(closeset_nb_classes, nb_classes):
            P[i, :] = 1.0 / closeset_nb_classes
        for i in range(closeset_nb_classes, nb_classes):
            P[:, i] = 0.0
    elif noise_type == 'pairflip':
        P = np.eye(nb_classes)
        P[0, 0], P[0, 1] = 1.0 - closeset_noise_ratio, closeset_noise_ratio
        for i in range(1, closeset_nb_classes - 1):
            P[i, i], P[i, i + 1] = 1.0 - closeset_noise_ratio, closeset_noise_ratio
        P[closeset_nb_classes - 1, closeset_nb_classes - 1] = 1.0 - closeset_noise_ratio
        P[closeset_nb_classes - 1, 0] = closeset_noise_ratio
        for i in range(closeset_nb_classes, nb_classes):
            P[i, :] = 1.0 / closeset_nb_classes
        for i in range(closeset_nb_classes, nb_classes):
            P[:, i] = 0.0
    else:
        raise AssertionError("noise type must be either symmetric or pairflip")
    return P

# add noise
def noisify(y_train, noise_transition_matrix, random_state=None, isonehot = False):
    y_train_noisy = multiclass_noisify(y_train, P=noise_transition_matrix, random_state=random_state, isonehot = isonehot)
    if isonehot == False:
        actual_noise = (y_train_noisy != y_train).mean()
    else:
        actual_noise = (y_train_noisy != np.argmax(y_train)).mean()
    assert actual_noise > 0.0 
    return y_train_noisy, actual_noise


def noisify_dataset(nb_classes=10, train_labels=None, noise_type=None,
                    closeset_noise_ratio=0.0, openset_noise_ratio=0.0, random_state=0, verbose=True, isonehot = False):
    noise_transition_matrix = generate_noise_matrix(noise_type, closeset_noise_ratio, openset_noise_ratio, nb_classes)
    train_noisy_labels, actual_noise_rate = noisify(train_labels, noise_transition_matrix, random_state, isonehot = isonehot)
    if verbose:
        print(f'Noise Transition Matrix: \n {noise_transition_matrix}')
        print(f'Noise Type: {noise_type} (close set: {closeset_noise_ratio}, open set: {openset_noise_ratio})\n'
              f'Actual Total Noise Ratio: {actual_noise_rate:.3f}')
    return train_noisy_labels, actual_noise_rate