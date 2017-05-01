import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_minecraft_dataset(features):
    d = np.load('hw5_data_'+features+'.npz')
    data = d['arr_0']
    target = d['arr_1']
    return (data, target)


def load_digits_dataset():
    digits = datasets.load_digits()
    return (digits.data, digits.target)


def load_iris_dataset():
    iris = datasets.load_iris()
    return (iris.data, iris.target)


def split_dataset(data, target, train_size=0.8):
    '''Splits the provided data and targets into training and test sets'''
    data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=train_size, random_state=0)
    return data_train, data_test, target_train, target_test


def get_minecraft(features='histogram'):
    #features can be either histogram, rgb, or gray
    data, target = load_minecraft_dataset(features)
    return split_dataset(data, target, train_size=0.75)


def get_digits():
    data, target = load_digits_dataset()
    return split_dataset(data, target, train_size=0.8)


def get_iris():
    data, target = load_iris_dataset()
    return split_dataset(data, target, train_size=0.8)


def get_first_n_samples(data, target, n):
    return data[0:n, :], target[0:n]
