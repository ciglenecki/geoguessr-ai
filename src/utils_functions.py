import argparse
import os
import sys
from datetime import datetime
from math import floor
from pathlib import Path
from typing import List, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class InvalidRatios(Exception):
    pass


T = TypeVar("T")


def name_without_extension(filename: Union[Path, str]):
    return Path(filename).stem


def split_by_ratio(array: np.ndarray, *ratios, use_whole_array=False) -> List[np.ndarray]:
    """
    Splits the ndarray for given ratios

    Arguments:
        array: array that will be splited
        use_whole_array: if set to True elements won't be discarted. Sum of ratios will be scaled to 1
        ratios: ratios used for splitting

    Example 1 use_whole_array = False:
        ratios = (0.2, 0.3)
        array = [1,2,3,4,5,6,7,8,9,10]
        returns [[1, 2], [3, 4, 5]]

    Example 2 use_whole_array = True:
        ratios = (0.2, 0.3)
        array = [1,2,3,4,5,6,7,8,9,10]
        returns [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10]]

    Example 3 use_whole_array = False:
        ratios = (0.2)
        array = [1,2,3,4,5,6,7,8,9,10]
        returns [[1, 2]]
    """

    ratios = np.array(ratios)
    if use_whole_array:
        ratios = ratios / ratios.sum()
        ratios = np.around(ratios, 3)
    ind = np.add.accumulate(np.array(ratios) * len(array)).astype(int)
    return [x.tolist() for x in np.split(array, ind)][: len(ratios)]


def get_train_test_indices(dataset: Dataset, test_size, dataset_frac=1.0, shuffle=True):
    dataset_size = floor(len(dataset) * dataset_frac)  # type: ignore # - dataseta has length only __len__ is implemented
    dataset_indices = np.arange(dataset_size)

    if shuffle:
        np.random.shuffle(dataset_indices)

    test_split_index = int(np.floor(test_size * dataset_size))
    train_indices, test_indices = dataset_indices[test_split_index:], dataset_indices[:test_split_index]

    train_len = test_split_index
    test_len = len(dataset_indices) - test_split_index
    return


def get_timestamp():
    return datetime.today().strftime("%y-%m-%d-%H-%M-%S")


def set_train_val_frac(dataset_size: int, train_split_factor, val_split_factor) -> Tuple[int, int]:
    """Set size for training and validation set
    Args:
        train_split_factor [0,1] - percentage of train images
        val_split_factor [0,1] - percentage of validation images
    """

    if train_split_factor + val_split_factor != 1.0:
        sys.exit("Train and split factor should add up to 1")

    train_frac: int = np.rint(train_split_factor * dataset_size)
    val_frac: int = dataset_size - train_frac

    return train_frac, val_frac


def one_hot_encode(index: int, length: int):
    zeros = np.zeros(shape=length)
    zeros[index] = 1
    return zeros


def np_set_default_printoptions():
    np.set_printoptions(edgeitems=3, infstr="inf", linewidth=75, nanstr="nan", precision=8, suppress=False, threshold=1000, formatter=None)


def imshow(img, y_true, y_pred=None):

    with torch.no_grad():
        np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

        if y_pred is None:
            torch.zeros(y_true.shape)

        # plt.gcf()
        plt.imshow(img.numpy().transpose((1, 2, 0)))
        plt.title("True {}\nPred: {}".format(y_true.numpy(), y_pred.numpy()))
        plt.show()
        np_set_default_printoptions()


def is_valid_fractions_array(array):
    if len(array) != 3 or sum(array) != 1:
        raise argparse.ArgumentError(array, "There has to be 3 fractions (train, val, test) that sum to 1")
    return array


def is_between_0_1(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def is_valid_image_size(x):
    valid_sizes = [224, 112, 56, 28, 14]
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a int literal" % (x,))
    if x not in valid_sizes:
        raise argparse.ArgumentTypeError("Size has to be any of: [224, 112, 56, 28, 14]")
    return x


def is_positive_int(value):
    int_value = int(value)
    if int_value < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_value


def is_valid_unfreeze_arg(arg):
    """Positive int or 'all'"""
    if type(arg) is str and arg == "all":
        return arg
    try:
        return is_positive_int(arg)  # is_positive_int's raise will be caught by the except
    except:
        raise argparse.ArgumentTypeError("%s has to be positive int or 'all'" % arg)
    return args


def is_valid_dir(arg):
    if not os.path.isdir(arg):
        raise argparse.ArgumentError(arg, "Argument should be a path to directory")
    return arg


class SocketConcatenator(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
        self.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def stdout_to_file(file: Path):
    """
    Pipes standard input to standard input and to a file.
    """
    print("Standard output piped to file:")
    f = open(Path(file), "w")
    sys.stdout = SocketConcatenator(sys.stdout, f)


def add_prefix_to_keys(dict: dict, prefix):
    return {prefix + k: v for k, v in dict.items()}


def is_primitive(obj):
    return not hasattr(obj, "__dict__") and type(obj) is not list


if __name__ == "__main__":
    pass
