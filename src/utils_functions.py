import argparse
import os
import random
import string
import sys
import time
from datetime import datetime
from glob import glob
from math import floor
from pathlib import Path
from typing import List, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InvalidRatios(Exception):
    pass


T = TypeVar("T")


def get_dirs_only(path: Path):
    """Return only top level directories in the path"""
    return [d for d in (os.path.join(path, d1) for d1 in os.listdir(path)) if os.path.isdir(d)]


def tensor_sum_of_elements_to_one(ten: torch.Tensor, dim):
    """Scales elements of the tensor so that the sum is 1"""
    return ten / torch.sum(ten, dim=dim, keepdim=True)


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
    dataset_size = floor(
        len(dataset) * dataset_frac
    )  # type: ignore # - dataseta has length only __len__ is implemented
    dataset_indices = np.arange(dataset_size)

    if shuffle:
        np.random.shuffle(dataset_indices)

    test_split_index = int(np.floor(test_size * dataset_size))
    train_indices, test_indices = (
        dataset_indices[test_split_index:],
        dataset_indices[:test_split_index],
    )

    train_len = test_split_index
    test_len = len(dataset_indices) - test_split_index
    return


def get_timestamp():
    return datetime.today().strftime("%m-%d-%H-%M-%S")


def one_hot_encode(index: int, length: int):
    zeros = np.zeros(shape=length)
    zeros[index] = 1
    return zeros


def np_set_default_printoptions():
    np.set_printoptions(
        edgeitems=3,
        infstr="inf",
        linewidth=75,
        nanstr="nan",
        precision=8,
        suppress=False,
        threshold=1000,
        formatter=None,
    )


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
    if type(arg) is str and (arg == "all" or "layer" in arg):
        return arg
    try:
        if is_positive_int(arg):  # is_positive_int's raise will be caught by the except
            return int(arg)
    except:
        raise argparse.ArgumentTypeError("%s has to be positive int or 'all'" % arg)


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


def safely_save_df(df: pd.DataFrame, filepath: Path):
    """Safely save the dataframe by using and removing temporary files"""

    print("Saving file...", filepath)
    path_tmp = Path(str(filepath) + ".tmp")
    path_bak = Path(str(filepath) + ".bak")
    df.to_csv(path_tmp, mode="w+", index=True, header=True)

    if os.path.isfile(filepath):
        os.rename(filepath, path_bak)
    os.rename(path_tmp, filepath)

    if os.path.isfile(path_bak):
        os.remove(path_bak)


def stdout_to_file(file: Path):
    """
    Pipes standard input to standard input and to a file.
    """
    print("Standard output piped to file:")
    f = open(Path(file), "w")
    sys.stdout = SocketConcatenator(sys.stdout, f)
    sys.stderr = SocketConcatenator(sys.stderr, f)


def add_prefix_to_keys(dict: dict, prefix):
    return {prefix + k: v for k, v in dict.items()}


def is_primitive(obj):
    return not hasattr(obj, "__dict__") and type(obj) is not list


def flatten(t):
    return [item for sublist in t for item in sublist]


def print_df_sample(df: pd.DataFrame):
    pd.set_option("display.max_columns", None)
    print(
        "\nSample of the dataframe:",
        "First 3 rows:",
        df.head(n=3),
        "Random 3 rows:",
        df.sample(n=3),
        "Last 3 rows:",
        df.tail(n=3),
        "Dataframe stats:",
        df.describe(),
        sep="\n\n\n",
    )
    pd.reset_option("display.max_columns")


nato_alphabet = {
    "A": "Alpha",
    "B": "Bravo",
    "C": "Charlie",
    "D": "Delta",
    "E": "Echo",
    "F": "Foxtrot",
    "G": "Golf",
    "H": "Hotel",
    "I": "India",
    "J": "Juliett",
    "K": "Kilo",
    "L": "Lima",
    "M": "Mike",
    "N": "November",
    "O": "Oscar",
    "P": "Papa",
    "Q": "Quebec",
    "R": "Romeo",
    "S": "Sierra",
    "T": "Tango",
    "U": "Uniform",
    "V": "Victor",
    "W": "Whiskey",
    "X": "X-ray",
    "Y": "Yankee",
    "Z": "Zulu",
}


def random_codeword():
    """
    Return e.g.:
        Alpha_13, Zulu_39, X-ray_95
    """
    random_letter = random.choice(string.ascii_uppercase)
    return "{}_{}".format(nato_alphabet[random_letter], random.randint(10, 99))


def timeit(func):
    def timed(*args, **kwargs):
        print("START", func.__qualname__)
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print("END", func.__qualname__, "time:", round((te - ts) * 1000, 1), "ms")
        return result

    return timed


if __name__ == "__main__":
    pass
