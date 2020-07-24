from ..Tensor import Tensor
import numpy as np


def prepare_dataset(arrays, size):
    for key, array in enumerate(arrays):
        data = np.array(array)
        if len(data.shape) != 3:
            arrays[key] = Tensor().add_array(data.flat).set_size(size)
        else:
            arrays[key] = Tensor().set_matrix(data)
    return arrays
