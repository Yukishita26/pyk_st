import numpy as np
import pandas as pd
import matplotlib as plt
import sys

class WrongSizeError(Exception):
    def __init__(self, text="Size not match"):
        self.text = text

# hight x width の array_like を length x 1 の array に縮約
def mean_with_weight(arr, weight=None, axis=0):
    if type(arr) == list: arr = np.array(arr)
    hight, width = arr.shape
    #length = arr.shape[1 - axis]
    vertical = arr.shape[axis]
    
    if weight == None:
        weight = np.ones(vertical)
    else:
        if len(weight) != vertical:
            raise WrongSizeError("weight doesn't match for array size")

    whole_weight = sum(weight)
    if axis == 0:
        return sum([arr[i,:] * weight[i] for i in range(hight)])/whole_weight
    else:
        return sum([arr[:,j] * weight[j] for j in range(width)])/whole_weight

if __name__ == "__main__":
    array = np.array([[1, 2, 3],
                      [4, 5, 6]])
    print(mean_with_weight(array, weight=[1, 0], axis=0))
    # => [1, 2, 3]

    print(mean_with_weight(array, weight=[1, 0, 1], axis=1))
    # => [2, 5]

    try:
        print(mean_with_weight(array, weight=[0, 1], axis=1))
    except WrongSizeError as wse:
        print(wse.text)
    # => weight doesn't match for array size

