import math


def tanh(x):
    return math.tanh(x)


def atanh(x):
    x = math.tanh(x)
    return 1 - x ** 2


functions = {
    'tanh': tanh
}

derivatives = {
    'tanh': atanh
}
