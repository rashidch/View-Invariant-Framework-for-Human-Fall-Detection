import numpy as np
import random

def uniform_sample(data_numpy, size):
    # input: T,V,C
    T, V, C = data_numpy.shape
    if T == size:
        return data_numpy
    interval = T / size
    print(interval)
    uniform_list = [int(i * interval) for i in range(size)]
    print(uniform_list)
    return data_numpy[uniform_list,:]


def random_sample(data_numpy, size):
    #increase temporal dimension length by replicating random samples with in sequence  
    # input: T,V,C
    T, V, C = data_numpy.shape
    if T == size:
        return data_numpy
    interval = int(np.ceil(size / T))
    random_list = sorted(random.sample(list(range(T))*interval, size))
    print(random_list)
    return data_numpy[random_list,:]

def random_choose_sample(data_numpy, size, center=False):
    # input: T,V,C
    T,V,C = data_numpy.shape
    if size < 0:
        assert 'resize shape is not right'
    if T == size:
        return data_numpy
    elif T < size:
        return data_numpy 
    else:
        if center:
            begin = (T - size) // 2
        else:
            begin = random.randint(0, T - size)
        print(begin)
        return data_numpy[begin:begin + size, :, :]