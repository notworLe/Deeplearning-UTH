import numpy as np
import torch
if __name__ == '__main__':
    actual = np.random.randint(0, 2, (2,3))
    predict = np.random.randint(0, 2, (2,3))

    ac_list = []

    for i in actual:
        ac_list.extend(i)
    print(ac_list)
    print(predict)



