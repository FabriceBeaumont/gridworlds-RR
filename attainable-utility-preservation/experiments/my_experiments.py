
import numpy as np
from collections import defaultdict
from datetime import datetime
import logging


if __name__ == "__main__":
    x = np.array([1, 2, 5, 6])

    mask = x > 3
    mask1 = x[x > 8]

    print(mask)
    print(len(mask))


    print(mask1)
    print(len(mask1))

    # 6, 55, 8, 5


#      a_1 a_2 a_3 a_4
# s_0 [[1, 2, 5, 6],
# s_1  [1, 2, 55, 6],
# s_3  [1, 8, 5, 6],
# s_4  [1, 2, 5, 0.1]]