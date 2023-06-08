
import numpy as np
from collections import defaultdict
from datetime import datetime
import logging


if __name__ == "__main__":
    test_matrix = np.array([[1, 2, 5, 6],
                            [1, 2, 55, 6],
                            [1, 8, 5, 6],
                            [1, 2, 5, 0.1]])
    action_coverage = test_matrix.max(axis=1)
    print(action_coverage)

    # 5, 55, 8, 5
