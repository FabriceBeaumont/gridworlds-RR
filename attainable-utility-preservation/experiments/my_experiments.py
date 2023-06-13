
import numpy as np
from collections import defaultdict
from datetime import datetime
import logging

# from agents.my_model_free_aup import ModelFreeAUPAgent
# from environment_helper import *

if __name__ == "__main__":
    # x = np.array([1, 2, 5, 6])
    x = [1, 2, 5, 6]

    last_state_id = x.index(5)
    print(last_state_id) # Expect 2


#      a_1 a_2 a_3 a_4
# s_0 [[1, 2, 5, 6],
# s_1  [1, 2, 55, 6],
# s_3  [1, 8, 5, 6],
# s_4  [1, 2, 5, 0.1]]