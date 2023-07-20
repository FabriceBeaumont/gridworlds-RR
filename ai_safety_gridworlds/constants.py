from typing import List, Dict, Set, Tuple
import numpy as np
from enum import Enum

class Baselines(Enum):
    STARTING_STATE_BASELINE: str    = "Starting"
    INACTION_BASELINE: str          = "Inaction"
    STEPWISE_INACTION_BASELINE: str = "Stepwise"
        
ACTIONS: Dict[int, str] = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "NOOP"
}


env_names                       = ['sokocoin0', 'sokocoin2']    # , 'sokocoin3'
env_state_set_size_estimates    = [100, 47648]                  # , 6988675     #TODO: sparse matrix for qtable/cable?
nr_episodes: int                = 10000
learning_rates: List[float]     = [.1, .3, .9]
discount_factors: List[float]   = [0.99, 1.]
base_lines: np.array            = np.array([Baselines.STARTING_STATE_BASELINE, Baselines.INACTION_BASELINE, Baselines.STEPWISE_INACTION_BASELINE])
