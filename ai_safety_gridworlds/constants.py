from typing import List, Dict, Set, Tuple
import numpy as np
from enum import Enum


# Maximum number of actions that can be performed by the agend during the states exploration in the preprocessing step.
MAX_NR_ACTIONS: int = 100
RESULTS_DIR: str    = "../../Results"    

fn_states: str      = "states.npy"
fn_id_states: str   = "states_id_str.npy"        
fn_actions: str     = "actions.npy"    
fn_runtime: str     = "states_rt.npy"
fn_qtable: str      = "qtable.npy"
fn_ctable: str      = "ctable.npy"
fn_states_dict: str = "states_id_dict.npy"

class Strategies(Enum):
    ESTIMATE_STATES: str   = "Estimate"
    EXPLORE_STATES: str    = "Explore"

class Baselines(Enum):
    STARTING_STATE_BASELINE: str    = "Starting"
    INACTION_BASELINE: str          = "Inaction"
    STEPWISE_INACTION_BASELINE: str = "Stepwise"

class Environments(Enum):
    SOKOCOIN0: str = "sokocoin0"
    SOKOCOIN2: str = "sokocoin2"
    SOKOCOIN3: str = "sokocoin3"

class PARAMETRS(Enum):
    ENV_NAME: str           = "Env Name"
    NR_EPISODES: str        = "Nr Episodes"
    LEARNING_RATE: str      = "Learning Rate"
    STATE_SET_STRATEGY: str = "States Set Strategy"
    BASELINE: str           = "Baseline"
    Q_DISCOUNT: str         = "Q Discount"
    BETA: str               = "Beta"

StateSpaceSizeEstimations: Dict[str, int] = {
    Environments.SOKOCOIN0.value: 100,
    Environments.SOKOCOIN2.value: 47648,
    Environments.SOKOCOIN3.value: 6988675,
}

ACTIONS: Dict[int, str] = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "NOOP"
}

# General expected scrope of the experiments:
env_names                       = [Environments.SOKOCOIN0, Environments.SOKOCOIN2]    # , Environments.SOKOCOIN3
nr_episodes: int                = 10000
learning_rates: List[float]     = [.1, .3, .9]
discount_factors: List[float]   = [0.99, 1.]
base_lines: np.array            = np.array([Baselines.STARTING_STATE_BASELINE, Baselines.INACTION_BASELINE, Baselines.STEPWISE_INACTION_BASELINE])
strategies: List[Strategies]    = [Strategies.ESTIMATE_STATES, Strategies.EXPLORE_STATES]