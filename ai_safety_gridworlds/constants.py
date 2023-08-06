from typing import List, Dict, Set, Tuple
import numpy as np
from enum import Enum


# Maximum number of actions that can be performed by the agend during the states exploration in the preprocessing step.
MAX_NR_ACTIONS: int = 100
RESULTS_DIR: str    = "../../GridworldResults"

dir_evaluation: str = "Evaluations"

# Definition of a couple of filenames for standardized file writing and reading.
fn_states_npy: str      = "states.npy"
fn_id_states_npy: str   = "states_id_str.npy"        
fn_actions_npy: str     = "actions.npy"    
fn_runtime_npy: str     = "states_rt.npy"
fn_qtable_npy: str      = "qtable.npy"
fn_ctable_npy: str      = "ctable.npy"
fn_states_dict_npy: str = "states_id_dict.npy"
fn_experiments_csv: str = "AllExperiments.csv"

fn_general_csv: str                 = "general.csv"
fn_performances_csv: str            = "performances_table.csv"
fn_plot1_performance_jpeg: str      = "plot1_performance.jpeg"
fn_plot2_results_jpeg: str          = "plot2_results.jpeg"
fn_plot3_results_smooth_jpeg: str   = "plot3_results_smooth.jpeg"
fn_plot4_tde_jpeg: str              = "plot4_tde.jpeg"
fn_agent_journey: str               = "agent_journey"

results_col_rewards: str        = 'Rewards'
results_col_performances: str   = 'Performances'
results_col_tdes: str           = 'Tdes'
results_col_episodes: str       = 'Episodes' 

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
    METHOD_NAME: str            = "Method Name"
    ENV_NAME: str               = "Env Name"
    NR_EPISODES: str            = "Nr Episodes"
    LEARNING_RATE: str          = "Learning Rate"
    STATE_SPACE_STRATEGY: str   = "States Set Strategy"
    BASELINE: str               = "Baseline"
    Q_DISCOUNT: str             = "Discount Factor"
    BETA: str                   = "Beta"
    MAX_NR_STEPS: str           = "Max. Nr. of Steps"


StateSpaceSizeEstimations: Dict[str, int] = {
    Environments.SOKOCOIN0.value: 100,
    Environments.SOKOCOIN2.value: 47648,
    Environments.SOKOCOIN3.value: 6988675,
}

TrueStateSpaceSizes: Dict[str, int] = {
    Environments.SOKOCOIN0.value: 60,
    Environments.SOKOCOIN2.value: 29830,
    Environments.SOKOCOIN3.value: 237350, # AT LEAST!
}

ACTIONS: Dict[int, str] = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "NOOP"
}