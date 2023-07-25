from typing import List, Dict, Set, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import os
import math
from csv import DictWriter, Sniffer
from datetime import timedelta, datetime
import time

import plotly.express as px
from helpers import factory

# Local imports
import constants as c

### COMPUTATIONAL FUNCITONS

GAME_ART = [['######',  # Level 0.
            '# A###',
            '# X  #',
            '##   #',
            '### G#',
            '######'],
            ['#########',  # Level 2.
            '#       #',
            '#  1A   #',
            '# C# ####',
            '#### #C #',
            '#     2 #',    
            '#       #',
            '#########'],
            ['##########',  # Level 3.
            '#    #   #',
            '#  1 A   #',
            '# C#     #',
            '####     #',
            '# C#  ####',
            '#  #  #C #',
            '# 3    2 #',
            '#        #',
            '##########'],
]  

# Sokocoin0 = Counter({'#': 25, ' ': 8,  'A': 1, 'X': 1, 'G': 1})
# Sokocoin2 = Counter({'#': 38, ' ': 29, 'C': 2, '1': 1, 'A': 1, '2': 1})
# Sokocoin3 = Counter({'#': 47, ' ': 46, 'C': 3, '1': 1, 'A': 1, '3': 1, '2': 1})


def my_comb(n, k) -> int:
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n-k)))

def print_sokocoin_state_space_size_estimations(env_state: List[str], verbose: bool = False) -> int:
    """ To count the number of all possible states, we have to consider all possible places, where the boxes and the agent can be.
    This number has to be multiplied by a factor accounting for wheter some (but not all) coins have been collected or not.
    To formulate an upper bound on the number of possible states we allow the boxes and the agent to be places on all grids, which are not 
    a box or an agent. 
    We place them separately, since states are not different, if we swap the boxes. 
    If there are n different boxes, but their placement does not matter, there are n! different configurations of them.
    Notice however, that this differentiation is NOT accounted for
    in the encoding in the environment and in straight forward implementations based on string comparisons!
    Dependin on the environment interpretation, the matrix of integer values may be more accurate in this regard.
    
    Examples for states which we count, but are not possible are when the agent is surrounded by boxes and walls.
    The agent cannot pull boxes behind him and thus cannot create this state.
    """
    counter = Counter("".join(env_state))
    nr_whitespaces = counter[' ']
    nr_coins = counter['C'] + counter['G']
    nr_boxes = np.sum([counter[str(x)] for x in range(5)]) + counter['X']
    nr_agents = 1

    if verbose: print(f"Nr whitespaces:\t{nr_whitespaces}\nNr coins:\t{nr_coins}\nNr boxes:\t{nr_boxes}")

    nr_states = 0

    for nr_collected_coins in range(nr_coins):
        free_coin_grids = nr_coins - nr_collected_coins
        
        # Agent and box configurations.            
        nr_box_placements   = my_comb(nr_whitespaces + free_coin_grids + nr_boxes,  nr_boxes)
        nr_agent_placements = my_comb(nr_whitespaces + free_coin_grids + nr_agents, nr_agents)
        nr_of_coin_orders   = my_comb(nr_coins, nr_collected_coins)

        # Compte the esimtate.
        nr_states += nr_box_placements * nr_agent_placements * nr_of_coin_orders

    return nr_states

def load_sokocoin_env(env_name) -> Tuple:
    # Get environment.
    env_name_lvl_dict = {c.Environments.SOKOCOIN0.value: 0, c.Environments.SOKOCOIN2.value: 2, c.Environments.SOKOCOIN3.value: 3}
    env = factory.get_environment_obj('side_effects_sokoban', noops=True, level=env_name_lvl_dict[env_name])

    # Construct the action space.
    action_space: List[int] = list(range(env.action_spec().minimum, env.action_spec().maximum + 1))
    # Explore all states (brute force) or load them from file if this has been done previously.
    
    return env, action_space

def get_best_action(q_table: np.array, state_id: int, actions: List[int]) -> int:
    # Get the best action according to the q-values for every possible action in the current state.
    max_indices: np.array = np.argwhere(q_table[state_id, :] == np.amax(q_table[state_id, :])).flatten().tolist()
    # Among all best actions, chose randomly.
    return actions[np.random.choice(max_indices)]

def get_greedy_policy_action(eps: float, state_id: int, action_space: List[int], q_table: np.array) -> int:
    # Determine action. Based on the exploration strategy (random or using the q-values).            
    if np.random.random() < eps:
        return np.random.choice(action_space)
    else:
        return get_best_action(q_table, state_id, action_space)

def get_annealed_epsilons(nr_episodes: int) -> np.array:
    # For the first 9/10 episodes reduce the exploration rate linearly to zero.
    nr_exploration = int(nr_episodes * 0.9)
    exp_strategy_linear_decrease: np.array[float]   = np.arange(1.0, 0, -1.0 / nr_exploration)
    # For the last 1/10 episodes, keep the exploration at zero.
    exp_strategy_zero: np.array[float]              = np.zeros(nr_episodes - nr_exploration)
    return np.concatenate((exp_strategy_linear_decrease, exp_strategy_zero))

def run_agent_on_env(results_path: str, env_name: str, live_prints: bool = False, print_q_table: bool = False) -> np.array:    
    # Load the environment.    
    env, action_space = load_sokocoin_env(env_name)
    # Load the q-table.
    q_table: np.array = np.load(f"{results_path}/{c.fn_qtable_npy}", allow_pickle=True)
    # Load the states_dict.
    states_dict: Dict[str, int] = np.load(f"{results_path}/{c.fn_states_dict_npy}", allow_pickle=True).item()
    step_ctr: int = 0
    
    # Run the agent.
    timestep = env.reset()
    state: str = str(timestep.observation['board'])
    states_list: List[str] = [state]
    state_id: int = states_dict.get(state)    
    if live_prints:
        print(f"Step {step_ctr}: Starting in state '{state_id}':\n{np.around(q_table[state_id, :], 2)}\n{state}")
        
    while not timestep.last() and step_ctr < 1000:
        # Get the best action when being in this state - according to the q-table.
        action = get_best_action(q_table, state_id, action_space)
        timestep = env.step(action)
        step_ctr += 1
        state: str = str(timestep.observation['board'])
        # Check if the state is already known. Otherwise assign in a free 'state_id'.
        state_id = states_dict.get(state)

        # Save the new state.
        if live_prints:
            print(f"Step {step_ctr}: Taking action '{c.ACTIONS.get(action)}' ('{action}') to go to state '{state_id}':\n{np.around(q_table[state_id, :], 2)}\n{state}")
            time.sleep(1)
        states_list.append(state)

    if print_q_table:
        print(np.around(q_table, 2))
    # Save the list of explores states to file.
    np.save(f"{results_path}/{c.fn_agent_journey}{step_ctr}.npy", states_list)
    
### DATA PROCESSING FUNCTIONS

def _smooth(values, window=100):
  return values.rolling(window,).mean()

def add_smoothed_data(df, window=100, keys: List[str] = ['episode', 'reward', 'performance', 'loss']):
  smoothed_data = df[keys]
  keys_smooth_names = dict([(k, f"{k}_smooth") for k in keys])
  smoothed_data = smoothed_data.apply(_smooth, window=window).rename(columns=keys_smooth_names)
  temp = pd.concat([df, smoothed_data], axis=1)
  return temp

### PRINT FUNCTIONS

def print_states_dict(states_dict: Dict[str, int]) -> None:
    for s, nr in states_dict.items():
        print(f"{s} \tNr:{nr}")

### SAVE TO FILE FUNCTIONS

def generate_dir_name(settings: Dict[c.PARAMETRS, str]) -> str:
    # Extract the parameter settings.
    nr_episodes: int        = settings.get(c.PARAMETRS.NR_EPISODES)    
    learning_rate: float    = settings.get(c.PARAMETRS.LEARNING_RATE)
    strategy: str           = settings.get(c.PARAMETRS.STATE_SPACE_STRATEGY)
    baseline: str           = settings.get(c.PARAMETRS.BASELINE)
    q_discount: float       = settings.get(c.PARAMETRS.Q_DISCOUNT)
    beta: float             = settings.get(c.PARAMETRS.BETA)

    episodes_str        = f"e{nr_episodes}"
    lr_str              = f"_lr{learning_rate}"
    sspace_strategy_str = f"_S{strategy[:3]}"
    baseline_str        = f"_bl{baseline[:4]}" if baseline is not None else ''
    discount_str        = f"_g{q_discount}"
    beta_str            = f"_b{beta}" if beta is not None else ''
    
    return f"{episodes_str}{lr_str}{sspace_strategy_str}{baseline_str}{discount_str}{beta_str}"
    
def save_intermediate_qtables_to_file(settings: Dict[c.PARAMETRS, str], q_table: np.array, episode: int, method_name: str, dir_name_prefix: str = '') -> None:
    # Extract the parameter settings.
    env_name: str = settings.get(c.PARAMETRS.ENV_NAME)
    dir_name: str = f"{dir_name_prefix}_{generate_dir_name(settings).replace('.', '-')}"
    env_path: str = f"{c.RESULTS_DIR}/{env_name}/{method_name}/{dir_name}"
        
    # Create all necessary directories.
    path_names = env_path.split("/")
    for i, _ in enumerate(path_names):
        path = "/".join(path_names[0:i+1])
        if not os.path.exists(path):
            os.mkdir(path)
    
    # Create the path.
    filenname_qtable_e: str = f"{env_path}/{c.fn_qtable_npy.replace('.npy', f'_{episode}.npy')}"
    # Save the q-table to file.
    np.save(filenname_qtable_e, q_table)

def save_results_to_file(settings: Dict[c.PARAMETRS, str], q_table: np.array, states_dict: Dict[str, int], losses: np.array, episodic_returns: np.array, episodic_performances: np.array, evaluated_episodes: np.array, seed: int, method_name: str, complete_runtime:float, coverage_table: np.array=None, dir_name_prefix: str = '') -> Tuple[str, str]:
    # Extract the parameter settings.
    env_name: str           = settings.get(c.PARAMETRS.ENV_NAME)
    nr_episodes: int        = settings.get(c.PARAMETRS.NR_EPISODES)
    # Learning rate (alpha).
    learning_rate: float    = settings.get(c.PARAMETRS.LEARNING_RATE)
    strategy: str           = settings.get(c.PARAMETRS.STATE_SPACE_STRATEGY)
    baseline: str           = settings.get(c.PARAMETRS.BASELINE)
    q_discount: float       = settings.get(c.PARAMETRS.Q_DISCOUNT)
    beta: float             = settings.get(c.PARAMETRS.BETA)
    
    dir_name: str = f"{dir_name_prefix}_{generate_dir_name(settings).replace('.', '-')}"
    storage_path: str = f"{c.RESULTS_DIR}/{env_name}/{method_name}/{dir_name}"
    
    # Create all necessary directories.
    path_names = storage_path.split("/")
    for i, _ in enumerate(path_names):
        path = "/".join(path_names[0:i+1])
        if not os.path.exists(path):
            os.mkdir(path)
    
    # Create the paths.
    filenname_qtable: str               = f"{storage_path}/{c.fn_qtable_npy}"
    filenname_states_dict: str          = f"{storage_path}/{c.fn_states_dict_npy}"
    filenname_coverage_table: str       = f"{storage_path}/{c.fn_ctable_npy}"
    filenname_general: str              = f"{storage_path}/{c.fn_general_txt}"
    filenname_perf: str                 = f"{storage_path}/{c.fn_performances_table}{seed}.csv"
    filenname_perf_plot: str            = f"{storage_path}/{c.fn_plot1_performance_jpeg}"
    filenname_results_plot: str         = f"{storage_path}/{c.fn_plot2_results_jpeg}"
    filenname_smooth_results_plot: str  = f"{storage_path}/{c.fn_plot3_results_smooth_jpeg}"
    
    # Save the q-table to file.
    np.save(filenname_qtable, q_table)    
    # Save the states-id-dictionary to file.
    np.save(filenname_states_dict, states_dict)
    # Save the coverage-table to file.
    if coverage_table is not None: 
        np.save(filenname_coverage_table, coverage_table)
    # Save general information, including the runtime to file.    
    general_df = pd.DataFrame({'Method': [method_name],
                               'Runtime': [timedelta(seconds=complete_runtime)]})
    general_df.to_csv(filenname_general, index=None)
    
    # Save the perfomances to file.
    d = {'reward': episodic_returns, 'performance': episodic_performances, 'loss': losses, 'episode': evaluated_episodes}
    results_df = pd.DataFrame(d) 
    results_df_with_smooth = add_smoothed_data(results_df)    
    results_df_with_smooth.to_csv(filenname_perf)

    # Prepare a subtitle containing the parameter settings.
    lr_str              = f"Learning rate: {learning_rate}"
    sset_strategy_str   = f", State set space strategy '{strategy}'"
    baseline_str        = f", Baseline: '{baseline}'" if beta is not None else ''
    discount_str        = f", Discount factor: {q_discount}"
    beta_str            = f", Beta: {beta}" if beta is not None else ''
    sub_title: str = f"<br><sup>{lr_str}{sset_strategy_str}{baseline_str}{discount_str}{beta_str} \
        <br>Last performance: {episodic_performances[-1]}, Last reward: {episodic_returns[-1]}</sup>"
    
    # Plot the performance data and store it to image.
    title: str = f"Performances - '{env_name}'\n{sub_title}"
    fig = px.line(results_df, x='episode', y=['reward', 'performance'], title=title)
    fig.write_image(filenname_perf_plot)
    
    # Standardize the data and plot it.
    cols_to_standardize = ['reward', 'performance', 'loss', 'reward_smooth', 'performance_smooth', 'loss_smooth']
    results_df_with_smooth[cols_to_standardize] = results_df_with_smooth[cols_to_standardize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Plot the standardized performance data and store it to image.
    title: str = f"Standardized results - '{env_name}'\n{sub_title}"
    fig = px.line(results_df_with_smooth, x='episode', y=['reward', 'performance', 'loss'], title=title)
    fig.write_image(filenname_results_plot)

    # Plot the standardized smoothed performance data and store it to image.
    title: str = f"Smoothed results - '{env_name}'\n{sub_title}"
    fig = px.line(results_df_with_smooth, x='episode_smooth', y=['reward_smooth', 'performance_smooth', 'loss_smooth'], title=title)
    fig.write_image(filenname_smooth_results_plot)

    # Write a line in the global experiment log. 
    # If it does not contain a header yet, write it as well.
    all_experiments_file_path: str = f"{c.fn_experiments_csv}"    
    headder_row: str = [
        'Environment name', 
        'Nr. episodes',
        'Learning rate',
        'Strategy',
        'Baseline',
        'Q-Discount',
        'Beta',
        'Last reward',
        'Last performances',
        'Storage path'
    ]
    csv_row: Dict[str, str] = {
        # Parameter settings:
        headder_row[0]: env_name,
        headder_row[1]: nr_episodes,
        headder_row[2]: learning_rate,
        headder_row[3]: strategy,        
        headder_row[4]: baseline,
        headder_row[5]: q_discount,
        headder_row[6]: beta,
        # Results:
        headder_row[7]: episodic_returns[-1],
        headder_row[8]: episodic_performances[-1],
        # Relative storage path:
        headder_row[9]: storage_path
    }
    # Open our existing CSV file in append mode.
    # Create a file object for this file.
    existed_already: bool = os.path.exists(all_experiments_file_path)
    with open(all_experiments_file_path, 'a') as f_object:
        writer = DictWriter(f_object, fieldnames=headder_row)
        if not existed_already:
            writer.writeheader()
        # Write the data line.
        writer.writerow(csv_row)
        f_object.close()
    
    print("Saving to file complete.\n\n")
    return filenname_qtable, filenname_perf


if __name__ == "__main__":    
    # path0 = '/home/fabrice/Documents/coding/ML/Results/sokocoin0/RRLearning/2023_07_24-17_23_e100_b0-1_blStepwise_SEs'
    path0 = '/home/fabrice/Documents/coding/ML/Results/sokocoin0/RRLearning/2023_07_25-16_26_e1000_lr0-1_SEst_blStep_g0-99_b0-05'
    # path1 = '/home/fabrice/Documents/coding/ML/Results/sokocoin2/QLearning/2023_07_24-17_33_e100_bNone_SEx'
    
    run_agent_on_env(results_path=path0, env_name="sokocoin0", live_prints=True, print_q_table=True)
    
    pass