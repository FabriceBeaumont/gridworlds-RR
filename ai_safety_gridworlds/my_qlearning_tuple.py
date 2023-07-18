from typing import List, Dict, Set, Tuple
from helpers import factory
import numpy as np
import pandas as pd
import os, time
import collections
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
import plotly.express as px
from enum import Enum

import multiprocessing
from functools import partial

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

# Maximum number of actions that can be performed by the agend during the states exploration in the preprocessing step.
MAX_NR_ACTIONS: int = 100

def _smooth(values, window=100):
  return values.rolling(window,).mean()


def add_smoothed_data(df, window=100, keys: List[str] = ['episode', 'reward', 'performance', 'loss']):
  smoothed_data = df[keys]
  keys_smooth_names = dict([(k, f"{k}_smooth") for k in keys])
  smoothed_data = smoothed_data.apply(_smooth, window=window).rename(columns=keys_smooth_names)
  temp = pd.concat([df, smoothed_data], axis=1)
  return temp


def print_actions_list(actions: List[int]) -> None:
    for a in actions:
        print(f"{ACTIONS[a]}, ", end="")

    print()

def print_states_dict(states_dict: Dict[str, int]) -> None:
    for s, nr in states_dict.items():
        print(f"{s} \tNr:{nr}")


# TODO: Extract parameters & settings
# TODO: Visualize results
# Todo: figure out how to plot envs
# f, ax = plt.subplots()
# ax.imshow(np.moveaxis(timestep.observation['RGB'], 0, -1), animated=False)
# plt.show()


def preprocessing_explore_all_states(env, action_space: List[int], env_name: str, allow_loading_and_saving: bool = True) -> Dict[str, int]:   
    # Visit all possible states.
    def explore(env, timestep, states_steps_dict: Dict[str, int], actions_so_far: List[int]=[]) -> Dict[str, int]:
        # NOTE: Running n-times against the wall and that the n+1 state the env changes. The exploration is greedy for NEW states.
        board_str: str = str(timestep.observation['board'])

        last_needed_nr_steps: int = states_steps_dict.get(board_str)
        # Continue, if the state is new (e.a. if the dictionary-get returned None).
        # Continue AS WELL, if the state is not new, but has now been reached with fewer steps!
        # (Otherwise reaching a state with inefficient steps prohibits further realistic exploration).
        if last_needed_nr_steps is None or last_needed_nr_steps > len(actions_so_far):
            states_steps_dict[board_str] = len(actions_so_far)
            states_actions_dict[board_str] = np.array(actions_so_far, dtype=int)                           # DEBUGGING DATA
            
            if not (len(states_steps_dict.keys()) % 50): print(f"\rExplored {len(states_steps_dict.keys())} states.", end="")

            # if not env._game_over and len(actions_so_far) < MAX_NR_ACTIONS:
            if not timestep.last() and len(actions_so_far) < MAX_NR_ACTIONS:
                for action in action_space:
                    # Explore all possible steps, after taking the current chosen action.
                    timestep = env.step(action)
                    states_steps_dict = explore(env, timestep, states_steps_dict, actions_so_far + [action])
                    # After the depth exploration, reset the environment, since it is probably changed in the recursion.
                    timestep = env.reset()
                    # Redo the action to have the same environment as before the recursion call.
                    for action in actions_so_far:
                        timestep = env.step(action)
        return states_steps_dict

    # If the states have been computed before, load them from file.
    file_dir: str = "AllStates"
    if not os.path.exists(file_dir): os.mkdir(file_dir)
    file_path: str = f"{file_dir}/{env_name}"
    if not os.path.exists(file_path): os.mkdir(file_path)
    
    filenname_states: str   = f"{file_path}/states.npy"
    filenname_runtime: str  = f"{file_path}/states_rt.npy"

    # DEBUGGING DATA:
    # Dict[id, str] - States backwards.
    filenname_states_reverse: str   = f"{file_path}/states_id_str.npy"        # DEBUGGING DATA:
    # Dict[str, actions_list:np.array].    
    filenname_actions: str   = f"{file_path}/actions.npy"                     # DEBUGGING DATA:
        
    if os.path.exists(filenname_states) and allow_loading_and_saving:
        structured_array = np.load(filenname_states, allow_pickle=True)
        runtime = np.load(filenname_runtime, allow_pickle=True)

        #TODO: Uncomment if initialization was complete
        int_states_dict = None #np.load(filenname_states_reverse)                        # DEBUGGING DATA:
        states_actions_dict = None #np.load(filenname_actions)                           # DEBUGGING DATA:
        
        return structured_array.item(), int_states_dict, states_actions_dict
    else:
        timestep = env.reset()
        states_steps_dict: Dict[str, int] = dict()
        states_actions_dict: Dict[str, np.array] = dict()                               # DEBUGGING DATA:

        start_time = time.time()        
        states_steps_dict = explore(env, timestep, states_steps_dict)
        end_time = time.time()
        states_set: Set = set(states_steps_dict.keys())
        elapsed_sec = end_time - start_time
                
        states_int_dict: Dict[str, int] = dict(zip(states_set, range(len(states_set)))) # DEBUGGING DATA:
        int_states_dict: Dict[str, int] = dict(zip(range(len(states_set)), states_set)) # DEBUGGING DATA:
        
        env.reset()
        print(f"\rExplored {len(states_set)} states, in {timedelta(seconds=elapsed_sec)} seconds", end="")
        if allow_loading_and_saving:
            np.save(filenname_runtime, elapsed_sec)
            np.save(filenname_states, states_int_dict)
            
            np.save(filenname_states_reverse, int_states_dict)                          # DEBUGGING DATA:
            np.save(filenname_actions, states_actions_dict)                             # DEBUGGING DATA:
            
        return states_int_dict, int_states_dict, states_actions_dict


def env_loader(env_name, verbose: bool=False) -> Tuple:
    # Get environment.
    env_name_lvl_dict = {'sokocoin0': 0, 'sokocoin2': 2, 'sokocoin3': 3}
    # env_name_size_dict  = {'sokocoin2': 72, 'sokocoin3': 100}; state_size = env_name_size_dict[env_name]
    env = factory.get_environment_obj('side_effects_sokoban', noops=True, level=env_name_lvl_dict[env_name])

    # Construct the action space.
    action_space: List[int] = list(range(env.action_spec().minimum, env.action_spec().maximum + 1))
    # Explore all states (brute force) or load them from file if this has been done previously.
    if verbose:
        print("\nExplore or load set of all states.", end="")
    a, i, s = preprocessing_explore_all_states(env, action_space, env_name, allow_loading_and_saving=True)
    states_dict: Dict[str, int] = a
    int_states_dict: Dict[int, str] = i                                                 # DEBUGGING DATA:
    states_actions_dict: Dict[str, np.array] = s                                        # DEBUGGING DATA:
    
    if verbose:
        print(f" (#{len(states_dict)} states, using {MAX_NR_ACTIONS} steps)")
    return env, action_space, states_dict, int_states_dict, states_actions_dict


def save_results_to_file(env_name: str, q_table: np.array, losses: np.array, episodic_returns: np.array, episodic_performances: np.array, evaluated_episodes: np.array, seed: int, method_name: str, dir_name:str, complete_runtime:float, coverage_table: np.array=None) -> Tuple[str, str]:
    # Create necessary directories to save perfomance and results
    time_tag: str = datetime.now().strftime("%Y_%m_%d-%H_%M")
    results_dir: str = "Results"    
    dir_time_tag: str = f"{time_tag}_{str(dir_name).replace('.', '-')}"
    env_path: str = f"{results_dir}/{env_name}/{method_name}/{dir_time_tag}"
    
    # Create all necessary directories.
    path_names = env_path.split("/")
    for i, _ in enumerate(path_names):
        path = "/".join(path_names[0:i+1])
        if not os.path.exists(path):
            os.mkdir(path)
    
    # Create the paths.
    filenname_qtable: str           = f"{env_path}/qtable.npy"
    filenname_coverage_table: str   = f"{env_path}/ctable.npy"
    filenname_general: str          = f"{env_path}/general.txt"
    filenname_perf: str             = f"{env_path}/performances_table_seed{seed}.csv"
    filenname_perf_plot: str        = f"{env_path}/plot1_performance.jpeg"
    filenname_results_plot: str        = f"{env_path}/plot2_results.jpeg"
    filenname_smooth_results_plot: str = f"{env_path}/plot3_results_smooth.jpeg"
    
    # Save the q-table to file.
    np.save(filenname_qtable, q_table)
    # Save the q-table to file.
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

    # Plot the performance data and store it to image.
    fig = px.line(results_df, x='episode', y=['reward', 'performance'], title=f"Performances - {env_name}")    
    fig.write_image(filenname_perf_plot)
    
    # Standardize the data and plot it.
    cols_to_standardize = ['reward', 'performance', 'loss', 'reward_smooth', 'performance_smooth', 'loss_smooth']
    results_df_with_smooth[cols_to_standardize] = results_df_with_smooth[cols_to_standardize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Plot the standardized performance data and store it to image.
    fig = px.line(results_df_with_smooth, x='episode', y=['reward', 'performance', 'loss'], title=f"Standardized results - {env_name}")    
    fig.write_image(filenname_results_plot)

    # Plot the standardized smoothed performance data and store it to image.
    fig = px.line(results_df_with_smooth, x='episode_smooth', y=['reward_smooth', 'performance_smooth', 'loss_smooth'], title=f"Smoothed results - {env_name}")
    fig.write_image(filenname_smooth_results_plot)

    print("Saving to file complete.\n\n")
    return filenname_qtable, filenname_perf

# Finished verion 1:
def run_q_learning_tupleStates(seed=42 , env_name='sokocoin2', nr_episodes: int = 1000):
    np.random.seed(seed)

    # Get environment.
    env_name_lvl_dict = {'sokocoin0': 0,'sokocoin2': 2, 'sokocoin3': 3}
    env_name_size_dict = {'sokocoin2': 72, 'sokocoin3': 100}

    env = factory.get_environment_obj('side_effects_sokoban', noops=True, level=env_name_lvl_dict[env_name])
    state_size = env_name_size_dict[env_name]
    action_space: List[int] = list(range(env.action_spec().minimum, env.action_spec().maximum + 1))
    
    # Initialize the environment.    
    start_timestep = env.reset()
   
    alpha:float = 0.1
    q_initialisation: float =0.0
    # Time discount/ costs for each time step. Aka 'gamma'.
    discount: float = 0.99

    # Store the Q-table.
    q_tables = collections.defaultdict(lambda: q_initialisation)

    # Run training.
    # returns, performances = run_training(agent, env, number_episodes=num_episodes, anneal=anneal)
    episodic_returns: List[float] = []
    episodic_performances: List[float] = []
    # Simulated annealing: decrease the epsilon per episode by 1/nr_episodes.

    # Initialize the exploration epsilon
    exploration_epsilons: np.array[float] = np.arange(1.0, 0, -1.0 / nr_episodes)
        
    _current_state, _current_action = None, None

    # Record the performance of the agent (run until the time has run out) for 'number_episodes' many episodes.
    for episode in range(nr_episodes):
        # Get the initial set of observations from the environment.
        timestep = env.reset()
        # Reset the variables for each episode.
        _current_state, _current_action = None, None
        exploration_epsilon: float = exploration_epsilons[episode]

        while True:
            # Perform a step.
            state: Tuple = tuple(map(tuple, np.copy(timestep.observation['board'])))

            # If this is NOT the initial state, update the q-values.
            # If this was the initial state, we do not have any reference q-values for states/actions before, and thus cannot update anything.
            if _current_state is not None:
                reward = timestep.reward
                
                # Get the best action according to the q-values for every possible action in the current state.
                values = [q_tables[(state, action)] for action in action_space]
                max_indices = [i for i, value in enumerate(values) if value == max(values)]
                # Among all best actions, chose randomly.
                max_action = action_space[np.random.choice(max_indices)]

                # Calculate the q-value delta.
                delta = (reward + discount * q_tables[(state, max_action)] - q_tables[(_current_state, _current_action)])
                
                # Update the q-values.
                q_tables[(_current_state, _current_action)] += alpha * delta
               
            _current_state: Tuple = state
            # Determine action. Based on the exploration strategy (random or using the q-values).            
            if np.random.random() < exploration_epsilon:
                action = np.random.choice(action_space)
            else:
                values = [q_tables[(state, action)] for action in action_space]
                max_indices = [i for i, value in enumerate(values) if value == max(values)]

                action = action_space[np.random.choice(max_indices)]

            timestep = env.step(action)            
            if timestep.last():
                # Update the q-values for the terminal state one last time.
                reward = timestep.reward
                # Calculate the q-value delta.
                delta = (reward + discount * q_tables[(state, max_action)] - q_tables[(_current_state, _current_action)])
                # Update the q-values.
                q_tables[(_current_state, _current_action)] += alpha * delta
                
                episodic_returns.append(env.episode_return)
                episodic_performances.append(env.get_last_performance())
                break

            _current_action = action
            
        if episode % 500 == 0:
            print('Episode', episode)
    
    d = {'reward': episodic_returns, 'performance': episodic_performances,
         'seed': [seed]*nr_episodes, 'episode': range(nr_episodes)}
    results_df = pd.DataFrame(d)
    results_df_1 = add_smoothed_data(results_df)
    file_name = f"Test_{env_name}_{nr_episodes}"
    results_df_1.to_csv(file_name)
    return episodic_returns, episodic_performances

if __name__ == "__main__":
    run_q_learning_tupleStates()