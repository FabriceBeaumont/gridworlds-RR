"""This file contains an implementation of the Q-learning algorithm.
    Notably, the q-table is set up as a defaultdict with tuples (state [str], action [Action]) as keys.
"""
from typing import List, Dict, Set, Tuple
import numpy as np
import pandas as pd
import os, time
import collections
from datetime import timedelta, datetime

import plotly.express as px
from helpers import factory

import warnings
warnings.filterwarnings("ignore")

# Local imports
import helper_fcts as hf
import constants as c

# Maximum number of actions that can be performed by the agend during the states exploration in the preprocessing step.
MAX_NR_ACTIONS: int = 100

# Finished verion 1:
def run_q_learning_tupleStates(env_name: str, nr_episodes: int, seed: int =42) -> Tuple[List[float], List[float]]:
    """This function in an implementation of the Q-learning algorithm.
    Notably, the q-table is set up as a defaultdict with tuples (state [str], action [Action]) as keys.

    Args:
        env_name (str): Name of the gridworlds environment.
        nr_episodes (int): Number of learning episodes.
        seed (int, optional): Value to initialize numpy.random. Defaults to 42.

    Returns:
        Tuple[List[float], List[float]]: Lists of episodic returns and performances.
    """
    np.random.seed(seed)

    # Get the environment.
    env_name_lvl_dict = {c.Environments.SOKOCOIN0: 0,c.Environments.SOKOCOIN2: 2, c.Environments.SOKOCOIN3: 3}
    env = factory.get_environment_obj('side_effects_sokoban', noops=True, level=env_name_lvl_dict[env_name])    
    action_space: List[int] = list(range(env.action_spec().minimum, env.action_spec().maximum + 1))
    
    # Initialize the environment.    
    env.reset()
    
    # Setup the learning rate.
    lr_alpha: float = 0.1
    # Set up the initial values of the q-table.
    q_initialisation: float = 0.0
    # Time discount/ costs for each time step. Aka 'gamma'.
    discount: float = 0.99
    # Setup the Q-table.
    q_tables = collections.defaultdict(lambda: q_initialisation)
    # Setup data structures for the returns and evaluation.
    episodic_returns: List[float] = []
    episodic_performances: List[float] = []
    
    # Initialize the exploration epsilon
    exploration_epsilons: np.array[float] = hf.get_annealed_epsilons(nr_episodes)
        
    # Set up training variables.
    _current_state: Tuple   = None
    _current_action: int    = None

    # Run training.
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
                q_tables[(_current_state, _current_action)] += lr_alpha * delta
               
            _current_state = state
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
                q_tables[(_current_state, _current_action)] += lr_alpha * delta
                
                episodic_returns.append(env.episode_return)
                episodic_performances.append(env.get_last_performance())
                break

            _current_action = action
            
        if episode % 500 == 0:
            print('Episode', episode)

    # Collect the returns and performances and save them to file.
    d = {'reward': episodic_returns, 'performance': episodic_performances,
         'seed': [seed]*nr_episodes, 'episode': range(nr_episodes)}
    results_df = pd.DataFrame(d)
    results_df_1 = hf.add_smoothed_data(results_df)
    file_name = f"Test_{env_name}_{nr_episodes}"
    results_df_1.to_csv(file_name)
    return episodic_returns, episodic_performances

if __name__ == "__main__":
    run_q_learning_tupleStates(env_name=c.Environments.SOKOCOIN0, nr_episodes=1000)