from typing import List, Dict, Set, Tuple
from ai_safety_gridworlds.helpers import factory
import numpy as np
import pandas as pd

import collections
# import safe_grid_gym
# import gym


def _smooth(values, window=100):
  return values.rolling(window,).mean()


def add_smoothed_data(df, groupby='seed', window=100):
  grouped = df.groupby(groupby)[['reward', 'performance']]
  grouped = grouped.apply(_smooth, window=window).rename(columns={
      'performance': 'performance_smooth', 'reward': 'reward_smooth'})
  temp = pd.concat([df, grouped], axis=1)
  return temp


# TODO: Remove this?
def run_experiment(env_name: str = 'side_effects_sokoban'):
    # Hyperparameters.
    alpha: float = 0.5
    gamma: float = 0.6
    epsilon: float = 0.1
    episodes:int = 50000

    # Initialize environment.
    env = factory.get_environment_obj(env_name, noops=True)
    action_space = env.action_spec()    # TODO: Type BoundedArraySpec

    obs_space: Dict = env.observation_spec() # TODO: Convert or use as pandas df??

    # In gridworlds the state space can be a tuple (height, width) or (height, width, channels).
    # Depending on the state representation. Here we assume a simple (height, width) representation.
    grid_height, grid_width = obs_space["board"].shape

    # Initialize Q-table with zeros.
    Q_table = np.zeros((grid_height, grid_width, action_space.maximum.max()+1))

    # TODO: CONTINUE
    for i_episode in range(episodes):
        # Reset the environment.
        state = env.reset()

        for t in range(100):
            # Choose action. Either explore randomly or exploit knowledge from Q-table.
            if np.random.uniform(0, 1) < epsilon:
                # Explore.
                action = action_space.sample()
            else:
                # Exploit.
                action = np.argmax(Q_table[state])

            # Take the action
            next_state, reward, done, info = env.step(action)

            # Update Q-table
            old_value = Q_table[state + (action,)]
            next_max = np.max(Q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            Q_table[state + (action,)] = new_value

            state = next_state

            if done:
                break


def merged_code(seed=42 , env_name='sokocoin2', nr_episodes: int = 1000):
    np.random.seed(seed)

    # Get environment.
    env_name_lvl_dict = {'sokocoin2': 2, 'sokocoin3': 3}
    env_name_size_dict = {'sokocoin2': 72, 'sokocoin3': 100}

    env = factory.get_environment_obj('side_effects_sokoban', noops=True, level=env_name_lvl_dict[env_name])
    state_size = env_name_size_dict[env_name]
    action_space = list(range(env.action_spec().minimum, env.action_spec().maximum + 1))

    # Initialize the environment.    
    start_timestep = env.reset()

    # TODO: Probably only needed next for the agent.
    # Get an environment for the simulation of the baseline state.
    # baseline_env = env = factory.get_environment_obj('side_effects_sokoban', noops=True, movement_reward=movement_reward, goal_reward=goal_reward, wall_reward=side_effect_reward, corner_reward=side_effect_reward, level=env_name_lvl_dict[env_name])

    # Initialize the agent:
    
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
            state: Tuple = tuple(map(tuple, np.copy(timestep.observation['board']))) # TODO: Find a suitable state definition

            # If this is NOT the initial state, update the q-values.
            # If this was the initial state, we do not have any reference q-values for states/actions before, and thus cannot update anything.
            if _current_state is not None:
                reward = timestep.reward
                
                # Get the best action according to the q-values for every possible action in the current state.
                values = [q_tables[(state, action)] for action in action_space]
                max_indices = [i for i, value in enumerate(values) if value == max(values)] # TODO: research how to optimize this code.                
                # Among all best actions, chose randomly.
                max_action = action_space[np.random.choice(max_indices)]

                # Calculate the q-value delta.
                delta = (reward + discount * q_tables[(state, max_action)] - q_tables[(_current_state, _current_action)])
                # TODO: track loss. How to comute the loss in this case.
                
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
        
        if episode % 500 == 0:
            print('Episode', episode)
    
    d = {'reward': episodic_returns, 'performance': episodic_performances,
         'seed': [seed]*nr_episodes, 'episode': range(nr_episodes)}
    results_df = pd.DataFrame(d)
    results_df_1 = add_smoothed_data(results_df)
    file_name = f"Test_{env_name}_{nr_episodes}"
    results_df_1.to_csv(file_name)

    # TODO: Visualize results
    # TODO: Save q-table
    # TODO: Script to read in constructed q-table - and execute according to it.
    # TODO: Continue with RR script.
    return episodic_returns, episodic_performances
    

if __name__ == "__main__":
    merged_code()
    # run_experiment()