from typing import List, Dict, Set, Tuple
from helpers import factory
import numpy as np
import pandas as pd
import os, time
import collections
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
import plotly.express as px


MAX_NR_ACTIONS: int = 50

def _smooth(values, window=100):
  return values.rolling(window,).mean()


def add_smoothed_data(df, window=100):
  smoothed_data = df[['reward', 'performance']]
  smoothed_data = smoothed_data.apply(_smooth, window=window).rename(columns={
      'performance': 'performance_smooth', 'reward': 'reward_smooth'})
  temp = pd.concat([df, smoothed_data], axis=1)
  return temp


# Todo: figure out hot to plot envs
# f, ax = plt.subplots()
# ax.imshow(np.moveaxis(timestep.observation['RGB'], 0, -1), animated=False)
# plt.show()

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


def preprocessing_explore_all_states(env, action_space: List[int], env_name: str) -> Dict[str, int]:
    
    # Visit all possible states.
    def explore(env, timestep, actions_so_far: List[int]=[]):        
        board_str: str = str(timestep.observation['board'])
        
        if board_str not in states_set:
            states_set.add(board_str)
            if not (len(states_set) % 50): print(f"\rExplored {len(states_set)} states.", end="")
            # print(board_str, end="\n\n")

            # if not timestep.last():                             # timestep = env.step(action)
            if not env._game_over and len(actions_so_far) < MAX_NR_ACTIONS:
                for action in action_space:
                    # Explore all possible steps, after taking the current chosen action.
                    timestep = env.step(action)
                    explore(env, timestep, actions_so_far + [action])
                    # After the depth exploration, reset the environment.
                    timestep = env.reset()
                    # For the continuation of the for-loop, execute the 'steps so far' agin.
                    for action in actions_so_far:
                        if timestep.last():
                            break
                        timestep = env.step(action)

    # If the states have been computed before, load them from file.
    file_dir: str = "AllStates"
    if not os.path.exists(file_dir): os.mkdir(file_dir)
    filenname_states: str   = f"{file_dir}/states_{env_name}.npy"
    filenname_runtime: str  = f"{file_dir}/states_{env_name}_rt.npy"
    
    if os.path.exists(filenname_states):
        structured_array = np.load(filenname_states, allow_pickle=True)
        runtime = np.load(filenname_runtime, allow_pickle=True)
        return structured_array.item()
    else:
        timestep = env.reset()
        states_set: Set = set()

        start_time = time.time()
        explore(env, timestep)
        end_time = time.time()
        elapsed_sec = end_time - start_time
                
        states_int_dict: Dict[str, int] = dict(zip(states_set, range(len(states_set))))
        env.reset()
        print(f"\rExplored {len(states_set)} states, in {timedelta(seconds=elapsed_sec)} seconds", end="")  # Expect > 6000        
        np.save(filenname_runtime, elapsed_sec)
        np.save(filenname_states, states_int_dict)
        return states_int_dict


def env_loader(env_name) -> Tuple:
    # Get environment.
    env_name_lvl_dict = {'sokocoin2': 2, 'sokocoin3': 3}
    # env_name_size_dict  = {'sokocoin2': 72, 'sokocoin3': 100}; state_size = env_name_size_dict[env_name]
    env = factory.get_environment_obj('side_effects_sokoban', noops=True, level=env_name_lvl_dict[env_name])

    # Construct the action space.
    action_space: List[int] = list(range(env.action_spec().minimum, env.action_spec().maximum + 1))
    # Explore all states (brute force) or load them from file if this has been done previously.
    print("Explore or load set of all states", end="")
    states_dict: Dict[str, int] = preprocessing_explore_all_states(env, action_space, env_name)
    print(f" (#{len(states_dict)})")
    return env, action_space, states_dict

# Finished verion 1:
def run_q_learning(env_name='sokocoin2', nr_episodes: int = 9000, seed: int = 42):

    def get_best_action(q_table: np.array, state_id: int) -> int:
        # Get the best action according to the q-values for every possible action in the current state.
        max_indices: np.array = np.argwhere(q_table[state_id, :] == np.amax(q_table[state_id, :])).flatten().tolist()
        # Among all best actions, chose randomly.
        return action_space[np.random.choice(max_indices)]

    def greedy_policy_action(eps: float, state_id: int, action_space: List[int], q_table: np.array) -> int:
        # Determine action. Based on the exploration strategy (random or using the q-values).            
        if np.random.random() < eps:
            return np.random.choice(action_space)
        else:
            return get_best_action(q_table, state_id)

    def save_results_to_file(q_table: np.array, losses: np.array, episodic_returns: np.array, episodic_performances: np.array, evaluated_episodes: np.array, seed: int) -> Tuple[str, str]:
        # Create necessary directories to save perfomance and results
        time_tag: str = datetime.now().strftime("%Y_%m_%d-%H_%M")
        results_dir: str = "Results"
        if not os.path.exists(results_dir): os.mkdir(results_dir)

        # Create the paths.
        filenname_qtable: str       = f"{results_dir}/{time_tag}_{env_name}_qtable.npy"
        filenname_performance: str  = f"{results_dir}/{time_tag}_{env_name}_performance_seed{seed}.csv"
        filenname_performance_plot: str = f"{results_dir}/{time_tag}_{env_name}_performance_plot.jpeg"
        
        # Save the q-table to file.
        np.save(filenname_qtable, q_table)

        # Save the perfomances to file.
        d = {'reward': episodic_returns, 'performance': episodic_performances, 'loss': losses, 'episode': evaluated_episodes}
        results_df = pd.DataFrame(d) 
        results_df_1 = add_smoothed_data(results_df)    
        results_df_1.to_csv(filenname_performance)


        
        # Create line graph using Plotly Express.
        # TODO: Smooth weg.
        # TODO: Check qtable. All zero???
        cols_to_standardize = ['reward', 'performance', 'loss']
        results_df[cols_to_standardize] = results_df[cols_to_standardize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        fig = px.line(results_df, x='episode', y=['reward', 'performance', 'loss'], title=f"Performances - {env_name}")
        
        # fig = px.line()
        # fig.add_trace(px.line(x=evaluated_episodes, y=episodic_returns, name='Returns').data[0])
        # fig.add_trace(px.line(x=evaluated_episodes, y=episodic_performances, name='Performances').data[0])
        # fig.add_trace(px.line(x=evaluated_episodes, y=losses, name='Accumulated Losses').data[0])
        # # Customize the plot (optional).
        # fig.update_layout(
        #     title = f"Performances - {env_name}",
        #     xaxis_title = "Episodes"
        # )        
        fig.write_image(filenname_performance_plot)
              

        return filenname_qtable, filenname_performance

    # Load the environment.
    env, action_space, states_dict = env_loader(env_name)    
    np.random.seed(seed)
    
    # TODO NEXT TASK: Adapt the q-learning to RR
    # TODO: Probably only needed next for the baseline simulation.
    # Get an environment for the simulation of the baseline state.
    # baseline_env = env = factory.get_environment_obj('side_effects_sokoban', noops=True, movement_reward=movement_reward, goal_reward=goal_reward, wall_reward=side_effect_reward, corner_reward=side_effect_reward, level=env_name_lvl_dict[env_name])

    # Initialize the agent:    
    alpha: float = 0.1
    q_init_value: float = 0.0
    # Time discount/ costs for each time step. Aka 'gamma'.
    discount: float = 0.99

    # Store the Q-table.
    q_table: np.array = q_init_value * np.ones((len(states_dict), len(action_space)))

    # Initialize datastructures for the evaluation.
    # Every 'loss_frequency' episode, save the last episodic returns and performance, and the accumulated loss.
    loss_frequency: int = 100
    eval_episode_ctr: int = 0
    # Save the actual names of the episodes, where a performance snapshot will be taken.
    evaluated_episodes: np.array        = np.zeros(loss_frequency)
    # Save the returns of the evaluated episodes.
    episodic_returns: np.array          = np.zeros(loss_frequency)
    # Save the performances of the evaluated episodes.
    episodic_performances: np.array     = np.zeros(loss_frequency)
    # Save the accumulated losses of the evaluated episodes.
    losses: np.array                    = np.zeros(loss_frequency)
    
    # Initialize the exploration epsilon
    exploration_epsilons: np.array[float] = np.arange(1.0, 0, -1.0 / nr_episodes)
        
    _last_state_id: int = None
    _last_action: int   = None

    # Run training.
    # Record the performance of the agent (run until the time has run out) for 'number_episodes' many episodes.
    for episode in range(nr_episodes):
        if not(episode % (nr_episodes//loss_frequency)): print(f"\rQ-Learning episode {episode}/{nr_episodes} ({round(episode/nr_episodes *100, 0)}%).", end="")
        # Get the initial set of observations from the environment.
        timestep = env.reset()
        # Reset the variables for each episode.
        _last_state_id: int = None
        _last_action: int   = None
        _nr_actions_so_far: int = 0
        _accumulated_loss: float = 0.        
        
        exploration_epsilon: float = exploration_epsilons[episode]

        while True:
            # Perform a step.
            try:
                state_id: int = states_dict[str(timestep.observation['board'])]
            except KeyError as e:
                error_message = f"PRE-PROCESSING WAS INCOMPLETE: Agent encountered a state during q-learning (in episode {episode}/{nr_episodes}), which was not explored in the preprocessing!"
                print(error_message)
                print(f"'Unknown' state:\n{str(timestep.observation['board'])}")
               
            # If this is NOT the initial state, update the q-values.
            # If this was the initial state, we do not have any reference q-values for states/actions before, and thus cannot update anything.
            if _last_state_id is not None:
                reward = timestep.reward
                # Get the best action according to the q-table (trained so far).
                max_action = get_best_action(q_table, state_id)
                # Calculate the q-value delta.
                delta = (reward + discount * q_table[state_id, max_action] - q_table[_last_state_id, _last_action])
                # Update the q-values.
                q_table[_last_state_id, _last_action] += alpha * delta

                # We define the loss as the 'squared temporal difference error'. In this case the delta^2.
                # Accumulate the squared delta for 'loss_frequency' many uniformly-selected episodes.
                if not(episode % (nr_episodes//loss_frequency)):
                    _accumulated_loss += delta**2
            
            # Break condition: If this was the last action, update the q-values for the terminal state one last time.            
            break_condition: bool = timestep.last() or _nr_actions_so_far < MAX_NR_ACTIONS
            if break_condition:
                # Before ending this episode, save the returns and performances.
                if not(episode % (nr_episodes//loss_frequency)):
                    episodic_returns[eval_episode_ctr]       = env.episode_return
                    episodic_performances[eval_episode_ctr]  = env.get_last_performance()
                    losses[eval_episode_ctr]                 = _accumulated_loss
                    evaluated_episodes[eval_episode_ctr]     = episode
                    eval_episode_ctr += 1
                break
            # Otherwise get the next action (according to the greedy exploration policy) and perform the step.
            else: 
                action: int = greedy_policy_action(exploration_epsilon, state_id, action_space, q_table)
                timestep = env.step(action)            
                _nr_actions_so_far += 1

            # Update the floop-variables.
            _last_state_id: int = state_id
            _last_action: int   = action

    save_results_to_file(q_table, losses, episodic_returns, episodic_performances, evaluated_episodes, seed)

    # TODO: Extract parameters & settings
    # TODO: Visualize results
    # TODO: Script to read in constructed q-table - and execute according to it.
    return episodic_returns, episodic_performances

# Finished verion 1:
def run_q_learning_tupleStates(seed=42 , env_name='sokocoin2', nr_episodes: int = 1000):
    np.random.seed(seed)

    # Get environment.
    env_name_lvl_dict = {'sokocoin2': 2, 'sokocoin3': 3}
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
    run_q_learning()
    # run_experiment()