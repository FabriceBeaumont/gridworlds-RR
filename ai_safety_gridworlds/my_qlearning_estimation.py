from typing import List, Dict, Set, Tuple
from helpers import factory
import numpy as np
import time
from enum import Enum
from datetime import timedelta, datetime

import warnings
warnings.filterwarnings("ignore")

# Local imports
import helper_fcts as hf

# Maximum number of actions that can be performed by the agend during the states exploration in the preprocessing step.
MAX_NR_ACTIONS: int = 100

# Set up a counter that will be used to assign state-ids to unknown states.
# This allows to identify matrix rows to individual states. 
STATE_ID_CTR: int = 0

class Baselines(Enum):
    STARTING_STATE_BASELINE: str    = "Starting"
    INACTION_BASELINE: str          = "Inaction"
    STEPWISE_INACTION_BASELINE: str = "Stepwise"

def env_loader(env_name) -> Tuple:
    # Get environment.
    env_name_lvl_dict = {'sokocoin0': 0, 'sokocoin2': 2, 'sokocoin3': 3}
    # env_name_size_dict  = {'sokocoin2': 72, 'sokocoin3': 100}; state_size = env_name_size_dict[env_name]
    env = factory.get_environment_obj('side_effects_sokoban', noops=True, level=env_name_lvl_dict[env_name])

    # Construct the action space.
    action_space: List[int] = list(range(env.action_spec().minimum, env.action_spec().maximum + 1))
    # Explore all states (brute force) or load them from file if this has been done previously.
    
    states_dict: Dict[str, int] = dict()
            
    return env, action_space, states_dict

# Q-Learning implementation
def run_q_learning(env_name: str, states_set_size: int, nr_episodes: int, learning_rate: float = .1, discount_factor: float = .99, set_loss_freq: int = None, seed: int = 42, verbose: bool = False):

    def get_set_state_id(state: str) -> int:
        global STATE_ID_CTR
        
        # If not already saved, add the state to the dictionary
        if states_dict.get(state) is None:
            states_dict[state] = STATE_ID_CTR
            STATE_ID_CTR += 1
                    
        assert STATE_ID_CTR < states_set_size, f"More than {states_set_size} satets needed! (In episode {episode}.)"

        return states_dict.get(state)
    
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

    start_time: float = time.time()
    
    # Load the environment.
    
    if verbose:
        print(f"Allocating a q-table with size {states_set_size} (estimate). \
            We expect to not encounter more different states using {MAX_NR_ACTIONS} steps)")
    env, action_space, states_dict = env_loader(env_name)
    np.random.seed(seed)
        
    # Initialize the agent:
    method_name: str = "QLearning"
    # Learning rate (alpha).
    learning_rate: float = learning_rate
    # Initialization value of the q-learning q-table.
    q_init_value: float = 0.0
    # Time discount/ costs for each time step. Aka 'gamma'.
    discount_factor: float = discount_factor

    # Store the Q-table.
    q_table: np.array = q_init_value * np.ones((states_set_size, len(action_space)))

    # Initialize datastructures for the evaluation.
    # Every 'nr_episodes/loss_frequency' episode, save the last episodic returns and performance, and the accumulated loss.
    loss_frequency: int = nr_episodes
    if set_loss_freq is not None:
        loss_frequency: int = set_loss_freq
        
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
    # For the first 9/10 episodes reduce the exploration rate linearly to zero.
    exp_strategy_linear_decrease: np.array[float]   = np.arange(1.0, 0, -1.0 / (nr_episodes * 0.9))
    # For the last 1/10 episodes, keep the exploration at zero.
    exp_strategy_zero: np.array[float]              = np.zeros(int(nr_episodes * 0.11))
    exploration_epsilons: np.array[float] = np.concatenate((exp_strategy_linear_decrease, exp_strategy_zero))
        
    _last_state_id: int = None
    _last_action: int   = None
    _last_episode_start_time: float = time.time()
    _all_episodes_rt_sum: float = 0.

    # Run training.
    # Record the performance of the agent (run until the time has run out) for 'number_episodes' many episodes.
    for episode in range(nr_episodes):
        # Progress print.
        if not(episode % (nr_episodes//100)): 
            rt_estimation_str: str = ''
            if episode > 0:
                _all_episodes_rt_sum += round(time.time() - _last_episode_start_time, 3)
                _all_episodes_rt_avg = _all_episodes_rt_sum / episode
                _expected_rt = round(_all_episodes_rt_avg * (nr_episodes - episode), 0)
                rt_estimation_str = f"(Avg. runtime so far {timedelta(seconds=_all_episodes_rt_avg)} - Estimated remaining runtime: {timedelta(seconds=_expected_rt)})"
            print(f"\rQ-Learning episode {episode}/{nr_episodes} ({round(episode/nr_episodes *100)}%). {rt_estimation_str}", end="")

        _last_episode_start_time = time.time()
        # Get the initial set of observations from the environment.
        timestep = env.reset()
        # Reset the variables for each episode.
        _current_state: str         = ""
        _last_state: str            = ""
        _last_state_id: int         = None
        _last_action: int           = None
        _nr_actions_so_far: int     = 0
        _actions_so_far: List[int]  = []
        _episode_loss: float        = 0.        
        exploration_epsilon: float  = exploration_epsilons[episode]

        while True:
            # Perform a step.
            _current_state: str = str(timestep.observation['board'])
            # Check if the state is already known. Otherwise assign in a free 'state_id'.
            state_id = get_set_state_id(_current_state)
                      
            # If this is NOT the initial state, update the q-values.
            # If this was the initial state, we do not have any reference q-values for states/actions before, and thus cannot update anything.
            if _last_state_id is not None:
                reward = timestep.reward
                # Get the best action according to the q-table (trained so far).
                max_action = get_best_action(q_table, state_id)
                # Calculate the q-value delta.
                delta = (reward + discount_factor * q_table[state_id, max_action] - q_table[_last_state_id, _last_action])
                # Update the q-values.
                q_table[_last_state_id, _last_action] += learning_rate * delta

                # We define the loss as the 'squared temporal difference error'. In this case the delta^2.
                # Accumulate the squared delta for 'loss_frequency' many uniformly-selected episodes.
                if not(episode % (nr_episodes//loss_frequency)):
                    _step_loss = delta**2
                    _episode_loss += _step_loss
            
            # Break condition: If this was the last action, update the q-values for the terminal state one last time.            
            break_condition: bool = timestep.last() or _nr_actions_so_far >= MAX_NR_ACTIONS
            if break_condition:
                # Before ending this episode, save the returns and performances.
                if not(episode % (nr_episodes//loss_frequency)):
                    episodic_returns[eval_episode_ctr]       = env.episode_return
                    episodic_performances[eval_episode_ctr]  = env.get_last_performance()
                    losses[eval_episode_ctr]                 = _episode_loss
                    evaluated_episodes[eval_episode_ctr]     = episode
                    eval_episode_ctr += 1
                break
            # Otherwise get the next action (according to the greedy exploration policy) and perform the step.
            else: 
                action: int = greedy_policy_action(exploration_epsilon, state_id, action_space, q_table)
                timestep = env.step(action)            
                _nr_actions_so_far += 1
                _actions_so_far.append(action)

            # Update the floop-variables.
            _last_state: str    = _current_state            
            _last_state_id: int = state_id
            _last_action: int   = action

    # Measure the runtime.
    runtime = time.time() - start_time
    # Save the results (including the runtime) to file.
    dir_name: str = f"e{nr_episodes}_d{discount_factor}"
    hf.save_results_to_file(env_name, q_table, losses, episodic_returns, episodic_performances, evaluated_episodes, seed, method_name=method_name,  dir_name=dir_name, complete_runtime=runtime)

    return episodic_returns, episodic_performances

# RR-Learning implementation
def run_rr_learning(env_name: str, states_set_size: int, nr_episodes: int, learning_rate: float = .1, discount_factor: float = .99, beta: float = 10, \
    baseline_setting: Baselines = Baselines.STARTING_STATE_BASELINE, set_loss_freq: int = None, seed: int = 42, verbose: bool = False):

    def get_set_state_id(state: str) -> int:
        global STATE_ID_CTR
        # If not already saved, add the state to the dictionary
        if states_dict.get(state) is None:
            states_dict[state] = STATE_ID_CTR            
            STATE_ID_CTR += 1
                    
        assert STATE_ID_CTR < states_set_size, f"More than {states_set_size} satets needed! (In episode {episode}.)"

        return states_dict.get(state)
        
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

    def get_the_baseline_state_id(_previous_baseline_id: int, _actions_so_far: List[int] = None) -> int:        
        # For the starting state baseline, nothing has to be computed. It is already set.
        if baseline_setting == Baselines.STARTING_STATE_BASELINE:
            _baseline_state_id = _previous_baseline_id

        # For the inaction baseline, get the already simulated state after '_actions_so_far_ctr' many inactions.
        if baseline_setting == Baselines.INACTION_BASELINE:
            _baseline_state_id = inaction_baseline_states[len(_actions_so_far) - 1]

        # For the stepwise inaction baseline, simulate doing the actions as before, but choosing the NOOP action for the last step.
        if baseline_setting == Baselines.STEPWISE_INACTION_BASELINE:                    
            # Up to the last time step, simulate the environment as before.
            env_simulation, _, _, _, _ = env_loader(env_name)
            for a in _actions_so_far[:1]:                        
                env_simulation.step(a)
            # But for the last time step perform the NOOP action.
            timestep = env_simulation.step(action_space[4])
            _baseline_state = str(timestep.observation['board'])
            _baseline_state_id = get_set_state_id(_baseline_state)

        return _baseline_state_id

    start_time: float = time.time()
    
    # Load the environment.
    if verbose:
        print(f"Allocating a q-table with size {states_set_size} (estimate). \
            We expect to not encounter more different states using {MAX_NR_ACTIONS} steps)")    
    env, action_space, states_dict = env_loader(env_name)
    np.random.seed(seed)
    
    # Initialize the agent:
    method_name: str = "RRLearning"
    # Learning rate (alpha).
    learning_rate: float = learning_rate
    # Initialization value of the q-learning q-table.
    q_init_value: float = 0.0
    # Time discount/ costs for each time step. Aka 'gamma'.
    discount: float = discount_factor
    # Coverage discount.
    c_discount: float = 1.0    
    # Create a directory name, where the (intermediate) results will be stored.
    dir_name: str = f"e{nr_episodes}_b{beta}_bl{baseline_setting}"    

    # Store the Q-table.
    q_table: np.array = q_init_value * np.ones((states_set_size, len(action_space)), dtype=float)
    # Store the coverage values (reachability). 
    # Entrz 'ij' gives the coverage of state 'j' when starting from state 'i'. ({From_states}x{To_states})
    coverage_table: np.array = np.eye(states_set_size, dtype=np.float32)

    # Initialize datastructures for the evaluation.
    # Every 'nr_episodes/loss_frequency' episode, save the last episodic returns and performance, and the accumulated loss.
    loss_frequency: int = nr_episodes
    if set_loss_freq is not None:
        loss_frequency: int = set_loss_freq
        
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
    # For the first 9/10 episodes reduce the exploration rate linearly to zero.
    exp_strategy_linear_decrease: np.array[float]   = np.arange(1.0, 0, -1.0 / (nr_episodes * 0.9))
    # For the last 1/10 episodes, keep the exploration at zero.
    exp_strategy_zero: np.array[float]              = np.zeros(int(nr_episodes * 0.11))
    exploration_epsilons: np.array[float] = np.concatenate((exp_strategy_linear_decrease, exp_strategy_zero))
        
    _last_state_id: int = None
    _last_action: int   = None
    _last_episode_start_time: float = time.time()
    _all_episodes_rt_sum: int = 0

    # Prepare the inaction baseline. Therefore, simulate doing nothing (for all avaliable time steps).    
    timestep = env.reset()
    inaction_baseline_states: np.array = None
    if baseline_setting == Baselines.INACTION_BASELINE:
        print(f"Initializing all inaction baseline states by simulating the environment when using the noop-action only (for at most {MAX_NR_ACTIONS} steps)...", end="")
        tmp_states_ids_lst: List[int] = []
        _actions_ctr: int = 0
        # Until no more steps left or the environment terminates, perform the noop-action and save the occurring states.
        while not timestep.last() and _actions_ctr <= MAX_NR_ACTIONS:
            timestep = env.step(action_space[4])
            tmp_id = get_set_state_id(str(timestep.observation['board']))
            tmp_states_ids_lst.append(tmp_id)
            _actions_ctr += 1

        inaction_baseline_states: np.array = np.array(tmp_states_ids_lst, dtype=int)
        print("Completed!")
    timestep = env.reset()
    
    # Run training.
    # Record the performance of the agent (run until the time has run out) for 'number_episodes' many episodes.
    for episode in range(nr_episodes):
        if not(episode % (nr_episodes//100)): 
            rt_estimation_str: str = ''
            if episode > 0:
                _all_episodes_rt_sum += round(time.time() - _last_episode_start_time, 3)
                _all_episodes_rt_avg = _all_episodes_rt_sum / episode
                _expected_rt = round(_all_episodes_rt_avg * (nr_episodes - episode), 0)
                rt_estimation_str = f"(Avg. runtime so far {timedelta(seconds=_all_episodes_rt_avg)} - Estimated remaining runtime: {timedelta(seconds=_expected_rt)})"
            print(f"\rRR-Learning episode {episode}/{nr_episodes} ({round(episode/nr_episodes *100)}%). {rt_estimation_str}", end="")
        _last_episode_start_time = time.time()
        # Get the initial set of observations from the environment.
        timestep = env.reset()
        # Reset the variables for each episode.
        _current_state: str = ""
        _last_state: str = ""
        _current_state_id: int = None
        _last_state_id: int = None
        _last_action: int   = None
        _actions_so_far_ctr: int = 0
        _actions_so_far: List[int] = []
        _episode_loss: float = 0.

        _baseline_state_id: int = None
                
        exploration_epsilon: float = exploration_epsilons[episode]

        while True:
            # Perform a step.
            try:    
                _current_state: str = str(timestep.observation['board'])
                _current_state_id = get_set_state_id(_current_state)
                                
            except KeyError as e:
                error_message = f"PRE-PROCESSING WAS INCOMPLETE: Agent encountered a state during q-learning (in episode {episode}/{nr_episodes}), which was not explored in the preprocessing!"
                print(error_message)
                print(f"'Unknown' state:\n{str(timestep.observation['board'])}")
                hf.print_actions_list(_actions_so_far)
                print(f"Previous state:\n{_last_state}")
                hf.print_states_dict(states_dict)

            # If this is the initial state, save its id as baseline.
            # Thus by default the starting state basline is set. If another baseline-setting is used, it will be overwritten anyways.
            # Note that in this case, we do not have any reference q-values for states/actions before, and thus cannot update anything.
            if _last_state_id is None:
                _baseline_state_id = _current_state_id
            # If this is NOT the initial state, update the q-values.
            else:
                # TODO: TEST THIS BASELINE STUFF, RR computation and coverage updates.                
                _baseline_state_id = get_the_baseline_state_id(_baseline_state_id, _actions_so_far)
                
                # UPDATE REACHABILITIES.
                # Calculate the coverage delta. 'alpha * [c_discount + V(s_old) - V(s_new)]'
                c_delta = c_discount + coverage_table[:, _last_state_id] - coverage_table[:, _current_state_id]
                # Update the q-values. 'V(s_new) = V(s_new) + alpha * [c_discount + V(s_old) - V(s_new)]'
                coverage_table[:, _current_state_id] += learning_rate * c_delta

                # CALCULATE RELATIVE REACHABILITY. - first formula page 7 in paper "Peanalizing.."
                # Compute the absolute reachability of all other states from the current state, compared to from the baseline state.
                diff: np.array = coverage_table[_baseline_state_id, :] - coverage_table[_current_state_id, :]                
                diff[diff<0] = 0
                # Average this reachability (clipped to non-negative values) to get the relative reachability.
                rr: float = np.mean(diff) # TODO: Multiply with state amount error factor?
                
                # UPDATE Q-VALUES.
                reward = timestep.reward - beta * rr
                # Get the best action according to the q-table (trained so far).
                max_action = get_best_action(q_table, _current_state_id)
                # Calculate the q-value delta.
                q_delta = (reward + discount * q_table[_current_state_id, max_action] - q_table[_last_state_id, _last_action])
                # Update the q-values.
                q_table[_last_state_id, _last_action] += learning_rate * q_delta


                # We define the loss as the 'squared temporal difference error'. In this case the delta^2.
                # Accumulate the squared delta for 'loss_frequency' many uniformly-selected episodes.
                if not(episode % (nr_episodes//loss_frequency)):
                    _step_loss = q_delta**2
                    _episode_loss += _step_loss            
            
            # Break condition: If this was the last action, update the q-values for the terminal state one last time.            
            break_condition: bool = timestep.last() or _actions_so_far_ctr >= MAX_NR_ACTIONS
            if break_condition:
                # Before ending this episode, save the returns and performances.
                if not(episode % (nr_episodes//loss_frequency)):
                    episodic_returns[eval_episode_ctr]       = env.episode_return
                    episodic_performances[eval_episode_ctr]  = env.get_last_performance()
                    losses[eval_episode_ctr]                 = _episode_loss
                    evaluated_episodes[eval_episode_ctr]     = episode
                    eval_episode_ctr += 1
                break
            # Otherwise get the next action (according to the greedy exploration policy) and perform the step.
            else: 
                action: int = greedy_policy_action(exploration_epsilon, _current_state_id, action_space, q_table)
                timestep = env.step(action)
                _actions_so_far_ctr += 1
                _actions_so_far.append(action)

            # Update the floop-variables.
            _last_state: str    = _current_state            
            _last_state_id: int = _current_state_id
            _last_action: int   = action

        # Save the intermediate q-tables for further research.
        if not episode % 10000:
            hf.save_intermediate_qtables_to_file(env_name, q_table, episode, method_name, dir_name)

    runtime = time.time() - start_time
    
    hf.save_results_to_file(env_name, q_table, losses, episodic_returns, episodic_performances, evaluated_episodes, seed, method_name=method_name, dir_name=dir_name, complete_runtime=runtime, coverage_table=coverage_table)

    return episodic_returns, episodic_performances


def run_experiments_q_vs_rr(env_names: List[str], env_sizes: List[int], nr_episodes: int, learning_rates: List[float], discount_factors: List[float], betas: List[float], baselines: List[Baselines]):
    for env_name, env_size in zip(env_names, env_sizes):        
        for lr in learning_rates:
            for discount in discount_factors:
                print(f"Current settings: {env_name}, E{nr_episodes}, lr{lr}, discount{discount}")
                run_q_learning(states_set_size=env_size, env_name=env_name, nr_episodes=nr_episodes, learning_rate=lr)

                for beta in betas:
                    for bl in baselines:
                        run_rr_learning(states_set_size=env_size, env_name=env_name, nr_episodes=nr_episodes, learning_rate=lr, beta=beta, baseline_setting=bl)

def experiment1():
    env_names                       = ['sokocoin0', 'sokocoin2']
    env_state_set_size_estimates    = [100, 47648]
    nr_episodes: int                = 10000
    learning_rates: List[float]     = [.1]
    discount_factors: List[float]   = [0.99]
    betas: List[float]              = [0.1, 3, 100]
    baselines: np.array            = np.array([Baselines.STARTING_STATE_BASELINE, Baselines.INACTION_BASELINE, Baselines.STEPWISE_INACTION_BASELINE])

    run_experiments_q_vs_rr(env_names, env_state_set_size_estimates, nr_episodes, learning_rates, discount_factors, betas, baselines)

env_names                       = ['sokocoin0', 'sokocoin2']    # , 'sokocoin3'
env_state_set_size_estimates    = [100, 47648]                  # , 6988675     #TODO: sparse matrix for qtable/cable?
nr_episodes: int                = 10000
learning_rates: List[float]     = [.1, .3, .9]
discount_factors: List[float]   = [0.99, 1.]
base_lines: np.array            = np.array([Baselines.STARTING_STATE_BASELINE, Baselines.INACTION_BASELINE, Baselines.STEPWISE_INACTION_BASELINE])

if __name__ == "__main__":
    experiment1()
    
    # TODOs: 
    # 1. Teste die Konvergenz der Action Values:  Plot the Q Table for multiple episodes, check for convergence
    # 4. Evaluate the constant-learning rate strategy. Therefore run for all Baselines, for beta [0.1, 3, 100] on sokocoin2
    # 5. Evaluate a dynamic -learning rate strategy (simulated annealing - constant change rate from 1.0 to 0.1). Therefore run for all Baselines, for beta [0.1, 3, 100] on sokocoin2