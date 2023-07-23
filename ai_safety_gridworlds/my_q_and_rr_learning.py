"""This file contains an implementation of the Q-learning and RR-learning algorithm.
    Notably, the q-table is set up as a np.array. To known how large it needs to be, 
    an estimation of the size of the states set is used.
"""
from typing import List, Dict, Set, Tuple, Any
import warnings
warnings.filterwarnings("ignore")
from datetime import timedelta
import time
import numpy as np
import os 

from helpers import factory

# Local imports
import helper_fcts as hf
from constants import Baselines, Environments, Strategies, PARAMETRS, StateSpaceSizeEstimations, MAX_NR_ACTIONS

class StateSpaceBuilder():
    
    def __init__(self, env_name: str, strategy: Strategies = Strategies.ESTIMATE_STATES, env = None, action_space = None):
        self.strategy = strategy
        self.env_name: str = env_name
        self.states_dict: Dict[str] = dict()

        # For the estimation strategy: 
        # Set up a counter that will be used to assign state-ids to unknown states.
        # This allows to identify matrix rows to individual states. 
        # Since an estimate is used, the states will be revealed and assigned to an id during the learning itself.
        self._estimation_state_id_ctr: int = 0

        if self.strategy == Strategies.EXPLORE_STATES.value:
            self.states_dict, _, _ = self._explore_states_in_n_steps(env=env, action_space=action_space, env_name=env_name)

    def get_nr_states(self) -> int:
        if self.strategy == Strategies.ESTIMATE_STATES.value:
            return StateSpaceSizeEstimations[self.env_name]
        if self.strategy == Strategies.EXPLORE_STATES.value:
            return len(self.states_dict.keys())

    def get_state_id(self, state: str) -> int:
        # Using the estimate strategy, the dictionary will be filled successively with state-id pairs.
        # In this case the q-table will be initialized using an estimate of the states space.
        # Assert, that the estimation is higher than the actually needed number of states!
        if self.strategy == Strategies.ESTIMATE_STATES.value:
            return self._get_set_state_id(state)
        # Using the exploration strategy, the dictionary has been initialized at class initialization (preprocessing)        
        # In this case the q-table as the exact same size
        # Since the exploration is limited by a number of steps, we can only assert, that all required states have been found already!
        if self.strategy == Strategies.EXPLORE_STATES.value:
            state_id: int = self.states_dict[state]
            assert state_id is not None, f"PRE-PROCESSING WAS INCOMPLETE: Agent encountered an unknown state!"
            return state_id
    
    def _get_set_state_id(self, state: str) -> int:
        # If not already saved, add the state to the dictionary
        if self.states_dict.get(state) is None:
            self.states_dict[state] = self._estimation_state_id_ctr
            self._estimation_state_id_ctr += 1
                    
        assert self._estimation_state_id_ctr < self.get_nr_states(), f"More than {self.get_nr_states()} states needed! The estimation was to low!"        
        return self.states_dict.get(state)
    
    def _explore_states_in_n_steps(self, env, action_space: List[int], env_name: str, allow_loading_and_saving: bool = True) -> Dict[str, int]:
        """This function returns the complete set of attainable states. Either by recursively exploring them (which may be quite costly), or loading it from a file.

        Args:
            env (_type_): Ai savety gridworlds environment.
            action_space (List[int]): List of possible actions. See constants.ACTIONS.
            env_name (str): Name of the environment.
            allow_loading_and_saving (bool, optional): If true, the functions tries to load the set of states first, and only explores them, if no suitable file has been found. Set it to false, in order to explore in any case. Defaults to True.

        Returns:
            Dict[str, int]: A dictionary of states [str] and the minimum of required steps to obtain them from the starting state.
        """
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
                states_actions_dict[board_str] = np.array(actions_so_far, dtype=int)
                
                if not (len(states_steps_dict.keys()) % 50): print(f"\rExploring states ... ({len(states_steps_dict.keys())} so far)", end="")
                
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

        # Set up the (possibly existing) file names and directories.
        file_dir: str = "AllStates"
        if not os.path.exists(file_dir): os.mkdir(file_dir)
        file_path: str = f"{file_dir}/{env_name.value}"
        if not os.path.exists(file_path): os.mkdir(file_path)
        
        # Dict[str, id] - Given an id, return the state.
        filenname_states_id: str    = f"{file_path}/states.npy"
        # Dict[id, str] - Given a state, return its id.
        filenname_id_states: str    = f"{file_path}/states_id_str.npy"
        # Dict[str, actions_list:np.array].    
        filenname_actions: str      = f"{file_path}/actions.npy"    
        filenname_runtime: str      = f"{file_path}/states_rt.npy"

        # If the states have been computed before, load them from file.
        if os.path.exists(filenname_states_id) and allow_loading_and_saving:
            structured_array    = np.load(filenname_states_id,  allow_pickle=True)
            int_states_dict     = np.load(filenname_id_states,  allow_pickle=True)
            states_actions_dict = np.load(filenname_actions,    allow_pickle=True)
            runtime             = np.load(filenname_runtime,    allow_pickle=True)
            
            return structured_array.item(), int_states_dict, states_actions_dict
        else:
            timestep = env.reset()
            states_steps_dict: Dict[str, int] = dict()
            states_actions_dict: Dict[str, np.array] = dict()

            start_time = time.time()        
            states_steps_dict = explore(env, timestep, states_steps_dict)
            end_time = time.time()
            states_set: Set = set(states_steps_dict.keys())
            elapsed_sec = end_time - start_time
                    
            states_int_dict: Dict[str, int] = dict(zip(states_set, range(len(states_set))))
            int_states_dict: Dict[str, int] = dict(zip(range(len(states_set)), states_set))
            
            env.reset()
            print(f"\rExplored {len(states_set)} states, in {timedelta(seconds=elapsed_sec)} seconds", end="")
            if allow_loading_and_saving:
                np.save(filenname_states_id, states_int_dict)            
                np.save(filenname_id_states, int_states_dict)
                np.save(filenname_actions, states_actions_dict)
                np.save(filenname_runtime, elapsed_sec)
                
            return states_int_dict, int_states_dict, states_actions_dict


# Q-Learning implementation.
def run_q_learning(settings: Dict[PARAMETRS, str], set_loss_freq: int = None, seed: int = 42, verbose: bool = False):

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

    # Extract the parameter settings.
    env_name        = settings.get(PARAMETRS.ENV_NAME)
    nr_episodes     = settings.get(PARAMETRS.NR_EPISODES)
    # Learning rate (alpha).
    learning_rate   = settings.get(PARAMETRS.LEARNING_RATE)
    strategy        = settings.get(PARAMETRS.STATE_SET_STRATEGY)
    # Time discount/ costs for each time step. Aka 'gamma'.
    q_discount : float = settings.get(PARAMETRS.Q_DISCOUNT)
        
    # Initialize the agent:
    method_name: str = "QLearning"    
    # Initialization value of the q-learning q-table.
    q_init_value: float = 0.0
    
    print(f">>RUN: {method_name}: env-{env_name}, e{nr_episodes}, lr{learning_rate}, {strategy}, d{q_discount}")    
    
    start_time: float = time.time()    
    # Load the environment.    
    env, action_space = hf.env_loader(env_name)
    states_set = StateSpaceBuilder(env_name=env_name, strategy=strategy, env=env, action_space=action_space)
    np.random.seed(seed)
    
    # Store the Q-table.
    q_table: np.array = q_init_value * np.ones((states_set.get_nr_states(), len(action_space)))

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
    exploration_epsilons: np.array[float] = hf.get_annealed_epsilons(nr_episodes)
        
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
            # Interpret the 0th episode as the first.
            e = episode + 1                
            if episode > 0:
                _all_episodes_rt_sum += round(time.time() - _last_episode_start_time, 3)
                _all_episodes_rt_avg = _all_episodes_rt_sum / e
                _expected_rt = round(_all_episodes_rt_avg * (nr_episodes - e), 0)
                rt_estimation_str = f"(Avg. runtime so far {timedelta(seconds=_all_episodes_rt_avg)} - Estimated remaining runtime: {timedelta(seconds=_expected_rt)})"
            print(f"\r{method_name} episode {e}/{nr_episodes} ({round(e/nr_episodes *100)}%). {rt_estimation_str}", end="")

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
            state_id = states_set.get_state_id(_current_state)
                      
            # If this is NOT the initial state, update the q-values.
            # If this was the initial state, we do not have any reference q-values for states/actions before, and thus cannot update anything.
            if _last_state_id is not None:
                reward = timestep.reward
                # Get the best action according to the q-table (trained so far).
                max_action = get_best_action(q_table, state_id)
                # Calculate the q-value delta.
                delta = (reward + q_discount * q_table[state_id, max_action] - q_table[_last_state_id, _last_action])
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
    # Print line to offset the '\r'-progress prints.
    print()    
    hf.save_results_to_file(settings, q_table, losses, episodic_returns, episodic_performances, evaluated_episodes, seed, method_name=method_name, complete_runtime=runtime)

    return episodic_returns, episodic_performances

# RR-Learning implementation.
def run_rr_learning(settings: Dict[PARAMETRS, str], set_loss_freq: int = None, seed: int = 42, verbose: bool = False, save_coverage_table: bool = True):
        
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
            env_simulation, _ = hf.env_loader(env_name)
            for a in _actions_so_far[:1]:                        
                env_simulation.step(a)
            # But for the last time step perform the NOOP action.
            timestep = env_simulation.step(action_space[4])
            _baseline_state = str(timestep.observation['board'])
            _baseline_state_id = states_set.get_state_id(_baseline_state)

        return _baseline_state_id

    # Extract the parameter settings.
    env_name: str           = settings.get(PARAMETRS.ENV_NAME)
    nr_episodes: int        = settings.get(PARAMETRS.NR_EPISODES)
    # Learning rate (alpha).
    learning_rate: float    = settings.get(PARAMETRS.LEARNING_RATE)
    strategy: str           = settings.get(PARAMETRS.STATE_SET_STRATEGY)
    # Time discount/ costs for each time step. Aka 'gamma'.    
    q_discount: float       = settings.get(PARAMETRS.Q_DISCOUNT)
    baseline: str           = settings.get(PARAMETRS.BASELINE)
    beta: float             = settings.get(PARAMETRS.BETA)    
    
    # Initialize the agent:
    method_name: str = "RRLearning"    
    # Initialization value of the q-learning q-table.
    q_init_value: float = 0.0
    # Coverage discount.
    c_discount: float = 1.0
    
    start_time: float = time.time()
    
    # Load the environment.
    env, action_space = hf.env_loader(env_name)
    states_set = StateSpaceBuilder(env_name=env_name, strategy=strategy, env=env, action_space=action_space)
    np.random.seed(seed)
    
    print(f">>RUN: {method_name}: env-{env_name}, e{nr_episodes}, lr{learning_rate}, {strategy}, d{q_discount}, bl-{baseline}, b{beta}")
    
    # Store the Q-table.
    q_table: np.array = q_init_value * np.ones((states_set.get_nr_states(), len(action_space)), dtype=float)
    # Store the coverage values (reachability). 
    # Entrz 'ij' gives the coverage of state 'j' when starting from state 'i'. ({From_states}x{To_states})
    states_set_size_estimate = StateSpaceSizeEstimations[env_name]
    c_table: np.array = np.eye(states_set_size_estimate, dtype=np.float32)

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
    exploration_epsilons: np.array[float] = hf.get_annealed_epsilons(nr_episodes)
        
    _last_state_id: int = None
    _last_action: int   = None
    _last_episode_start_time: float = time.time()
    _all_episodes_rt_sum: int = 0

    # Prepare the inaction baseline. Therefore, simulate doing nothing (for all avaliable time steps).    
    timestep = env.reset()
    inaction_baseline_states: np.array = None
    if baseline == Baselines.INACTION_BASELINE.value:
        print(f"Initializing all inaction baseline states by simulating the environment when using the noop-action only (for at most {MAX_NR_ACTIONS} steps)...", end="")
        tmp_states_ids_lst: List[int] = []
        _actions_ctr: int = 0
        # Until no more steps left or the environment terminates, perform the noop-action and save the occurring states.
        while not timestep.last() and _actions_ctr <= MAX_NR_ACTIONS:
            timestep = env.step(action_space[4])
            tmp_id = states_set.get_state_id(str(timestep.observation['board']))
            tmp_states_ids_lst.append(tmp_id)
            _actions_ctr += 1

        inaction_baseline_states: np.array = np.array(tmp_states_ids_lst, dtype=int)
        print("Completed!")
    timestep = env.reset()
    
    # Run training.
    # Record the performance of the agent (run until the time has run out) for 'number_episodes' many episodes.
    for episode in range(nr_episodes):
        # Progress print.
        if not(episode % (nr_episodes//100)): 
            rt_estimation_str: str = ''
            # Interpret the 0th episode as the first.
            e = episode + 1
            if episode > 0:
                _all_episodes_rt_sum += round(time.time() - _last_episode_start_time, 3)
                _all_episodes_rt_avg = _all_episodes_rt_sum / e
                _expected_rt = round(_all_episodes_rt_avg * (nr_episodes - e), 0)
                rt_estimation_str = f"(Avg. runtime so far {timedelta(seconds=_all_episodes_rt_avg)} - Estimated remaining runtime: {timedelta(seconds=_expected_rt)})"
            print(f"\r{method_name} episode {e}/{nr_episodes} ({round(e/nr_episodes *100)}%). {rt_estimation_str}", end="")
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
            _current_state: str = str(timestep.observation['board'])
            _current_state_id = states_set.get_state_id(_current_state)       

            # If this is the initial state, save its id as baseline.
            # Thus by default the starting state basline is set. If another baseline-setting is used, it will be overwritten anyways.
            # Note that in this case, we do not have any reference q-values for states/actions before, and thus cannot update anything.
            if _last_state_id is None:
                _baseline_state_id = _current_state_id
            # If this is NOT the initial state, update the q-values.
            else:
                _baseline_state_id = get_the_baseline_state_id(_baseline_state_id, _actions_so_far)
                
                # UPDATE REACHABILITIES.
                # Calculate the coverage delta. 'alpha * [c_discount + V(s_old) - V(s_new)]'
                c_delta = c_discount + c_table[:, _last_state_id] - c_table[:, _current_state_id]
                # Update the q-values. 'V(s_new) = V(s_new) + alpha * [c_discount + V(s_old) - V(s_new)]'
                c_table[:, _current_state_id] += learning_rate * c_delta

                # CALCULATE RELATIVE REACHABILITY. - first formula page 7 in paper "Peanalizing.."
                # Compute the absolute reachability of all other states from the current state, compared to from the baseline state.
                diff: np.array = c_table[_baseline_state_id, :] - c_table[_current_state_id, :]                
                diff[diff<0] = 0
                # Average this reachability (clipped to non-negative values) to get the relative reachability.
                rr: float = np.mean(diff) # TODO: Multiply with state amount error factor?
                
                # UPDATE Q-VALUES.
                reward = timestep.reward - beta * rr
                # Get the best action according to the q-table (trained so far).
                max_action = get_best_action(q_table, _current_state_id)
                # Calculate the q-value delta.
                q_delta = (reward + q_discount * q_table[_current_state_id, max_action] - q_table[_last_state_id, _last_action])
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
        if episode % 1000 == 0:
            hf.save_intermediate_qtables_to_file(settings, q_table, episode, method_name)

    runtime = time.time() - start_time
    # Print line to offset the '\r'-progress prints.
    print()
    if not save_coverage_table:
        c_table = None
    hf.save_results_to_file(settings, q_table, losses, episodic_returns, episodic_performances, evaluated_episodes, seed, method_name=method_name, complete_runtime=runtime, coverage_table=c_table)

    return episodic_returns, episodic_performances

def create_settings_dict(env_name: str, nr_episodes: int, learning_rate: float = None, state_set_strategy: str = None, q_discount: float = None, baseline: str = None, beta: float = None) -> Dict[PARAMETRS, str]:
    settings: Dict[PARAMETRS, Any] = dict()

    settings[PARAMETRS.ENV_NAME]            = env_name
    settings[PARAMETRS.NR_EPISODES]         = nr_episodes
    settings[PARAMETRS.LEARNING_RATE]       = learning_rate
    settings[PARAMETRS.STATE_SET_STRATEGY]  = state_set_strategy
    settings[PARAMETRS.BASELINE]            = baseline
    settings[PARAMETRS.Q_DISCOUNT]          = q_discount
    settings[PARAMETRS.BETA]                = beta

    return settings

# EXPERIMENTS
def run_experiments_q_vs_rr(env_names: List[Baselines], nr_episodes: int, learning_rates: List[float], discount_factors: List[float], betas: List[float], baselines: List[Baselines], strategy: Strategies = Strategies.ESTIMATE_STATES, save_coverage_table: bool = True):
    print()
    nr_experiments: int = len(env_names) * len(learning_rates) * len(discount_factors)
    ctr: int = 1
    
    for n in env_names:
        for lr in learning_rates:
            for d in discount_factors:
                print(f"Experiment {ctr}/{nr_experiments}: {n.value}, e{nr_episodes}, lr{lr}, discount{d}, state set-{strategy.value}")
                settings = create_settings_dict(env_name=n.value, nr_episodes=nr_episodes, learning_rate=lr, state_set_strategy=strategy.value, q_discount=d)
                run_q_learning(settings)
                
                for beta in betas:
                    for bl in baselines:
                        settings = create_settings_dict(env_name=n.value, nr_episodes=nr_episodes, learning_rate=lr, state_set_strategy=strategy.value, q_discount=d, baseline=bl.value, beta=beta)
                        run_rr_learning(settings, save_coverage_table)

def demo():
    env_names                       = [Environments.SOKOCOIN0, Environments.SOKOCOIN2]
    nr_episodes: int                = 100
    learning_rates: List[float]     = [.1]
    discount_factors: List[float]   = [0.99]
    betas: List[float]              = [0.1]
    baselines: np.array             = np.array([Baselines.STARTING_STATE_BASELINE, Baselines.INACTION_BASELINE, Baselines.STEPWISE_INACTION_BASELINE])
    
    # Perform the learning using an estimation of the state space size.
    run_experiments_q_vs_rr(env_names, nr_episodes, learning_rates, discount_factors, betas, baselines, strategy=Strategies.ESTIMATE_STATES, save_coverage_table=False)
    # Perform the learning using an preprocessed exploration of the state space size.
    run_experiments_q_vs_rr(env_names, nr_episodes, learning_rates, discount_factors, betas, baselines, strategy=Strategies.EXPLORE_STATES, save_coverage_table=False)
    

def experiment1_base_estimate():
    env_names                       = [Environments.SOKOCOIN0.value, Environments.SOKOCOIN2]
    nr_episodes: int                = 10000
    learning_rates: List[float]     = [.1]
    discount_factors: List[float]   = [0.99]
    betas: List[float]              = [0.1, 3, 100]
    baselines: np.array             = np.array([Baselines.STARTING_STATE_BASELINE, Baselines.INACTION_BASELINE, Baselines.STEPWISE_INACTION_BASELINE])
        
    run_experiments_q_vs_rr(env_names, nr_episodes, learning_rates, discount_factors, betas, baselines, strategy=Strategies.ESTIMATE_STATES)


def experiment2_base_estimate():
    env_names                       = [Environments.SOKOCOIN0.value, Environments.SOKOCOIN2]
    nr_episodes: int                = 10000
    learning_rates: List[float]     = [.1]
    discount_factors: List[float]   = [0.99]
    betas: List[float]              = [0.1, 3, 100]
    baselines: np.array             = np.array([Baselines.STARTING_STATE_BASELINE, Baselines.INACTION_BASELINE, Baselines.STEPWISE_INACTION_BASELINE])
        
    run_experiments_q_vs_rr(env_names, nr_episodes, learning_rates, discount_factors, betas, baselines, strategy=Strategies.EXPLORE_STATES)

if __name__ == "__main__":
    demo()
    
    # TODOs: 
    # X. check qtable heatmaps # 1. Teste die Konvergenz der Action Values:  Plot the Q Table for multiple episodes, check for convergence
    # X. Debugging - check Baseline Compttation and qual.




    
    # 4. Evaluate the constant-learning rate strategy. Therefore run for all Baselines, for beta [0.1, 3, 100] on sokocoin2
    # 5. Evaluate a dynamic -learning rate strategy (simulated annealing - constant change rate from 1.0 to 0.1). Therefore run for all Baselines, for beta [0.1, 3, 100] on sokocoin2