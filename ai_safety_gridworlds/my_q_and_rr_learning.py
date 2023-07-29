"""This file contains an implementation of the Q-learning and RR-learning algorithm.
    Notably, the q-table is set up as a np.array. To known how large it needs to be, 
    an estimation of the size of the states set is used.
"""
from typing import List, Dict, Set, Tuple, Any
import warnings
warnings.filterwarnings("ignore")
from alive_progress import alive_bar

from datetime import timedelta, datetime
import time
import numpy as np
import os 

# Local imports
import helper_fcts as hf
import constants as c
from constants import Baselines, Environments, Strategies, PARAMETRS, StateSpaceSizeEstimations, MAX_NR_ACTIONS

class StateSpaceBuilder():
    
    def __init__(self, env_name: str, strategy: Strategies = Strategies.ESTIMATE_STATES, env = None, action_space = None):
        self.strategy = strategy
        self.env_name: str = env_name
        self.states_dict: Dict[str, int] = dict()

        # For the estimation strategy: 
        # Set up a counter that will be used to assign state-ids to unknown states.
        # This allows to identify matrix rows to individual states. 
        # Since an estimate is used, the states will be revealed and assigned to an id during the learning itself.
        self._estimation_state_id_ctr: int = 0

        if self.strategy == Strategies.EXPLORE_STATES.value:
            self.states_dict, _, _ = self._explore_states_in_n_steps(env=env, action_space=action_space, env_name=env_name)

    def get_states_dict(self) -> Dict[str, int]:
        return self.states_dict
    
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
        # Using the exploration strategy, the dictionary has been initialized at class initialization (preprocessing).
        # In this case the q-table as the exact same size.
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
                
                if not (len(states_steps_dict.keys()) % 50): print(f"\rExploring state space ... ({len(states_steps_dict.keys())} so far)", end="")
                
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
        file_path: str = f"{file_dir}/{env_name}"
        if not os.path.exists(file_path): os.mkdir(file_path)
        
        # Dict[str, id] - Given an id, return the state.
        filenname_states_id: str    = f"{file_path}/{c.fn_states_npy}"
        # Dict[id, str] - Given a state, return its id.
        filenname_id_states: str    = f"{file_path}/{c.fn_id_states_npy}"
        # Dict[str, actions_list:np.array].    
        filenname_actions: str      = f"{file_path}/{c.fn_actions_npy}"
        filenname_runtime: str      = f"{file_path}/{c.fn_runtime_npy}"

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
def run_q_learning(settings: Dict[PARAMETRS, str], set_tde_freq: int = None, seed: int = 42, verbose: bool = False):

    # Extract the parameter settings.
    env_name        = settings.get(PARAMETRS.ENV_NAME)
    nr_episodes     = settings.get(PARAMETRS.NR_EPISODES)
    # Learning rate (alpha).
    learning_rate   = settings.get(PARAMETRS.LEARNING_RATE)
    strategy        = settings.get(PARAMETRS.STATE_SPACE_STRATEGY)
    # Time discount/ costs for each time step. Aka 'gamma'.
    q_discount : float = settings.get(PARAMETRS.Q_DISCOUNT)
        
    # Initialize the agent:
    method_name: str = "QLearning"
    # Compute a time tag to define the directory where results are stored.
    time_tag: str = datetime.now().strftime("%Y_%m_%d-%H_%M")
    
    print(f">>RUN: {method_name}: env-{env_name}, e{nr_episodes}, lr{learning_rate}, {strategy}, d{q_discount}")    
    
    start_time: float = time.time()    
    # Load the environment.    
    env, action_space = hf.load_sokocoin_env(env_name)
    states_set = StateSpaceBuilder(env_name=env_name, strategy=strategy, env=env, action_space=action_space)
    np.random.seed(seed)
    
    # Store the Q-table.
    q_table: np.array = np.zeros((states_set.get_nr_states(), len(action_space)))

    # Initialize datastructures for the evaluation.
    # Every 'nr_episodes/loss_frequency' episode, save the last episodic returns and performance, and the accumulated loss.
    tde_frequency: int = nr_episodes
    if set_tde_freq is not None:
        tde_frequency: int = set_tde_freq
        
    eval_episode_ctr: int = 0
    # Save the actual names of the episodes, where a performance snapshot will be taken.
    evaluated_episodes: np.array        = np.zeros(tde_frequency)
    # Save the returns of the evaluated episodes.
    episodic_returns: np.array          = np.zeros(tde_frequency)
    # Save the performances of the evaluated episodes.
    episodic_performances: np.array     = np.zeros(tde_frequency)
    # Save the accumulated losses of the evaluated episodes.
    tdes: np.array                    = np.zeros(tde_frequency)
        
    # Initialize the exploration epsilon
    exploration_epsilons: np.array[float] = hf.get_annealed_epsilons(nr_episodes)
        
    state_id_old: int = None
    last_action: int   = None

    # Run training.
    # Record the performance of the agent (run until the time has run out) for 'number_episodes' many episodes.
    with alive_bar(nr_episodes) as bar:
        for episode in range(nr_episodes):        
            # Get the initial set of observations from the environment.
            timestep = env.reset()
            # Reset the variables for each episode.
            state_new: str              = ""
            state_old: str              = ""
            state_id_old: int           = None
            last_action: int            = None
            _actions_so_far: List[int]  = []
            _episode_tde: float         = 0.        
            exploration_epsilon: float  = exploration_epsilons[episode]

            while True:
                # Perform a step.
                state_new: str = str(timestep.observation['board'])
                # Check if the state is already known. Otherwise assign in a free 'state_id'.
                state_id_new = states_set.get_state_id(state_new)
                        
                # If this is NOT the initial state, update the q-values.
                # If this was the initial state, we do not have any reference q-values for states/actions before, and thus cannot update anything.
                if state_id_old is not None:
                    reward = timestep.reward
                    # Get the best action according to the q-table (trained so far).
                    max_action = hf.get_best_action(q_table, state_id_new, action_space)
                    # Calculate the q-value delta.
                    q_delta = (reward + q_discount * q_table[state_id_new, max_action] - q_table[state_id_old, last_action])
                    # Update the q-values.
                    q_table[state_id_old, last_action] += learning_rate * q_delta

                    # We define the loss as the 'squared temporal difference error'. In this case the delta^2.
                    # Accumulate the squared delta for 'loss_frequency' many uniformly-selected episodes.
                    if not(episode % (nr_episodes//tde_frequency)):
                        _step_loss = q_delta**2
                        _episode_tde += _step_loss
                
                # Break condition: If this was the last action, update the q-values for the terminal state one last time.            
                break_condition: bool = timestep.last() or len(_actions_so_far) >= MAX_NR_ACTIONS
                if break_condition:
                    # Before ending this episode, save the returns and performances.
                    if not(episode % (nr_episodes//tde_frequency)):
                        episodic_returns[eval_episode_ctr]      = env.episode_return
                        episodic_performances[eval_episode_ctr] = env.get_last_performance()
                        tdes[eval_episode_ctr]                  = _episode_tde
                        evaluated_episodes[eval_episode_ctr]    = episode
                        eval_episode_ctr += 1
                    break
                # Otherwise get the next action (according to the greedy exploration policy) and perform the step.
                else: 
                    action: int = hf.get_greedy_policy_action(exploration_epsilon, state_id_new, action_space, q_table)
                    timestep = env.step(action)
                    _actions_so_far.append(action)

                # Update the floop-variables.
                state_old: str    = state_new            
                state_id_old: int = state_id_new
                last_action: int   = action
            bar()

    # Measure the runtime.
    runtime = time.time() - start_time
    hf.save_results_to_file(settings, q_table, states_set.get_states_dict(), tdes, episodic_returns, episodic_performances, evaluated_episodes, seed, method_name=method_name, complete_runtime=runtime, dir_name_prefix=time_tag)

    return episodic_returns, episodic_performances

# RR-Learning implementation.
def run_rr_learning(settings: Dict[PARAMETRS, str], set_tde_freq: int = None, seed: int = 42, verbose: bool = False, save_coverage_table: bool = True):
    
    def get_the_baseline_state_id(_previous_baseline_id: int, _actions_so_far: List[int] = None) -> int:        
        # For the starting state baseline, nothing has to be computed. It is already set.
        if baseline == Baselines.STARTING_STATE_BASELINE.value:
            _baseline_state_id = _previous_baseline_id

        # For the inaction baseline, get the already simulated state after '_actions_so_far_ctr' many inactions.
        if baseline == Baselines.INACTION_BASELINE.value:
            _baseline_state_id = inaction_baseline_states[len(_actions_so_far) - 1]

        # For the stepwise inaction baseline, simulate doing the actions as before, but choosing the NOOP action for the last step.
        if baseline == Baselines.STEPWISE_INACTION_BASELINE.value:
            # Up to the last time step, simulate the environment as before.
            env_simulation, _ = hf.load_sokocoin_env(env_name)
            env_simulation.reset()
            for a in _actions_so_far[:-1]:                        
                env_simulation.step(a)
            # But for the last time step perform the NOOP action.
            timestep = env_simulation.step(action_space[4])
            _baseline_state = str(timestep.observation['board'])
            _baseline_state_id = states_set.get_state_id(_baseline_state)

        return _baseline_state_id
    
    # Initialize the agent:
    method_name: str = "RRLearning"
    # Compute a time tag to define the directory where results are stored.
    time_tag: str = datetime.now().strftime("%Y_%m_%d-%H_%M")

    # Extract the parameter settings.
    env_name: str           = settings.get(PARAMETRS.ENV_NAME)
    nr_episodes: int        = settings.get(PARAMETRS.NR_EPISODES)
    # Learning rate (alpha).
    learning_rate: float    = settings.get(PARAMETRS.LEARNING_RATE)
    strategy: str           = settings.get(PARAMETRS.STATE_SPACE_STRATEGY)
    # Time discount/ costs for each time step. Aka 'gamma'.    
    q_discount: float       = settings.get(PARAMETRS.Q_DISCOUNT)
    baseline: str           = settings.get(PARAMETRS.BASELINE)
    beta: float             = settings.get(PARAMETRS.BETA)    
        
    start_time: float = time.time()
    
    # Load the environment.
    env, action_space = hf.load_sokocoin_env(env_name)
    states_set = StateSpaceBuilder(env_name=env_name, strategy=strategy, env=env, action_space=action_space)
    np.random.seed(seed)
    
    print(f">>RUN: {method_name}: env-{env_name}, e{nr_episodes}, lr{learning_rate}, {strategy}, d{q_discount}, bl-{baseline}, b{beta}")
    
    # Store the Q-table.
    n: int = states_set.get_nr_states()
    q_table: np.array = np.zeros((n, len(action_space)), dtype=float)
    # Store the coverage values (reachability). 
    # Entrz 'ij' gives the coverage of state 'j' when starting from state 'i'. ({From_states}x{To_states})    
    c_table: np.array = np.zeros((n, n), dtype=np.float32)

    # Initialize datastructures for the evaluation.
    # Every 'nr_episodes/tde_frequency' episode, save the last episodic returns and performance, and the accumulated tde (temporal difference error).
    tde_frequency: int = nr_episodes
    if set_tde_freq is not None:
        tde_frequency: int = set_tde_freq
        
    eval_episode_ctr: int = 0
    # Save the actual names of the episodes, where a performance snapshot will be taken.
    evaluated_episodes: np.array        = np.zeros(tde_frequency)
    # Save the returns of the evaluated episodes.
    episodic_returns: np.array          = np.zeros(tde_frequency)
    # Save the performances of the evaluated episodes.
    episodic_performances: np.array     = np.zeros(tde_frequency)
    # Save the accumulated temporal difference errors of the evaluated episodes.
    tdes: np.array                      = np.zeros(tde_frequency)
    
    # Initialize the exploration epsilon
    exploration_epsilons: np.array[float] = hf.get_annealed_epsilons(nr_episodes)
      
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
    
    state_old_id: int = None
    _last_action: int   = None
    _use_encountered_states_for_avg_only: bool = True # TODO: Feature or not?
    
    # Run training.
    # Record the performance of the agent (run until the time has run out) for 'number_episodes' many episodes.
    with alive_bar(nr_episodes) as bar:
        for episode in range(nr_episodes):            
            # Get the initial set of observations from the environment.
            timestep = env.reset()
            # Reset the variables for each episode.
            state_new: str             = ""
            state_old: str             = ""
            state_new_id: int           = None
            state_old_id: int           = None
            _last_action: int           = None
            _actions_so_far: List[int]  = []
            _episode_tde: float        = 0.
            exploration_epsilon: float  = exploration_epsilons[episode]

            _state_baseline_id: int     = None

            _states_set: Set[str] = set()
            _states_id_set: Set[str] = set()
            
            while True:
                # Perform a step.
                state_new: str = str(timestep.observation['board'])
                _states_set.add(state_new)
                state_new_id = states_set.get_state_id(state_new)
                _states_id_set.add(state_new_id)

                if verbose:
                    print(f"\n\nFrom OLD state {state_old_id}\n{state_old}")
                    print(f"To NEW state {state_new_id}\n{state_new}")

                # If this is the initial state, save its id as baseline.
                # Thus by default the starting state basline is set. If another baseline-setting is used, it will be overwritten anyways.
                # Note that in this case, we do not have any reference q-values for states/actions before, and thus cannot update anything.
                if state_old_id is None:
                    _state_baseline_id = state_new_id
                # If this is NOT the initial state, update the q-values.
                else:
                    _state_baseline_id = get_the_baseline_state_id(_state_baseline_id, _actions_so_far)
                    
                    # UPDATE REACHABILITIES. Reachability is a measure of states that can be reached(?).
                    
                    # Calculate the coverage delta: '[c_reward + V(s_new) - V(s_old)]'.
                    # We update the coverage functions for all states 's'. Since the coverage value of a state 'j', when starting at state 'i' is denoted
                    # as matrix entry 'C_{i,j}', we compare the coverage values for all states when starting at the old vs. the new state. 
                    # Thus these are the rows of the coverage matrix.
                    # TODO: WAS DA LOOOOS c_delta = q_discount * c_table[state_new_id, :] - c_table[state_old_id, :]
                    c_delta = q_discount * c_table[:, state_new_id] - c_table[state_old_id, :]
                    # Add the coverage reward. It is simly 1.0, if the new state equals the last. And zero otherwise. Thus we add it to the right index.
                    c_delta[state_new_id] += 1
                    # Update the c-values. 'V(s_old) = V(s_old) + alpha * [c_discount + V(s_new) - V(s_old)]'.
                    c_table[state_old_id, :] += learning_rate * c_delta

                    # CALCULATE RELATIVE REACHABILITY. - first formula page 7 in paper "Peanalizing.."
                    # Compute the absolute reachability of all other states from the current state, compared to from the baseline state.
                    diff: np.array = c_table[_state_baseline_id, :] - c_table[state_new_id, :]
                    # Apply the maximum function. This ensures, that we do not punish reachabilities higher than the one of the baseline.
                    diff[diff<0] = 0.0
                    # if any(diff):
                    #     print("Non-zero difference")
                    # Average this reachability (clipped to non-negative values) to get the relative reachability.
                    # Notice that if we estimate the size of the state space, and are wrong by 'E', that is the estimated size is '|S| + E',
                    # then we can use a beta of 'beta * (1+E/|S|)' to expect the same restults as when using 'beta' and the correct size '|S|'.
                    # [Since: 1 / (|S| + E) ==> wrong by factor 1 / (1+E/|S|) ].
                    factor: float = 1.0
                    if _use_encountered_states_for_avg_only:
                        factor = len(diff) / len(_states_set)
                    d_rr: float = np.mean(diff) * factor
                    # if d_rr > 0:
                    #     print("Non-zero rr-penalty")
                    
                    # UPDATE Q-VALUES.
                    rr_reward = timestep.reward - beta * d_rr
                    # Get the best action according to the q-table (trained so far).
                    max_action = hf.get_best_action(q_table, state_new_id, action_space)
                    # Calculate the q-value delta.
                    q_delta = (rr_reward + q_discount * q_table[state_new_id, max_action] - q_table[state_old_id, _last_action])
                    # Update the q-values.
                    q_table[state_old_id, _last_action] += learning_rate * q_delta

                    if verbose:
                        print("Value functions")
                        print(f"s_OLD -> S:\t{np.around(c_table[state_old_id, :], 2)}")
                        print(f"s_NEW -> S:\t{np.around(c_table[state_new_id, :], 2)}")
                        print(f"RR-reward punishment:\t{d_rr}")
                        print(f"New env-reward:\t{rr_reward}")
                        print("Q-Table values:")
                        print(f"Q(s_OLD({state_old_id}), {_last_action}={c.ACTIONS.get(_last_action)}:\t{np.around(q_table[state_old_id, _last_action], 2)}")
                        print(f"Q(s_NEW({state_new_id}), {max_action}={c.ACTIONS.get(max_action)} (best):\t{np.around(q_table[state_new_id, max_action], 2)}")
                        print(f"C-Table for SxS with S={_states_id_set}:\n{c_table[list(_states_id_set), list(_states_id_set)]}")
                        print(f"Q-Table for states S={_states_id_set}:\n{q_table[list(_states_id_set), :]}")
                    
                    # We define the temporal difference error actually as the 'squared temporal difference error'. In this case the delta^2.
                    # Accumulate the squared delta for 'loss_frequency' many uniformly-selected episodes.
                    if not(episode % (nr_episodes//tde_frequency)):
                        _step_tde = q_delta**2
                        _episode_tde += _step_tde
                
                # Break condition: If this was the last action, update the q-values for the terminal state one last time.            
                break_condition: bool = timestep.last() or len(_actions_so_far) >= MAX_NR_ACTIONS
                if break_condition:
                    # Before ending this episode, save the returns and performances.
                    if not(episode % (nr_episodes//tde_frequency)):
                        episodic_returns[eval_episode_ctr]       = env.episode_return
                        episodic_performances[eval_episode_ctr]  = env.get_last_performance()
                        tdes[eval_episode_ctr]                 = _episode_tde
                        evaluated_episodes[eval_episode_ctr]     = episode
                        eval_episode_ctr += 1
                    break
                # Otherwise get the next action (according to the greedy exploration policy) and perform the step.
                else: 
                    action: int = hf.get_greedy_policy_action(exploration_epsilon, state_new_id, action_space, q_table)
                    timestep = env.step(action)
                    _actions_so_far.append(action)

                # Update the floop-variables.
                state_old: str    = state_new
                state_old_id: int = state_new_id
                # Save the last action, which brought the agent from the previous state to the now called old state.
                _last_action: int = action

            # Save the intermediate q-tables for further research.
            if episode == (nr_episodes // 3) or episode == 2 * (nr_episodes // 3):
                hf.save_intermediate_qtables_to_file(settings, q_table, episode, method_name, dir_name_prefix=time_tag)
            bar()

    runtime = time.time() - start_time
    if not save_coverage_table:
        c_table = None
    hf.save_results_to_file(settings, q_table, states_set.get_states_dict(), tdes, episodic_returns, episodic_performances, evaluated_episodes, seed, method_name=method_name, complete_runtime=runtime, coverage_table=c_table, dir_name_prefix=time_tag)

    return episodic_returns, episodic_performances

def create_settings_dict(env_name: str, nr_episodes: int, learning_rate: float = None, statespace_strategy: str = None, q_discount: float = None, baseline: str = None, beta: float = None) -> Dict[PARAMETRS, str]:
    """Store the parameter settings in a dictionary for simplified forwarding of the parameters.

    Args:
        env_name (str): Name of the gridworlds environment.
        nr_episodes (int): Number of learning episodes.
        learning_rate (float, optional): Factor to weight the computed delta for coverage and q-values. Assume values in (0,1).
        statespace_strategy (str, optional): Strategy to explore or estimate the number of possible states. Defaults to None.
        q_discount (float, optional): Factor to weight of the coverage and q-value of furture states. Defaults to None.
        baseline (str, optional): Baseline definition in the RR-Algorithm. Defaults to None.
        beta (float, optional): Factor to weight the relative reachability penality on the given reward. Beta = 0.0 implies standard Q-Learning. Defaults to None.

    Returns:
        Dict[PARAMETRS, str]: Dictionary of all possibly set parameters and their values.
    """
    settings: Dict[PARAMETRS, Any] = dict()

    settings[PARAMETRS.ENV_NAME]                = env_name
    settings[PARAMETRS.NR_EPISODES]             = nr_episodes
    settings[PARAMETRS.LEARNING_RATE]           = learning_rate
    settings[PARAMETRS.STATE_SPACE_STRATEGY]    = statespace_strategy
    settings[PARAMETRS.BASELINE]                = baseline
    settings[PARAMETRS.Q_DISCOUNT]              = q_discount
    settings[PARAMETRS.BETA]                    = beta
    
    return settings

# EXPERIMENTS
def run_experiments_q_vs_rr(env_names: List[Baselines], nr_episodes: int, learning_rates: List[float], discount_factors: List[float], betas: List[float], baselines: List[Baselines], strategy: Strategies = Strategies.ESTIMATE_STATES, save_coverage_table: bool = True):
    print()
    nr_experiments: int = len(env_names) * len(learning_rates) * len(discount_factors) * (len(betas) * len(baselines) + 1)
    ctr: int = 1
    
    for n in env_names:
        for lr in learning_rates:
            for d in discount_factors:
                print(f"Experiment {ctr}/{nr_experiments} - QLearning: {n.value}, e{nr_episodes}, lr{lr}, discount{d}, StateSetSizeStrategy:{strategy.value}")
                settings = create_settings_dict(env_name=n.value, nr_episodes=nr_episodes, learning_rate=lr, statespace_strategy=strategy.value, q_discount=d)
                run_q_learning(settings)
                ctr += 1
                
                for beta in betas:
                    for bl in baselines:
                        print(f"Experiment {ctr}/{nr_experiments} - RRLearning: {n.value}, e{nr_episodes}, lr{lr}, discount{d}, StateSetSizeStrategy:{strategy.value}")
                        settings = create_settings_dict(env_name=n.value, nr_episodes=nr_episodes, learning_rate=lr, statespace_strategy=strategy.value, q_discount=d, baseline=bl.value, beta=beta)
                        run_rr_learning(settings, save_coverage_table=save_coverage_table)
                        ctr += 1

def demo():
    env_names: Environments         = [Environments.SOKOCOIN0, Environments.SOKOCOIN2]
    nr_episodes: int                = 100
    learning_rates: List[float]     = [.1]
    discount_factors: List[float]   = [0.99]
    betas: List[float]              = [0.1]
    baselines: np.array[Baselines]  = np.array([Baselines.STARTING_STATE_BASELINE, Baselines.STEPWISE_INACTION_BASELINE])
    
    # Perform the learning using an estimation of the state space size.
    run_experiments_q_vs_rr(env_names, nr_episodes, learning_rates, discount_factors, betas, baselines, strategy=Strategies.ESTIMATE_STATES, save_coverage_table=True)
    # Perform the learning using an preprocessed exploration of the state space size.
    run_experiments_q_vs_rr(env_names, nr_episodes, learning_rates, discount_factors, betas, baselines, strategy=Strategies.EXPLORE_STATES, save_coverage_table=True)

def experiment_right_box_movement_girdsearch():    
    for lr in [0.5]:      # 0.1, 0.5, 1.0
        for discount in [0.99, 1.0]:
            for beta in [0.05, 20]:
                settings = create_settings_dict(
                    env_name=Environments.SOKOCOIN0.value, 
                    nr_episodes=1000, 
                    learning_rate=lr,
                    statespace_strategy=Strategies.ESTIMATE_STATES.value, 
                    q_discount=discount, 
                    baseline=Baselines.STARTING_STATE_BASELINE.value, 
                    beta=beta
                )
                run_rr_learning(settings, verbose=False, save_coverage_table=True)

def tiny_runtest_rr():    
    for bl in [Baselines.STARTING_STATE_BASELINE.value]:#, Baselines.STEPWISE_INACTION_BASELINE.value]:
        settings = create_settings_dict(
            env_name=Environments.SOKOCOIN0.value, 
            nr_episodes=3000, 
            learning_rate=.1,
            statespace_strategy=Strategies.ESTIMATE_STATES.value,
            q_discount=1.0,
            baseline=bl, 
            beta=.05
        )
        run_rr_learning(settings, save_coverage_table=True)

def experiment_complete_estimate():
    env_names: Environments         = [Environments.SOKOCOIN0, Environments.SOKOCOIN2]
    nr_episodes: int                = 10000
    learning_rates: List[float]     = [.1, .5]
    discount_factors: List[float]   = [0.99, 1.]
    betas: List[float]              = [0.1, 3, 100]
    baselines: np.array[Baselines]  = np.array([Baselines.STARTING_STATE_BASELINE, Baselines.STEPWISE_INACTION_BASELINE])
        
    run_experiments_q_vs_rr(env_names, nr_episodes, learning_rates, discount_factors, betas, baselines, strategy=Strategies.EXPLORE_STATES)

if __name__ == "__main__":
    tiny_runtest_rr()
    # experiment_right_box_movement_girdsearch()
    # experiment_right_box_movement_girdsearch()
        
    # TODO extras: Render the environements. Maybe even to Gif. > Run agent. TODO: Unused code in 'not_used.py'. Eliminate inaction baseline from experiments.
    # TODO extras: Implement usage of a sparse matrix.