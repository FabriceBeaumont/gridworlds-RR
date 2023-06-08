from ai_safety_gridworlds.environments.shared import safety_game
from collections import defaultdict
import experiments.environment_helper as environment_helper
import numpy as np

from copy import copy
from datetime import datetime
import os

import logging

# Used papers:
# 2019_Krakovna - "Penalizing side effects using stepwise relative reachability" - https://arxiv.org/abs/1806.01186


# Significant changes compared to the original 'model_free_aup.py'
# Name Map: Old -> New
# lembd         -> beta
# AUP_Q         -> q_table
# attainable_Q  -> coverage

# Other changes:
# Removed 'state_attainable'. Use the true setting always.
# Removed 'num_rewards' & 'rpenalties' with it as well.


def get_logger(name, filename):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s: %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=filename,
                        filemode='w',
                        encoding='utf-8')
    # Create console handler and set level to debug.
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Create formatter for the logger.
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s', datefmt='%m-%d %H:%M')
    # Add the formatter to console handler.
    ch.setFormatter(formatter)

    # Add the console handler to the logger.
    logger = logging.getLogger(name)
    logger.addHandler(ch)

    return logger


logger_file_name = 'logs/' + datetime.now().strftime("%y-%m-%d_%H-%M-%S") + 'my.log'
logger = get_logger('mymodelfree_training', logger_file_name)


class ModelFreeAUPAgent:
    name = "My Model-free AUP"
    # Learning rate as scaling for the updated reward.
    learning_rate = 1
    # Chance of choosing greedy action in training.
    # Start with a lower chance (less greedy), later choose a higher change (more greedy).
    greedy_epsilon = [.2, .9]
    percentile_less_greedy_episodes = 2.0 / 3
    # TODO: Grid search for 'beta': scaling of the intrinsic pseudo-reward (Gird search= 0.1, 0.3, 1, 3, 10, 30, 100, 300)
    default = {'beta': 0.1, 'discount': .996, 'episodes': 200}  # 6000

    def __init__(self, env, beta=default['beta'], discount=default['discount'], episodes=default['episodes'], trials=50, use_scale=False):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy.

        :param env:         Simulator.
        :param beta:        Impact tuning parameter/ Scaling of the intrinsic pseudo-reward. (float)
        :param discount:    'gamma' in the papers. Inverted cost of time steps (context of coverage function). (float)
        :param episodes:    Number of episodes. (int)
        :param trials:      Number of trials. (int)
        :param use_scale:   True - scale the penalty. False - scale only to non-zero. (bool)
        """
        # List of ids for every action.
        self.actions = range(env.action_spec().maximum + 1)
        n = len(self.actions)
        probsnp = (1.0 / (n - 1)) * (np.ones((n, n)) - np.eye(n))
        self.probs = probsnp.tolist()
        self.discount = discount
        self.beta = beta
        self.episodes = episodes
        self.trials = trials
        self.use_scale = use_scale
        self.env_name = env.name

        # Get a list of indicator functions which return the 'env.GOAL_REWARD' for one specific state.
        # The list contains such functions for all attainable states.
        # This list can be considered as list of all possible (attainable) states.
        logger.info("Calling 'Environment_helper.derive_possible_rewards'")
        self.attainable_set = environment_helper.derive_possible_rewards(env)
        # Store the coverage value (e.g. reachability) between all two states.
        # Since this matrix would be huge and is costly to compute (equivalent to the perfect solution), we calculate approximative values and update the entries during training. Since the actions and reached states depend on the time, which they were taken, this is not a transition matrix from one state to the next, but rather an evaluation of an action taken in the current state, with the goal to reach another state.
        # Type: Dict[np.matrix] - key:State_now, row:State_goal, col:Action to chose for this goal.
        self.coverage = None
        # Q-Table: As dictionary (key:State, value:List[Actions])
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

        # Start the training of the agent.
        logger.info('Start training')
        self.train(env)

    def train(self, env):
        """Perfom the training. This will iteratively update the 'coverage' and 'q_table', 
        which results in a strategy for the agent.
        """
        # Performance stats for each trial and each episode:
        self.performance = np.zeros((self.trials, self.episodes / 10))
        # 0: high-impact, incomplete; 1: high-impact, complete; 2: low-impact, incomplete; 3: low-impact, complete
        self.counts = np.zeros(4)

        for trial in range(self.trials):
            logger.info('Trial ' + str(trial) + '/' + str(self.trials) +
                        "(" + str(round(trial / self.trials * 100, 2)) + '%)')

            # Initialize the coverate and q-table for this trial.
            self.coverage = defaultdict(lambda: np.zeros(
                (len(self.attainable_set), len(self.actions))))
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

            # Train for all episodes.
            for episode in range(self.episodes):
                # Logging:
                if round(episode / self.episodes, 2) in [0.25, 0.5, 0.75]:
                    logger.info("Currently at episode " + str(episode+1) +
                                " (" + str(int(episode+1 / self.episodes)*100) + "%)")

                # Iterate through all time steps.
                time_step = env.reset()
                # Select the epsilon according to the e-greedy learning strategy.
                epsilon = self.select_epsilon(episode)
                while not time_step.last():
                    # Observe the board, get the current state.
                    last_state = str(time_step.observation['board'])
                    # Select an action (greedily or at random).
                    action = self.behavior_action(last_state, epsilon)
                    # Get the next time step. The agent will use the selected action in this time step.
                    time_step = env.step(action)
                    # Simulate the environment if the agent takes the action. Update the
                    # 'coverage' and 'q_table' accordingly.
                    self.update_greedy(last_state, action, time_step)

                # Save stats: Every tenth episode, use the trained agent to act upon the environment (not randomly).
                # Save its performance to the stats matrix.
                if episode % 10 == 0:
                    _, a, p, _ = environment_helper.run_episode(self, env)
                    actions, self.performance[trial][episode / 10] = a, p

            # Save the q-table:
            self.save_q_table(self.q_table, trial)

            # Save stats:
            # Count, how often a performance was reached. Therefore use performance values rounded as integer.
            # The performance value '-2' is the lowes expected value and stored in index '2'.
            # Increase the counter of the performance of the last episode in this trial (again shifted by '2').
            self.counts[int(self.performance[trial, -1]) + 2] += 1

        env.reset()

    def act(self, obs):
        """Returns the best action to perform at this time step, according to the learned q-table.

        Args:
            obs (_type_): Observation of the current board. (Aka the state.)

        Returns:
            action: The optimal action, according to the learned q-table.
        """
        return self.q_table[str(obs['board'])].argmax()

    def select_epsilon(self, episode):
        """Decides to use the greedy epsilon based on the number of episodes.

        Args:
            episode (int): Number of the current episode.

        Returns:
            float: The epsilon to use for the e-greedy search.
        """
        if episode <= self.percentile_less_greedy_episodes * self.episodes:
            # Less greedy epsilon.
            epsilon = self.greedy_epsilon[0]
        else:
            # More greedy epsilon.
            epsilon = self.greedy_epsilon[1]

        return epsilon

    def save_q_table(self, q_table, trial):
        """Save the q-table of the last trial to file.

        Args:
            q_table (Dict): Learned q-table.
            trial (int): Index of the current trial.
        """
        path = "q_tables/" + self.env_name
        file_name = str(datetime.now().strftime(
            "%y-%m-%d_%H-%M-%S")) + "_Qtable_t" + str(trial)

        if not os.path.exists(path):
            os.makedirs(path)

        np.save(path + "/" + file_name, np.array(dict(q_table)))

    def read_q_table(self, file_name):
        # TODO: LATER. Allow the agent to be initialized w/o training but read in q-table. And act accordingly.
        pass

    def behavior_action(self, state, epsilon):
        """Returns the e-greedy action for the state board string.

        Args:
            state (str): Observation of the current board. (Aka the state.)
            epsilon (float): Epsilon for the e-greedy search.

        Returns:
            Action: Action according to the e-greedy search (random action, or according to learned knowledge.)
        """
        greedy_action = self.q_table[state].argmax()
        if np.random.random() < epsilon or len(self.actions) == 1:
            return greedy_action
        else:
            # Choose any other action.
            return np.random.choice(self.actions, p=self.probs[greedy_action])

    def get_penalty(self, state, action):
        """Compute the penalty of the performed action in the current state.
           For the computation use the inaction baseline and scale the penalty.

           Equivalent to the formula for d_{RR} given in section 'Relative reachability' p.7
           in 2019_Krakovna, following the inaction baseline strategy.
           That is, to perform the no-operation action instead of the chosen one in the current time step.

        Args:
            state (str): Observation of the current board. (Aka the state.)
            action (Action): Action performed by the agent in the current time step.

        Returns:
            float: Penalty for performing this action in the current time step.
        """
        if len(self.attainable_set) == 0:
            return 0

        # Vector of coverage to all states - when taking action 'action' in state 'state'.
        # $R(s_t ; s)$
        coverage_of_action = self.coverage[state][:, action]

        # Vector of coverage to all states - when taking action 'noop-action' in state 'state'.
        # $R(s'_t ; s)$ - where $s'_t$ is the inaction baseline.
        coverage_of_noop = self.coverage[state][:,
                                                safety_game.Actions.NOTHING]
        diff = coverage_of_noop - coverage_of_action

        # Scaling number or vector (per-AU)
        if self.use_scale:
            scale = sum(abs(coverage_of_noop))
            if scale == 0:
                scale = 1
            penalty = sum(abs(diff) / scale)
        else:
            scale = np.copy(coverage_of_noop)
            # Avoid division by zero.
            scale[scale == 0] = 1
            penalty = np.average(np.divide(abs(diff), scale))

        # Scaled difference between taking action and doing nothing.
        return self.beta * penalty

    def update_greedy(self, last_state, action, time_step):
        """Perform temporal difference (TD) update on observed reward.

        Args:
            last_state (str): Observation of the current board. (Aka the state.)
            action (Action): Action performed by the agent in the current time step.
            time_step (int): Current time step, where the action is taken, from the last state.
        """
        def calculate_q_update():
            """Compute the update for the q-table."""
            # Diminish the reward of the environment with the penalty due to the taken action.
            reward = time_step.reward - self.get_penalty(last_state, action)

            new_q = self.q_table[new_state].max()
            old_q = self.q_table[last_state][action]

            return self.learning_rate * (reward + self.discount * new_q - old_q)

        def calculate_coverage_update(state_idx=None):
            """Calculate the update for the coverage function at the given index)."""
            reward = self.attainable_set[state_idx](new_state)

            if reward != 0.0:
                pass

            new_q = self.coverage[new_state][state_idx].max()
            old_q = self.coverage[last_state][state_idx, action]

            return self.learning_rate * (reward + self.discount * new_q - old_q)

        new_state = str(time_step.observation['board'])
        # Update the coverage values, based on the state reached by taking the selected action.
        # Therefore iterate through all states and compute the update.
        for state_idx in range(len(self.attainable_set)):
            update = calculate_coverage_update(state_idx)
            self.coverage[last_state][state_idx, action] += update

        # TODO: WHY? Needed?
        # Clip the coverage for all states in the column of the choosen action to {0,1}.
        self.coverage[last_state][:, action] = np.clip(
            self.coverage[last_state][:, action], 0, 1)

        # Update the q-table.
        self.q_table[last_state][action] += calculate_q_update()
