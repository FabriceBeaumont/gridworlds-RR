from ai_safety_gridworlds.environments.shared import safety_game
from collections import defaultdict
import experiments.environment_helper as environment_helper
import numpy as np

from copy import copy
from datetime import datetime
import os
from enum import Enum

import logging


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
logger = get_logger('srr_training', logger_file_name)


class Baselines(Enum):
    STEPWISE = "stepwise_baseline"
    STARTING = "starting_state_baseline"
    INACTION = "inaction_baseline"


class ModelFreeSRRAgent:
    name = "Stepwise RR"
    # Learning rate as scaling for the updated reward.
    learning_rate = 1
    # Chance of choosing greedy action in training.
    # Start with a lower chance (less greedy), later choose a higher change (more greedy).
    greedy_epsilon = [.2, .9]
    percentile_less_greedy_episodes = 2.0 / 3
    # TODO: Grid search for 'beta': scaling of the intrinsic pseudo-reward (Gird search= 0.1, 0.3, 1, 3, 10, 30, 100, 300)
    default = {'beta': 0.1, 'discount': .996, 'episodes': 200,
               'baseline': Baselines.INACTION}  # 6000

    def __init__(self, env, baseline=default['baseline'],  beta=default['beta'], discount=default['discount'], episodes=default['episodes'], trials=50, use_scale=False):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy.

        :param env: Simulator.
        :param beta: Impact tuning parameter/ Scaling of the intrinsic pseudo-reward
        :param discount: 'gamma' in the papers. (float)
        :param episodes: Number of episodes. (int)
        :param trials: Number of trials. (int)
        :param use_scale: True - scale the penalty. False - scale only to non-zero. (bool)
        """
        self.actions = range(env.action_spec().maximum + 1)
        n = len(self.actions)
        probsnp = (1.0 / (n - 1)) * (np.ones((n, n)) - np.eye(n))
        self.probs = probsnp.tolist()
        self.baseline = baseline  # type: <Baselines class>
        self.discount = discount  # type: float
        self.episodes = episodes
        self.trials = trials
        self.beta = beta
        # Get a list of indicator functions which return the 'env.GOAL_REWARD' for one specific state.
        # The list contains such functions for all attainable states.
        logger.info("Calling 'Environment_helper.derive_possible_rewards'")
        self.attainable_set = environment_helper.derive_possible_rewards(env)
        # Defining the coverage function relative reachabiliy R(x,y) between any two states x and y.
        self.relative_reachability = defaultdict(lambda: np.zeros((len(self.attainable_set), len(
            self.attainable_set))))  # Dict[[str, str], float] - [[state, state], reachability]
        # TODO: Implement Init / (saving and read-in?)

        self.use_scale = use_scale
        self.env_name = env.name

        # Start the training of the agent.
        logger.info('Start training')
        self.train(env)

    def train(self, env):
        # Performance stats for each trial and each episode:
        self.performance = np.zeros((self.trials, self.episodes / 10))
        # 0: high-impact, incomplete; 1: high-impact, complete; 2: low-impact, incomplete; 3: low-impact, complete
        self.counts = np.zeros(4)

        for trial in range(self.trials):
            logger.info('Trial ' + str(trial) + '/' + str(self.trials) +
                        "(" + str(round(trial / self.trials * 100, 2)) + '%)')
            # Q-Tables FOR EACH states.
            self.attainable_Q = defaultdict(lambda: np.zeros(
                (len(self.attainable_set), len(self.actions))))
            # Q-Table: As dictionary (key:State, value:List[Actions])
            self.Q_table = defaultdict(lambda: np.zeros(len(self.actions)))

            # Train for all episodes.
            for episode in range(self.episodes):
                # Logging:
                if round(episode / self.episodes, 2) in [0.25, 0.5, 0.75]:
                    logger.info("Currently at episode " + str(episode+1) +
                                " (" + str(int(episode+1 / self.episodes)*100) + "%)")
                # Iterate through all time steps.
                time_step = env.reset()
                epsilon = self.select_epsilon(episode)
                while not time_step.last():
                    # Observe the board.
                    last_state = str(time_step.observation['board'])
                    # Select an action (greedily or at random).
                    action = self.get_action(last_state, epsilon)
                    time_step = env.step(action)
                    self.update_greedy(last_state, action, time_step)

                # Save stats: Every tenth episode, use the trained agent to act upon the environment (not randomly).
                # Save its performance to the stats matrix.
                if episode % 10 == 0:
                    _, a, p, _ = environment_helper.run_episode(self, env)
                    actions, self.performance[trial][episode / 10] = a, p

            # Save the q-table:
            self.save_q_table(self.Q_table, trial)

            # Save stats:
            # Count, how often a performance was reached. Therefore use performance values rounded as integer.
            # The performance value '-2' is the lowes expected value and stored in index '2'.
            # Increase the counter of the performance of the last episode in this trial (again shifted by '2').
            self.counts[int(self.performance[trial, -1]) + 2] += 1

        env.reset()

    def act(self, obs):
        return self.Q_table[str(obs['board'])].argmax()

    def select_epsilon(self, episode):
        """ Decides to use the greedy epsilon based on the number of episodes."""
        if episode <= self.percentile_less_greedy_episodes * self.episodes:
            # Less greedy epsilon.
            epsilon = self.greedy_epsilon[0]
        else:
            # More greedy epsilon.
            epsilon = self.greedy_epsilon[1]

        return epsilon

    def save_q_table(self, q_table, trial):
        """Save the q-table of the last trial to file."""
        path = "q_tables/" + self.env_name
        file_name = str(datetime.now().strftime(
            "%y-%m-%d_%H-%M-%S")) + "_Qtable_t" + str(trial)

        if not os.path.exists(path):
            os.makedirs(path)

        np.save(path + "/" + file_name, np.array(dict(q_table)))

    def read_q_table(self, file_name):
        # TODO: LATER. Allow the agent to be initialized w/o training but read in q-table. And act accordingly.
        pass

    def get_action(self, board, epsilon):
        """ Returns the e-greedy action for the state board string."""
        greedy_action = self.Q_table[board].argmax()
        if np.random.random() < epsilon or len(self.actions) == 1:
            return greedy_action
        else:
            # Choose anything else.
            return np.random.choice(self.actions, p=self.probs[greedy_action])

    def get_baseline_state(self, last_state, time_step=0):
        """Select the baseline state according to the baseline policy.

        Args:
            last_state (str): The last state.
            time_step (int, optional): The time step. Only needed for the inaction baseline. 
            Defaults to 0.

        Returns:
            str: The baseline state.
        """
        # TODO: Implement
        if self.baseline == Baselines.INACTION:
            return self.inaction_baseline(t)
        elif if self.baseline == Baselines.STARTING:
            return self.inaction_baseline(0)
        elif if self.baseline == Baselines.STEPWISE:
            return last_state
        else:
            # TODO: catch error
            return None

    def get_rr_penalty(self, state, action, baseline_state):
        """ Compute the penalty of the performed action in the current state.

            Scaled relative reachability - that is the average reduction in reachability of all states from the current state compared to the baseline.
        """
        if len(self.attainable_set) == 0:
            return 0

        # Compute the sum over all max( r_bs[S] - r_s[S] , 0) as a vector difference.
        diff = np.array(self.relative_reachability[baseline_state]) - \
            np.array(self.relative_reachability[current_state])

        # Apply the max(*,0) function.
        diff[diff < 0] = 0

        # Scaling number or vector (per-AU)
        scale = sum(diff)
        if scale == 0:
            scale = 1
        penalty = diff / scale

        return self.beta * penalty

    def update_greedy(self, last_state, action, time_step):
        """Perform temporal difference (TD) update on observed reward."""

        def calculate_q_update():
            """Compute the update for the q-table."""

            # TODO: Change this update function?

            baseline_state = self.get_baseline_state(last_state, time_step)

            reward = time_step.reward - \
                self.get_rr_penalty(last_state, action, baseline_state)
            new_Q = self.Q_table[new_state].max()
            old_Q = self.Q_table[last_state][action]

            return self.learning_rate * (reward + self.discount * new_Q - old_Q)

        def calculate_attainable_update(attainable_idx=None):
            """Calculate the update for the attainable function at the given index)."""
            reward = self.attainable_set[attainable_idx](new_state)

            if reward != 0.0:
                pass

            new_Q = self.attainable_Q[new_state][attainable_idx].max()
            old_Q = self.attainable_Q[last_state][attainable_idx, action]

            return self.learning_rate * (reward + self.discount * new_Q - old_Q)

        new_state = str(time_step.observation['board'])
        # Learn the attainable reward functions.
        # TODO: Understand this / do we need to replace this?
        for attainable_idx in range(len(self.attainable_set)):
            update = calculate_attainable_update(attainable_idx)
            self.attainable_Q[last_state][attainable_idx, action] += update

        # TODO: WHY? Needed?
        # Clip the rewards for all states in the column of the choosen action to {0,1}.
        self.attainable_Q[last_state][:, action] = np.clip(
            self.attainable_Q[last_state][:, action], 0, 1)

        self.Q_table[last_state][action] += calculate_q_update()
