from ai_safety_gridworlds.environments.shared import safety_game
import numpy as np


class Baselines(Enum):
    # Starting state baseline. Use the initial state of the board as baseline.
    STARTING = "start"
    # Inaction baseline. Use the state as baseline, which would be obtained, by taking the inaction baseline in all time steps up to the current one.
    INACTION = "inaction"
    # Stepwise baseline. Use the state as baseline, which would be obtained, by using the noop action instead of the last chosen action.
    STEPWISE = "stepwise"
    # Stepwise rollout baseline. Use the state as baseline, which would be obtained, by using the noop action instead of the last chosen action. Look all (k) steps ahead and observe the environment when using the noop action from now on (inaction rollout).
    STEPWISE_ROLLOUT = "stepwise_rollout"


class Deviations(Enum):
    # Absolute deviation means that the penality update is computed based on any change of the coverage value.
    # Positive or negative change is punished. 
    # Thus use the summary function 'f(x) = |x|' [implemented as 'abs(x)'].
    ABSOLUTE = "absolute"
    # Decrease in deviation means, that coverage improvements (decrease) are not punished.
    # Thus use the summary function 'f(x) = max(0, x)' [implemented as 'diff[diff > 0]'].
    DECREASE = "decrease"


class AUPAgent():
    """
    Attainable utility-preserving agent.
    """
    name = 'AUP'

    def __init__(self, coverage, beta=1/1.501, discount=.996, baseline=Baselines.STEPWISE, deviation=Deviations.DECREASE, use_scale=False):
        """
        :param coverage:    Coverage functions for all states. Dict[np.matrix] - key:State_now, row:State_goal, col:Action to chose for this goal.
        :param beta:        Scale harshness of penalty. Impact tuning parameter/ Scaling of the intrinsic pseudo-reward. (float)
        :param discount:
        :param baseline:    That with respect to which we calculate impact.
        :param deviation:   How to penalize shifts in attainable utility.
        """
        self.coverage = coverage
        self.beta = beta
        self.discount = discount
        self.baseline = baseline
        self.baseline_state_coverage = 0.0
        self.deviation = deviation
        self.use_scale = use_scale
        self.cached_actions = dict()
        self.init_agent_name()

    def init_agent_name(self):
        # TODO: Check if self.name is needed elswhere. Otherwise use 'AUP_stepwise', 'AUP_startingState', ...
        if baseline != Baselines.STEPWISE:
            self.name = baseline.capitalize()
            if baseline == Baselines.STARTING:
                self.name = 'Starting state'
        if deviation != Deviations.ABSOLUTE:
            self.name = deviation.capitalize()

        if baseline == Baselines.INACTION and deviation == Deviations.DECREASE:
            self.name = 'Relative reachability'

    def init_baseline_state(self, env, steps_left):
        """Initialize the 'baseline_state_coverage' based on the selected 'baseline' definition.
        Note that this can only be done for the starting state and inaction baseline, since all other baselines are defined dynamically based on the last taken action and the time step.
        """
        if self.baseline == Baselines.STARTING:
            # For the starting state baseline take the state of the board right now (recall that so far no actions have been made / no time has passed.)
            bl_state = str(env.last_observations['board'])
            bl_state_coverage = self.coverage[bl_state].max(axis=1)
        elif self.baseline == Baselines.INACTION:
            # For the inaction baseline, simulate the environment for the necessary time steps while
            # taking the noop action each time.
            self.restart(env, [safety_game.Actions.NOTHING] * steps_left)
            bl_state = str(env.last_observations['board'])
            bl_state_coverage = self.coverage[].max(axis=1)
            env.reset()

        self.baseline_state = bl_state
        self.baseline_state_coverage = bl_state_coverage

    def compute_baseline_coverage(self, env, so_far, steps_left)
        """Return the covage of the basline state. That is how easily all other states can be reached from the baseline state.
        '[R(s_0; s)]_{s \in S}' in the papers

        Returns:
            List[float]: _description_
        """
        if self.baseline == Baselines.STARTING or self.baseline == Baselines.INACTION:
            # In these cases the coverage of the baseline has already been computed in advance.
            return self.baseline_state_coverage

        if self.baseline == Baselines.STEPWISE:
            # Perform the actions so far, the noop action instead of the suggested one, and noop action for all further time steps.
            inaction_plan = so_far + [safety_game.Actions.NOTHING] * steps_left

            # Simulate the environment, when taking the inaction plan.
            self.restart(env, inaction_plan)
            baseline_state = str(env._last_observations['board'])

            # Get the coverage of all states, when taking only the noop-action.
            self.baseline_state_coverage = self.coverage[baseline_state][:, safety_game.Actions.NOTHING]

        if self.baseline == Baselines.STEPWISE_ROLLOUT:
            # TODO: Add code for the stepwise rollout baseline here.
            # TODO: Recursive formula from section 'Modifications required with the stepwise inaction baseline' / page 8
            pass

        return self.baseline_state_coverage

    def get_actions(self, env, steps_left, so_far=[]):
        """Figure out the n-step optimal plan, returning it and its return.

        :param env: Simulator.
        :param steps_left: How many steps to plan over. (int)
        :param so_far: Actions taken up until now. (List[Actions])
        """
        if steps_left == 0:
            return [], 0

        # If no action has been taken so far, compute the baseline state.
        # Note that depending on the baseline definition, the baseline state will be overwritten later on.
        if len(so_far) == 0:
            self.init_baseline_state(env, steps_left)

        # Hash the current state and the steps left. This hash will be used to store the best action for this state and time.
        current_hash = (str(env.last_observations['board']), steps_left)
        # If the hash is known, a best action for this state and time is already avaliable and will be returned.
        if current_hash not in self.cached_actions:
            # Save a list of optimal actions and their respective rewards.
            best_actions = []
            best_return_value = float('-inf')
            # Simulate the next time step for each available action independently.
            for a in range(env.action_spec().maximum + 1):
                # Compute the reward for taking action 'a'.
                reward, terminated = self.penalized_reward(
                    env, a, steps_left, so_far)
                if not terminated:
                    # Continue simulating, for one less time step and when taking action 'a' now.
                    actions, return_value = self.get_actions(
                        env, steps_left - 1, so_far + [a])
                else:
                    actions = []
                    return_value = 0

                return_value *= self.discount
                # If the gained reward is better than the last known, update the action as better action.
                if reward + return_value > best_return_value:
                    best_actions, best_return_value = [
                        a] + actions, r + return_value
                self.restart(env, so_far)

            # After simulating all possible actions, the best ones are stored and returned.
            self.cached_actions[current_hash] = best_actions, best_return_value
        return self.cached_actions[current_hash]

    @staticmethod
    def restart(env, actions):
        """Reset the environment and return the result of executing the action sequence."""
        time_step = env.reset()
        for action in actions:
            if time_step.last():
                break
            time_step = env.step(action)

    def penalized_reward(self, env, action, steps_left, so_far=[]):
        """The penalized reward for taking the given action in the current state. Steps the environment forward.

        :param env:         Simulator.
        :param action:      The action in question. (Action)
        :param steps_left:  How many steps are left in the plan. (int)
        :param so_far:      Actions taken up until now. (List[Action])
        :returns penalized_reward:
        :returns is_last:   Whether the episode is terminated. (bool)
        """
        # Perform the action.
        time_step = env.step(action)
        # Assume no reward.
        reward = 0
        if time_step.reward:
            # Update the reward, if the environment returns is (e.a. winning the game).
            reward = time_step.reward
        scaled_penalty = 0

        if self.coverage:            
            # Perform the actions so far, the chosen actions, and noop action for all further time steps.
            action_plan = so_far + [action] + [safety_game.Actions.NOTHING] * (steps_left - 1)

            # Simulate the environment, when taking the action plan.
            self.restart(env, action_plan)
            state = str(env._last_observations['board'])
            # Save the coverage of the current state (when taking the action).
            # That is how easily all other states are reachable from the current state.

            # action_coverage \in S
            # [action_coverage]_i = how easily is state i reachable from the current state. 
            # (Measured by taking the action corresponding to the column of maximum value.)
            # Thus return the maximum value for each row.
            # '[R(s_t; s)]_{s \in S}' in the papers
            action_coverage = self.coverage[state].max(axis=1)

            # '[R(s_0; s)]_{s \in S}' in the papers
            baseline_coverage = self.compute_baseline_coverage(env, so_far, steps_left)

            diff = baseline_coverage - action_coverage
                        
            # Do not penalize increases, thus use 'max(diff, 0)'.
            diff[diff > 0] = 0
            # Define the penalty as the average reduction in reachability of all states from the current state, 
            # compared to the baseline state.
            penalty = np.average(diff)    
                    
            # Scale the penalty using the impact tuning parameter 'beta'.
            scaled_penalty = self.beta * penalty
            self.restart(env, so_far + [action])
            
        return reward - scaled_penalty, time_step.last()
