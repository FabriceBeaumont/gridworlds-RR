from ai_safety_gridworlds.environments.shared import safety_game
import numpy as np


class AUPAgent():
    """
    Attainable utility-preserving agent.
    """
    name = 'AUP'

    def __init__(self, attainable_Q, lambd=1/1.501, discount=.996, baseline='stepwise', deviation='absolute',
                 use_scale=False):
        """
        :param attainable_Q: Q functions for the attainable set.
        :param lambd: Scale harshness of penalty.
        :param discount:
        :param baseline: That with respect to which we calculate impact.
        :param deviation: How to penalize shifts in attainable utility.
        """
        self.attainable_Q = attainable_Q
        self.lambd = lambd
        self.discount = discount
        self.baseline = baseline
        self.deviation = deviation
        self.use_scale = use_scale

        if baseline != 'stepwise':
            self.name = baseline.capitalize()
            if baseline == 'start':
                self.name = 'Starting state'
        if deviation != 'absolute':
            self.name = deviation.capitalize()

        if baseline == 'inaction' and deviation == 'decrease':
            self.name = 'Relative reachability'

        self.cached_actions = dict()

    def get_actions(self, env, steps_left, so_far=[]):
        """Figure out the n-step optimal plan, returning it and its return.

        :param env: Simulator.
        :param steps_left: How many steps to plan over.
        :param so_far: Actions taken up until now.
        """
        if steps_left == 0: return [], 0
        if len(so_far) == 0:
            if self.baseline == 'start':
                self.null = self.attainable_Q[str(env.last_observations['board'])].max(axis=1)
            elif self.baseline == 'inaction':
                self.restart(env, [safety_game.Actions.NOTHING] * steps_left)
                self.null = self.attainable_Q[str(env.last_observations['board'])].max(axis=1)
                env.reset()
        current_hash = (str(env.last_observations['board']), steps_left)
        if current_hash not in self.cached_actions:
            best_actions, best_ret = [], float('-inf')
            for a in range(env.action_spec().maximum + 1): # for each available action
                r, done = self.penalized_reward(env, a, steps_left, so_far)
                if not done:
                    actions, ret = self.get_actions(env, steps_left - 1, so_far + [a])
                else:
                    actions, ret = [], 0
                ret *= self.discount
                if r + ret > best_ret:
                    best_actions, best_ret = [a] + actions, r + ret
                self.restart(env, so_far)

            self.cached_actions[current_hash] = best_actions, best_ret
        return self.cached_actions[current_hash]

    @staticmethod
    def restart(env, actions):
        """Reset the environment and return the result of executing the action sequence."""
        time_step = env.reset()
        for action in actions:
            if time_step.last(): break
            time_step = env.step(action)

    def penalized_reward(self, env, action, steps_left, so_far=[]):
        """The penalized reward for taking the given action in the current state. Steps the environment forward.

        :param env: Simulator.
        :param action: The action in question.
        :param steps_left: How many steps are left in the plan.
        :param so_far: Actions taken up until now.
        :returns penalized_reward:
        :returns is_last: Whether the episode is terminated.
        """
        time_step = env.step(action)
        reward, scaled_penalty = time_step.reward if time_step.reward else 0, 0
        if self.attainable_Q:
            action_plan, inaction_plan = so_far + [action] + [safety_game.Actions.NOTHING] * (steps_left - 1), \
                                         so_far + [safety_game.Actions.NOTHING] * steps_left

            self.restart(env, action_plan)
            action_attainable = self.attainable_Q[str(env._last_observations['board'])].max(axis=1)

            self.restart(env, inaction_plan)
            null_attainable = self.attainable_Q[str(env._last_observations['board'])][:, safety_game.Actions.NOTHING] \
                if self.baseline == 'stepwise' else self.null
            diff = action_attainable - null_attainable
            if self.deviation == 'decrease':
                diff[diff > 0] = 0  # don't penalize increases

            # Scaling number or vector (per-AU)
            if self.use_scale:
                scale = sum(abs(null_attainable))
                if scale == 0:
                    scale = 1
                penalty = sum(abs(diff) / scale)
            else:
                scale = np.copy(null_attainable)
                scale[scale == 0] = 1  # avoid division by zero
                penalty = np.average(np.divide(abs(diff), scale))

            scaled_penalty = self.lambd * penalty
            self.restart(env, so_far + [action])
        return reward - scaled_penalty, time_step.last()
