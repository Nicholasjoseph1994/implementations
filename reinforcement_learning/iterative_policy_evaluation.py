""""
Implementation of Iterative Policy Evaluation from Sutton and Barto (Page 61)
"""

import itertools
import sys
import numpy as np

def iterative_policy_evaluation(policy,
                                states,
                                actions,
                                transition,
                                rewards,
                                discount=0.99,
                                threshold=0.01):
    """
    This performs iterative policy evaluation to calculate the value function for a policy.

    Parameters
    ----------
    policy: (State, Action) --> float
        The probability assigned to taking the action in the given state by the policy
    states: [State]
        The state space of the MDP
    actions: [Action]
        The action space of the MDP
    transition: (state, action, next_state, reward) --> float
        The transition function of the MDP
    rewards: [float]
        All possible rewards
    discount: float
        Exponential discount rate applied to future rewards
    threshold: float
        Threshold for changes in the value function used to determine when convergence has been reached.

    Returns
    -------
    dict: States are keys and the corresponding value function results are the values

    """
    update = sys.maxint

    # Initialize values to zero
    values = {state: 0 for state in states}

    # Iterate until convergence
    while update > threshold:
        update = 0
        for state in states:
            initial_value = values[state]

            # Use Bellman equation update
            values[state] = sum([
                 policy(state, action) * \
                 sum([transition(s, a, n_s, r) * (r + discount * values[n_s])
                     for (s, a, n_s, r) in itertools.product(states, actions, states, rewards)
                     if s == state and a == action])
                 for action in actions])
            update = max(update, abs( initial_value - values[state]))
    return values

def test_iterative_policy_evaluation():
    states = range(15)
    actions = np.array(['up', 'down', 'right', 'left'])
    def transition(state, action, next_state, reward):
        # Remain in terminal state with no reward
        if state == 0:
            return float(next_state == 0 and reward == 0)
        if action == 'up':
            target_state = state - 4 if state > 3 else state
        if action == 'down':
            target_state = state + 4 if state < 12 else state
        if action == 'right':
            target_state = state + 1 if state % 4 != 3 else state
        if action == 'left':
            target_state = state - 1 if state % 4 != 0 else state
        if target_state == 15:
            target_state = 0

        return float(target_state == next_state and reward == -1)
    rewards = [-1]
    policy = lambda state, action: 1. / len(actions)


    value_function = np.array(iterative_policy_evaluation(policy, states, actions, transition, rewards, discount=1, threshold=0.0001).values())
    true_value_function = np.array([0., -14., -20., -22., -14., -18., -20., -20., -20., -20., -18., -14., -22., -20., -14.])
    assert np.all(np.isclose(value_function, true_value_function, atol=0.001))

test_iterative_policy_evaluation()