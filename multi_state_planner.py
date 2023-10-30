import numpy as np
#import pulp
import math
import argparse
from utils.generate_multistate_mdp_utils import generate_pomdp, generate_states
from planner import valueEvaluation, Q_pi, brute_force_search

if __name__ == "__main__":
    actions = ["R", "B", "G"]
    sensingActions = [action + "S" for action in actions]
    numHeadStates = 2
    windowLength = 1
    states = generate_states(numHeadStates, actions, windowLength)

    T = {
        "R": np.array([[0.8, 0.2], [0.2, 0.8]]),
        "B": np.array([[0.8, 0.2], [0.2, 0.8]]),
        "G": np.array([[0.8, 0.2], [0.2, 0.8]])
    }

    C = {
        "R": np.array([0, 1]),
        "B": np.array([1, 0]),
        "G": np.array([0, 0])
    }

    for action in actions:
        print(T[action], C[action])
    mdp = generate_pomdp(1, T, C, 0.9, actions, numHeadStates, windowLength)

    policy, value_function  = brute_force_search(states, actions+sensingActions, mdp, 0.5, windowLength)
    # print(policy)

    opt_policy, opt_val = brute_force_search(states, actions+sensingActions, mdp, 0.5, windowLength)
    policy = {i: tuple([opt_policy[tuple([i])]]) for i in range(numHeadStates)}

    for i in range(numHeadStates):
        while(policy[i][-1][-1] != 'S'):
            policy[i] = policy[i] + tuple([opt_policy[tuple([i])+policy[i]]])

    for i in range(numHeadStates):
        print(f"state {i}, policy: {policy[i]}, value: {opt_val[tuple([i])]}")