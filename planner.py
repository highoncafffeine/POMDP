import numpy as np
import pulp
import math
import argparse
from generate_mdp import generate_states


def valueEvaluation(policy, states, actions, transition, gamma, window_len):
    v0 = {state: 0 for state in states}
    v1 = {state: 0 for state in states}
    max_error = 1e-12
    count = 0

    a0 = 0
    while(1):
        a0+=1
        for state in states:
            v1[state] = 0
            for nextState in transition[state][policy[state]].keys():
                prob = transition[state][policy[state]][nextState]['prob']
                cost = transition[state][policy[state]][nextState]['cost']
                v1[state] += (prob*(cost + (gamma*v0[nextState])))
        if np.max(np.abs(np.subtract(v0,v1))) < max_error:
            break

        v0 = np.copy(v1)
        count += 1

    return v1

def Q_pi(V, s, a, transition, gamma, window_len):
    return sum([transition[s][a][s_]['prob']*(transition[s][a][s_]['cost']+gamma*V[s_]) for s_ in transition[s][a].keys()])

def brute_force_search(states, actions, transition, gamma, window_len):
    policy = {state: 'RS' for state in states}
    while(True):
        V = valueEvaluation(policy, states, actions, transition, gamma, window_len)
        improved_policy = policy.copy()
        improvable = False
        for s in states:
            for a in actions:
                if a == improved_policy[s]:
                    continue
                t_policy = policy.copy()
                if(len(s)<window_len+1 or (a[-1] == 'S')):
                    t_policy[s] = a
                if(V[s] - Q_pi(V, s, a) > 1e-7):
                    improved_policy[s] = a
                    improvable = True
                    break
        if not improvable:
            break
        else:
            policy = improved_policy.copy()
    value_function = valueEvaluation(policy, states, actions, transition, gamma, window_len)
    policy = policy
    return policy, val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp")
    parser.add_argument("--policy",default= "")
    parser.add_argument("--optimal", action=argparse.BooleanOptionalAction)
    parser.add_argument("--window_len", default = -1)
    args = parser.parse_args()

    file = args.mdp
    pol_file = args.policy
    optimal = args.optimal
    window_len  = int(args.window_len)

    with open(file) as f:
        content = f.readlines()
    file = [content[i][:-1].split() for i in range(len(content))]   


    mdp ={}
    mdp['numStates'] = int(file[0][1])
    mdp['numActions'] = int(file[1][1])
    actions = ['R', 'B', 'RS', 'BS']
    states0, states1 = generate_states(WINDOW_LEN = window_len)
    states = states0 + states1

    mdp['end'] = []
    for i in range(1,len(file[2])):
        mdp['end'].append(int(file[2][i]))

    mdp['transition'] = {state: {action: {} for action in actions} for state in states} #[[[] for i in range(mdp['numActions'])] for j in range(mdp['numStates'])]
    for i in range(3,len(file)-2):
        state = file[i][1]
        action = file[i][2]
        nextState = file[i][3]
        if nextState not in mdp['transition'][state][action].keys():
            mdp['transition'][state][action][nextState] = []
        mdp['transition'][state][action][nextState].append(float(file[i][4]), float(file[i][5]))

    mdp['discount'] = float(file[-1][1])


    numStates  = mdp['numStates']
    numActions = mdp['numActions']
    transition = mdp['transition']
    gamma      = mdp['discount']
    end_states = mdp['end']


    if pol_file != "":
        given_policy = []
        with open(pol_file) as p:
            pol_content = p.readlines()
        pol_file = [pol_content[i].split() for i in range(len(pol_content))]   
        for i in range(len(pol_file)):
            given_policy.append(pol_file[i][0])
        V0 = valueEvaluation(given_policy, numStates, numActions, transition, gamma, end_states)
        print(V0)

    if(optimal):
        opt_policy, opt_val = brute_force_search(states, actions, transition, gamma, end_states)
        print(opt_policy[0], opt_policy[1], "\n", opt_val[0], opt_val[1])
        # for k in p2.keys():
        #     print(k, p2[k])