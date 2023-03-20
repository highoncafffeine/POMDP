import numpy as np
import pulp
import math
import argparse


def valueEvaluation(policy, numStates, numActions, transition, gamma, end_states):
    v0 = np.zeros(numStates)
    v1 = np.zeros(numStates)
    max_error = 1e-12
    count = 0

    a0 = 0
    while(1):
        a0+=1
        for i in range(numStates):
            if i in end_states:
                continue
            v1[i] = 0
            for k in transition[i][policy[i]]:
                t = len(policy[i])
                discount = gamma**(t)
                v1[i] += (k[2]*(k[1] + (discount*v0[k[0]])))
        if np.max(np.abs(np.subtract(v0,v1))) < max_error:
            break

        v0 = np.copy(v1)
        count += 1

    return v1    

def brute_force_search(numStates, numActions, transition, gamma, end_states):
    value_functions = {}
    opt_policy = ['R', 'R']
    opt_val = [math.inf, math.inf]
    for a1 in actions:
        for a2 in actions:
            given_policy = [a1, a2]
            V0 = valueEvaluation(given_policy, numStates, numActions, transition, gamma, end_states)
            if V0[0]<opt_val[0] and V0[1]<opt_val[1]:
                opt_val = V0
                opt_policy = given_policy
            # print(given_policy, V0)
            value_functions[tuple(given_policy)] = V0

    # p1 = {k: v for k, v in sorted(value_functions.items(), key=lambda item: item[1][1])}
    # sort policies by value of state 0
    p2 = {k: v for k, v in sorted(value_functions.items(), key=lambda item: item[1][0])}

    return p2, opt_policy, opt_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp")
    parser.add_argument("--policy",default= "")
    parser.add_argument("--optimal", action=argparse.BooleanOptionalAction)
    parser.add_argument("--len_action", default = -1)
    args = parser.parse_args()

    file = args.mdp
    pol_file = args.policy
    optimal = args.optimal
    len_action  = int(args.len_action)

    with open(file) as f:
        content = f.readlines()
    file = [content[i][:-1].split() for i in range(len(content))]   


    mdp ={}
    mdp['numStates'] = int(file[0][1])
    mdp['numActions'] = int(file[1][1])

    mdp['end'] = []
    for i in range(1,len(file[2])):
        mdp['end'].append(int(file[2][i]))

    mdp['transition'] = {0:{}, 1:{}} #[[[] for i in range(mdp['numActions'])] for j in range(mdp['numStates'])]
    for i in range(3,len(file)-2):
        m = int(file[i][1])
        n = file[i][2]
        if n not in mdp['transition'][m].keys():
            mdp['transition'][m][n] = []
        mdp['transition'][m][n].append((int(file[i][3]), float(file[i][4]), float(file[i][5])))

    mdp['discount'] = float(file[-1][1])


    numStates  = mdp['numStates']
    numActions = mdp['numActions']
    transition = mdp['transition']
    gamma      = mdp['discount']
    end_states = mdp['end']

    actions = list(mdp['transition'][0].keys())
    if len_action>0:
        actions = [x for x in actions if len(x)<=len_action]

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
        p2, opt_policy, opt_val = brute_force_search(numStates, numActions, transition, gamma, end_states)
        print(opt_policy[0], opt_policy[1], "\n", opt_val[0], opt_val[1])
        # for k in p2.keys():
        #     print(k, p2[k])