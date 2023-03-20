import numpy as np
import pulp
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--K", default = None)
parser.add_argument("--len_action", default = None)
parser.add_argument("--seed", default = 0)

args = parser.parse_args()
np.random.seed(int(args.seed))

K = args.K
MAX_LEN_ACTION = args.len_action
if K == None:
    K = 0.1
else:
    K = float(K)
if MAX_LEN_ACTION == None:
    MAX_LEN_ACTION = 2
else:
    MAX_LEN_ACTION = int(MAX_LEN_ACTION)
numStates = 2               
numActions = 2

p0 = np.random.rand()
p1 = np.random.rand()
p2 = np.random.rand()
p3 = np.random.rand()
Tr= np.array([[p0, 1-p0],
              [p1, 1-p1]])

Tb = np.array([[p2, 1-p2],
               [p3, 1-p3]])

Cr = np.random.rand(2) #np.array([0,1])
Cb = np.random.rand(2) #np.array([1,0])

T = {}
T['R'] = Tr
T['B'] = Tb

C = {}
C['R'] = Cr
C['B'] = Cb

gamma = 0.5

# actions = R, B, RB, RR, BR, BB
print("numStates", 2)
print("numActions", 2**(MAX_LEN_ACTION+1) - 2)
print("end -1")
mdp = {}
mdp[0] = {}
mdp[1] = {}

def binary_to_string(num, len_action):
    binary_string = "{0:b}".format(num)
    # print(binary_string)
    binary_string = "0"*(len_action-len(binary_string))+binary_string
    # print(binary_string)
    string = ""
    for char in binary_string:
        if char == '0':
            string += 'R'
        else:
            string += 'B'
    return string
    # print(string)


for i in range(numStates):
    for ii in range(numActions):
        action = binary_to_string(ii, 1)
        mdp[i][action] = {}
        for j in range(numStates):
            mdp[i][action][j] = {}
            # print(action)
            prob = T[action][i][j]
            cost = C[action][i]
            mdp[i][action][j]['prob'] = prob
            mdp[i][action][j]['cost'] = cost
            # print(i, action, j, prob, cost)

# print(mdp)
for len_action in range(2, MAX_LEN_ACTION+1):
    for i in range(numStates):
        state = i
        for ii in range(2**len_action):
            action = binary_to_string(ii, len_action)
            mdp[i][action] = {}
            for j in range(numStates):
                next_state = j
                mdp[i][action][next_state] = {}
                prob = sum([mdp[i][action[:-1]][k]['prob']*T[action[-1]][k][j] for k in range(numStates)])
                cost = sum([mdp[i][action[:-1]][k]['prob']*(mdp[i][action[:-1]][k]['cost']+(gamma**(len_action-1))*C[action[-1]][k]) for k in range(numStates)])
                mdp[i][action][j]['prob'] = prob
                mdp[i][action][j]['cost'] = cost


for i in range(numStates):
    for action in mdp[i].keys():
        for j in range(numStates):
            state = i
            action  = action
            next_state = j
            cost = mdp[i][action][j]['cost']
            prob = mdp[i][action][j]['prob']
            print("transition ", state, " ", action, " ", next_state, " ", cost + (gamma**(len(action)))*K, " ", prob)


print("gamma ", gamma)
