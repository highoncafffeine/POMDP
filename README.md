# Optimal policy for partially Observable MDP state sensing cost k with r blind action  
A Partially Observable MDP is a generalization of the MDP setting where the system dynamics are determined
by an MDP, but the agent cannot directly observe the underlying state. In this setting, the agent is aware of
its initial state but is unaware of their exact state after taking any action, except when it pays a fixed cost
K, and the state is revealed to the agent. An action where the agent does not query its state is called a blind
action, and the agent is said to be in a blind state. We apply an additional constraint that the agent can
take at most r blind actions, i.e., after r consecutive blind actions, the agent is forced to pay price K and query
its state. We must find an efficient algorithm to evaluate the optimal policy for the agent in this setting.


[Report](Report.pdf)
