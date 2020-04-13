import time
import random
import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import pandas as pd
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)

evnt = 'FrozenLake-v0'
size = 8
random = generate_random_map(size)
env = gym.make(evnt, desc=random)
env.seed(seed)

n_actions = env.action_space.n
n_states = env.observation_space.n

# For Eval
deltas = []

# Hyperparams
theta = 0.01
gamma = 0.9

def policy_evaluation(states, policy):
    delta = theta * 2
    while delta > theta:
        deltas.append(delta)
        delta = 0
        for s in range(n_states):
            new_s = 0.
            for a in range(n_actions):
                for possible in env.P[s][a]:
                    prob, s_prime, reward, done = possible
                    if done:
                        new_s += policy[s,a] * prob * reward
                    else:
                        new_s += policy[s,a] * prob * (reward + gamma * states[s_prime])
                        
            delta = max(delta, np.abs(new_s-states[s]))
            states[s] = new_s
    return states

def policy_improvement(states, policy):
    policy_stable = True
    for s in range(n_states):
        old = np.argmax(policy[s])
        temp = np.zeros((n_actions))
        for a in range(n_actions):
            for possible in env.P[s][a]:
                prob, s_prime, reward, done = possible
                if done:
                    temp[a] += prob * reward
                else:
                    temp[a] += prob * (reward + gamma * states[s_prime])
        policy[s] = np.zeros((n_actions))
        policy[s, np.argmax(temp)] = 1.
        
        if old != np.argmax(policy[s]):
            policy_stable = False
    return policy_stable, states, policy

def policy_iteration(): 
    policy = np.ones((n_states, n_actions))/n_actions
           
    states = np.zeros(n_states)
    policy_stable = False
    while not policy_stable:
        # policy eval
        states = policy_evaluation(states, policy)
        # policy improvement
        policy_stable, states, policy = policy_improvement(states, policy)           
    return states, policy

if __name__ == '__main__':
    start = time.time()
    states, policy = policy_iteration()
    end = time.time()
    
    timer = end - start

    scores_final = []

    for _ in range(100):
        score = 0
        
        s = env.reset()
        done = False
        while not done:
            a = np.argmax(policy[s])
            sp, reward, done, info = env.step(a)
            score += reward
            s = sp
            
        scores_final.append(score)
        mean_final = pd.Series(scores_final).mean()

    print mean_final
    
    print "-" * 70
    print "Policy Iteration"
    print evnt + " " + "Size: " + str(size)
    print "Gamma: " + str(gamma)
    print "Theta: " + str(theta)
    print "Time: " + str(timer)
    print "deltas = " + str(deltas)
    print
    
    """
    desc = env.desc.tolist() 
    decode = [[c.decode('utf-8') for c in line] for line in desc]
    
    pol = {0:'<', 1:'v',2:'>',3:'^'}
    vis = []
    for x in range(len(decode)):
        temp = []
        for y in range(len(decode[x])):
            if decode[x][y] not in 'HG':
                temp.append(pol[np.argmax(policy[(x * 4) + y])])
            else:
                temp.append(decode[x][y])
        vis.append(temp)
    
    for line in vis:
        print "   " + "".join(line)
    
    print
   
    plt.plot(range(len(deltas)-1), deltas[1:], label='Improvements')
    plt.plot(range(len(deltas)-1), [theta] * len(deltas[1:]), label='Theta')
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Change', fontsize=18)
    plt.legend(loc='best')
    plt.show()
    """
