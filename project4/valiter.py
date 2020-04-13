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
size = 22
random = generate_random_map(size)
env = gym.make(evnt, desc=random)

n_actions = env.action_space.n
n_states = env.observation_space.n

# For Eval
deltas = []

# Hyperparams
theta = 0.01
gamma = 0.9

def value_iteration(env):  
    U = [0.] * n_states
    delta = theta * 2
    while delta > theta:
        deltas.append(delta)
        delta = 0

        # Calc value of each state
        for s in range(n_states):            
            temp = np.zeros((n_actions))
            for action in range(n_actions):
                for possible in env.P[s][action]:
                    prob, s_prime, reward, done = possible
                    if done:
                        temp[action] += prob * reward
                    else:
                        temp[action] += prob * (reward + gamma * U[s_prime])

            maxval = np.max(temp)
            delta = max(delta, np.abs(maxval-U[s]))
            U[s] = maxval

    # extract policy from values
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        temp = np.zeros(n_actions)
        for action in range(n_actions):
            for possible in env.P[s][action]:
                prob, s_prime, reward, done = possible
                temp[action] += prob * (reward + gamma * U[s_prime])
        policy[s,np.argmax(temp)] = 1

    return U, policy

if __name__ == '__main__':
    start = time.time()
    states, policy = value_iteration(env)
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
    """    
    print "-" * 70
    print "Value Iteration"
    print evnt + " " + "Size: " + str(size)
    print "Gamma: " + str(gamma)
    print "Theta: " + str(theta)
    print "Time: " + str(timer)
    print "deltas = " + str(deltas)
    print
    
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
    
scores = []
for _ in range(1000):
    state = env.reset()
    score = 0   
     
    done = False
    while not done:
        action = np.argmax(policy[state])
        s_prime, reward, done, info = env.step(action)
        score += reward
        state = s_prime
    env.render()
    scores.append(score)
    
print pd.Series(scores).mean()



plt.plot(range(len(deltas)-1), deltas[1:], label='Improvements')
plt.plot(range(len(deltas)-1), [theta] * len(deltas[1:]), label='Theta')
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Change', fontsize=18)
plt.legend(loc='best')
plt.show()
"""
