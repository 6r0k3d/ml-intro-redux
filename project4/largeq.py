import time
import random
import math
import gym
import numpy as np
import pandas as pd
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

def epsilon_greedy(qtable, state, epsilon, num_actions):
    greedy = np.random.uniform(0,1)
    
    if greedy < epsilon:
        return np.random.randint(num_actions)
    else:
        return np.argmax(qtable[state])


# Hyperparams
gamma = 0.99
alpha = 0.8
a_decay = 0.999
a_min = 0.12
epsilon = 1.0
e_decay = 0.9999
e_min = 0.01

seed = 0
np.random.seed(seed)

evnt = 'FrozenLake-v0'
size = 20
random = generate_random_map(size, 0.9)
env = gym.make(evnt, desc=random)
env.seed(seed)

world = [y for x in env.desc for y in x]

n_actions = env.action_space.n
n_states = env.observation_space.n

scores = []

qtable = np.random.rand(env.observation_space.n, env.action_space.n)
#memory = np.zeros((env.observation_space.n, env.action_space.n))
last = 0
episode = 0
start = time.time()
while 0.03 > last or pd.isnull(last):
    state = env.reset()
        
    score = 0
    reward = 0
    done = False
    while not done:
        action = epsilon_greedy(qtable, state, epsilon, env.action_space.n)
        s_prime, reward, done, info = env.step(action)
        score += reward
        if done and world[s_prime] == 'H':
            reward = -1

        qtable[state, action] += alpha * (reward + (gamma * qtable[s_prime, np.argmax(qtable[s_prime])]) - qtable[state,action])
        state = s_prime
    
    if epsilon > e_min:
        epsilon *= e_decay
    
    if alpha > a_min:
        alpha *= a_decay
    

    #print qtable
    #print "\tL\tD\t\tR\tU"    
    print episode, alpha, epsilon, last
    env.render()
    episode += 1
    scores.append(score)
    mean = pd.Series(scores).rolling(100).mean()
    last = mean.iloc[-1]
    
    
end = time.time()
scores_final = []

for _ in range(int(math.floor(len(scores)/2))):
    score = 0
    
    s = env.reset()
    
    done = False
    while not done:
        a = epsilon_greedy(qtable, s, epsilon, n_actions)
        sp, reward, done, info = env.step(a)
        score += reward
        s = sp
        
    scores_final.append(score)
    mean_final = pd.Series(scores_final).rolling(100).mean()

timer = end - start

print "-" * 70
print "Q Learner"
print evnt + " " + "Size: " + str(size)
print "Gamma: " + str(gamma)
print "Epsilon: " + str(epsilon)
print "Decay: " + str(e_decay)
print "Time: " + str(timer)
print

desc = env.desc.tolist() 
decode = [[c.decode('utf-8') for c in line] for line in desc]

pol = {0:'<', 1:'v',2:'>',3:'^'}
vis = []
for x in range(len(decode)):
    temp = []
    for y in range(len(decode[x])):
        if decode[x][y] not in 'HG':
            temp.append(pol[np.argmax(qtable[(x * 4) + y])])
        else:
            temp.append(decode[x][y])
    vis.append(temp)

for line in vis:
    print "   " + "".join(line)

plt.plot(range(len(mean.values.tolist())), mean.values.tolist(), label='Average Training Score')
plt.plot(range(len(mean_final.values.tolist())), mean_final.values.tolist(), label='Average Trained Score')
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Score', fontsize=18)
plt.legend(loc='best')
plt.show()        
        
