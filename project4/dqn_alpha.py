import statistics
import random
import numpy as np
import pandas as pd
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

seed = 0
# Init PRNG
random.seed(seed)
np.random.seed(seed)

evnt = 'FrozenLake-v0'
size = 8
rndm = generate_random_map(size)
env = gym.make(evnt, desc=rndm)
env.seed(0)

import tensorflow as tf
tf.random.set_seed(seed)

from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def epsilon_greedy(model, state, epsilon, num_actions):
    greedy = np.random.uniform(0,1)

    if greedy < epsilon:
        return np.random.randint(num_actions)
    else:
        predictions = model.predict(np.reshape(state,(1,-1)))
        return np.argmax(predictions)
        
# Hyperparams
#alpha = 0.001
gamma = 0.99
epsilon = 1.0
e_decay = 0.9
e_min = 0.01
episodes = 2000
memory_size = 500000
batch_size = 64
update_freq = 5
c_update = 0 # for ddqn
c_steps = 2 # for ddqn
n1 = 150
n2 = 150


# Data
episode_nums = []
scores = []

# Init environment
n_actions = env.action_space.n
n_states = env.observation_space.n

# Init replay memory D to capacity N
memory = deque(maxlen=memory_size)

# Init action value function Q
q = Sequential()
q.add(Dense(n1, input_dim=n_states, activation='relu'))
q.add(Dense(n2, activation='relu'))
q.add(Dense(n_actions, activation='linear'))
q.compile(loss='mse', optimizer=Adam(lr=0.1))

last = 0
e = 0
while 0.96 > last or pd.isnull(last):
    # Init sequence
    state = env.reset()
    #print "State", state
    
    score = 0
    done = False
    
    while not done:
        #env.render()
        action = epsilon_greedy(q, state, epsilon, n_actions)

        s_prime, reward, done, info = env.step(action)
        score += reward            
        memory.append((state, action, reward, s_prime, done))
        
        #print "action", action
        #print "s_prime", s_prime
        #print "reward", reward
        #print "done", done
        #print "memory", memory[-1]
        
        state = s_prime
        #print "state change", state
        #print

        if e % update_freq == 0 and len(memory) >= batch_size:
            # Sample random minibatch of transitions
            minibatch = np.array(random.sample(memory, batch_size))
            #print "minibatch"
            #print minibatch            
            #print
            
            states = np.array(minibatch[:,0])
            actions = np.array(minibatch[:,1])
            rewards = np.array(minibatch[:,2])
            sp = np.array(minibatch[:,3])
            fin = np.array(minibatch[:,4])

            #print "sp"
            #print sp
            #print
            
            # Set y_j with Q-hat
            
            print sp.shape
            qhat_max = np.max(q.predict(sp), axis=1)
            #print "qhat_max"
            #print qhat_max
            #print

            #print "rewards"
            #print reward
            #print
            
            #print "gamma"
            #print gamma
            #print
            
            targets = np.where(fin==True,
                rewards, # y_j if done == True
                rewards + gamma * qhat_max) # y_j if done == False

            #print "target"
            #print targets    
            #print
            
            # Get current Q values
            q_target = q.predict(states)
            #print "q_target"
            #print q_target
            #print
            
            # Update Q values with y_j
            rows = np.array([x for x in range(len(states))])
            q_target[rows,actions] = targets
            #print "q_target update"
            #print q_target
            #print
            
            # Perform Gradient Descent
            q.train_on_batch(states, q_target)
            #exit()
            #raw_input("Enter...")
            
    if reward == 1 and epsilon > e_min:
        epsilon *= e_decay

    scores.append(score)
    mean = pd.Series(scores).rolling(100).mean()
    last = mean.iloc[-1]
    
    print "episode", e, len(memory),"\t", format(epsilon, '.2f'), "\t", round(score,2), "\t", last
    e += 1
    

scores_final = []

for _ in range(int(math.floor(len(scores)/2))):
    score = 0
    
    #env.render()
    s = env.reset()
    
    done = False
    while not done:
        a = epsilon_greedy(q, s, epsilon, n_actions)
        sp, reward, done, info = env.step(a)
        score += reward
        s = sp
                    
    scores_final.append(score)
    mean_final = pd.Series(scores_final).rolling(100).mean()
    
plt.plot(range(len(mean.values.tolist())), mean.values.tolist(), label='Average Training Score')
plt.plot(range(len(mean_final.values.tolist())), mean_final.values.tolist(), label='Average Trained Score')
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Score', fontsize=18)
plt.legend(loc='best')
plt.show()     

