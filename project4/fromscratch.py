import random
import numpy as np
import gym
import pandas as pd

seed = 0

evnt = 'Taxi-v2'
env = gym.make(evnt)#, is_slippery=False)
env.seed(seed)

n_actions = env.action_space.n
n_states = env.observation_space.n

gamma = 0.9
theta = 0.001

def value_iteration(env, gamma, theta):
    state_len = n_states
    action_len = n_actions
    
    delta = theta * 2
    states = np.zeros((state_len))
    while delta > theta:
        delta = 0
        for s in range(state_len):
            temp_array = np.zeros((action_len))
            for a in range(action_len):
                trans_list = env.P[s][a]
                for i in trans_list:
                    prob, next, reward, done = i
                    if done:
                        temp_array[a] += prob*reward
                    else:
                        temp_array[a] += prob*(reward+gamma*states[next])
            v_max = np.max(temp_array)
            delta = max(delta, np.abs(v_max-states[s]))
            states[s] = v_max
   
    policy = np.zeros((state_len, action_len))
    for s in range(state_len):
        temp_array = np.zeros((action_len))
        for a in range(action_len):
            transitions_list = env.P[s][a]
            for i in transitions_list:
                prob, next, reward, done = i
                temp_array[a] += prob*(reward+gamma*states[next])
        policy[s, np.argmax(temp_array)] = 1            
    
    return states, policy
    
states, policy = value_iteration(env, gamma, theta)

for x in range(len(states)):
    print x, states[x]
    
raw_input("E")

episodes_count = 100
scores, episodes = [], []

for i in range(episodes_count):
    state = env.reset()
    score = 0
    episode_len = 0
    while True:
        print [states[env.P[state][a][0][1]] for a in range(n_actions)]
        raw_input("E")    
        action = np.argmax([states[env.P[state][a][0][1]] for a in range(n_actions)])
        state, reward, done, info = env.step(action)
        score += reward
        episode_len += 1
        env.render()
        raw_input("E")
        if done:
            scores.append(score)
            episodes.append(episode_len)
            break
            
print("Avg Reward: {} Avg Length: {}".format(np.mean(scores), np.mean(episodes)))
