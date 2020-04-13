import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp
states = 50

P, R = example.forest(S=states)

#pi = mdp.QLearning(P, R, 0.99, n_iter=500000, alpha=0.3, alpha_min=0.1, epsilon_min=0.1, epsilon_decay=0.9999)
pi = mdp.ValueIteration(P, R, 0.99)

pi.run()
#print("deltas_" + str(gamma)[2:] + " = " + str(pi.deltas))

for x in pi.run_stats:
	print(x)

print(pi.policy)

l = len(pi.run_stats)-1
print('Time: ', pi.run_stats[l]['Time'], "Reward: ", pi.run_stats[l]['Reward'])

