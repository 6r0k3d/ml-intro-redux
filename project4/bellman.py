isBadSide = [0,0,0,0,1,1,0,1,0,0,1,0]

def R(s,a):
	if a == 0:
		return s
	else:
		return 0		

def T(s, a, N):
	if a == 0:
		return [[1., -1]]
	else:
		sprimes = []
		for x in range(0,len(isBadSide)):
			if isBadSide[x] == 0:
				sprimes.append([1./N, s + x + 1])

		return sprimes				

def value_iteration(N, gamma=1.0, theta=0.001):
	print("N: ", N)
	U1 = [0.] * 8 * N
	U1.append(0)
	print("U1: ", U1)

	while 1:
		U = U1[:]
		print("U: ", U)
		for s in range(0,N*6):
					
			U[s] = max([sum([p * (R(s,a) + gamma * U1[s1]) for (p,s1) in T(s,a,N)]) for a in range(0,2)])
	
		a = U[:]
		b = U1[:]

		print("U1: ", U1)
		print
		delta = max([a - b for a, b in zip(a,b)])
		if delta < theta:
			return U
		
		U1 = U[:]	
	
print value_iteration(len(isBadSide))
