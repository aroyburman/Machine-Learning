import numpy as np
import matplotlib.pyplot as plt
m=1000000
n=1000
z = np.array([])

##For loop runs from i = 1...n 
## Z(i) = max(X)
for i in range (n):
	X = np.random.normal(0,1,m)
	z=np.append(z,max(X))

plt.hist(z)
plt.title("m= 1000000")
	
## Show the plot
plt.show()