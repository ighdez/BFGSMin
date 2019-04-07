# Load packages
import numpy as np
import scipy.stats as st
import bfgsmin as bfgs

# Set initial setup of parameters
np.random.seed(666)
B = np.array([4,0.7])
N = 100000
K = B.shape[0]

# Construct x and y following normal linear model
s2 = 10
u = np.random.normal(0,np.sqrt(s2),N)

x0 = np.ones(N)
x1 = np.random.uniform(0,1,N)
X = np.stack((x0,x1),axis=1)

y = np.dot(X,B) + u

startv = np.array([1,1,0])

# Construct LLF
def llf(param):
 Bhat = param[0:K]
 s2hat = np.exp(param[K])
 xb = np.dot(X,Bhat)
 ll = np.log(st.norm.pdf(y-xb,0,np.sqrt(s2hat)))
 return(-sum(ll))

llf(startv)
bfgs.gr(llf,startv)

res = bfgs.bfgsmin(llf,startv)

bfgs.bprint(res)