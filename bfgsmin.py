# Load packages
import numpy as np
import scipy.stats as st
import sys
import json

# Numerical gradient function
def gr(f,param,eps=1e-07):
 K = param.shape[0]
 
 gr = np.full([K,1],np.nan)
 ej = np.eye(K)
 for k in range(K):
  gr[k,0] = (f(param + eps*ej[:,k]) - f(param - eps*ej[:,k]))/(2*eps)
 
 return(gr)

# BFGS function
def bfgsmin(f,x0,maxiter=1000,tol=np.sqrt(np.finfo(float).eps),verbose=False,hessian=False):
 import warnings
 warnings.filterwarnings("ignore")
 
 # Initialize
 x = x0
 f_val = f(x)
 f_old = f_val
 g0 = gr(f,x)
 H0 = np.eye(x.shape[0])
 g_diff = np.inf
 c1 = 1e-04
 lambd = 1
 convergence = -1
 
 # Start algorithm
 for iter in range(maxiter):
  print('Optimizing / Iter No. ' + str(int(iter)) + ' / F-Value: ' + str(round(f_val,2)) + ' / Step size: ' + str(round(lambd,6)) + ' / Diff: ' + str(round(g_diff,6)))
  lambd = 1
  
  # Check if some stopping criterion is satisfied
  if g_diff <= (1+np.abs(f_old))*tol*tol:
   convergence = 0
   print('\nOptimization complete!')
   break
  
  # Construct direction vector
  
  d = np.dot(np.linalg.solve(-H0,np.eye(H0.shape[0])),g0)
  m = np.dot(d.T,g0).flatten()
  
  # Select step to satisfy the Armijo's Rule
  while True:
   x1 = x + lambd*d.flatten()
   f1 = f(x1)
   ftest = f_val + c1*lambd*m
   if (np.isnan(f1) == False) and (f1 <= ftest):
    break
   else:
    lambd = lambd*0.2
   
  # Construct the improvement and relative gradient improvement
  s0 = lambd*d
  g1 = gr(f,x1)
  y0 = g1 - g0
  
  H1 = H0 + np.dot(y0,y0.T)/np.dot(y0.T,s0).flatten() - (np.dot(np.dot(np.dot(H0,s0),s0.T),H0))/np.dot(np.dot(s0.T,H0),s0).flatten()
  g0 = g1.copy()
  f_new = f(x1)
  g_diff = np.abs(m).flatten()[0]
  f_old = f_val.copy()
  f_val = f_new.copy()
  x = x1.copy()
  H0 = H1.copy()
  
 if iter >=maxiter:
  convergence = 2
  
 h_final = H0
 
 return({'convergence': convergence, 'iterations': iter, 'max_f': f_val, 'par_max': x, 'hessian': h_final})
 
# Print pretty tables
def bprint(object):
 for k,v in object.items():
  print('')
  print(k+':')
  print(v)
  print('')
 

 
