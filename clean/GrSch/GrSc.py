import numpy as np
import random
 
np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x)},linewidth=500)

#Generate matrix here
nbas = 6

A    = np.zeros((nbas,nbas))
for i in range(nbas):
  for j in range(nbas):
    A[i,j] = random.uniform(-1.00000,1.00000)

#A[:,100] = A[:,99]
#A    = np.asarray([[0.0000,1.0000,2.0000],[1.0000,2.0000,0.0000],[2.0000,0.0000,1.0000]])

nbas = len(A[:,0])
nvalence = int(nbas/2)
A0   = np.zeros((nbas,nbas))
R    = np.zeros((nbas,nbas))
T    = np.identity(nbas)
A0[:,:] = A[:,:].copy()

print("Initial matrix A","\n",A)
R[:,:nbas] = A[:,:nbas].copy()
print("R initially partitioned","\n",R)

##Normalization 
#for j in range(nvalence):
#  norm = np.dot(R[:,j],R[:,j])
#  norm = 1.0/np.sqrt(norm)
#  R[:,j]*=norm
#  A[:,j]*=norm
#print("Normalized initial R up to nvalence","\n",R)

for j in range(nbas):
  for i in range(nbas):
    if i<j :
      proj = np.dot(A[:,j],R[:,i])
      norm = np.dot(R[:,i],R[:,i])
      T[i,j] = proj/norm
      if norm < 1.0E-10:
        print("Linear dependency found between functions i,j",i,j,norm)
        exit()
      R[:,j] = R[:,j]-R[:,i]*T[i,j]

print("Schmidt Orthogonal matrix R","\n",R)
print("Transformation T","\n",T)

R[:,0:nvalence] = A[:,0:nvalence]
print("Recovering NMB initial functions","\n",R)

#Evaluating properties
print("cols are orthonormal")
S = np.zeros((nbas,nbas))
S0 = np.zeros((nbas,nbas))
E = np.identity(nbas)
for i in range(nbas):
  for j in range(nbas): 
    S0[i,j] = np.dot(A0[:,i],A0[:,j])
    S[i,j]= np.dot(R[:,i],R[:,j])

print("Original Overlaping","\n",S0)
print("Overlaping after orthogonalization","\n",S)
